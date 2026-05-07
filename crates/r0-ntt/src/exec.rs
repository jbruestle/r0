//! NTT executor: owns device resources and launches kernels according to a plan.

use cubecl::prelude::*;
use cubecl::server::Handle;
use r0_field::MontyParameters;

use crate::fwd_pass::ntt_fwd_pass;
use crate::inv_pass::ntt_inv_pass;
use crate::plan::{plan_heuristic, DeviceLimits, NttPlan, PassConfig};
use crate::twiddles::{build_partial_fwd_twiddles, build_partial_inv_twiddles, n_inv};
use crate::PARTIAL_TWIDDLE_LEN;

/// Default scratch buffer size: 64 MiB.
const DEFAULT_SCRATCH_BYTES: usize = 64 * 1024 * 1024;
const ELEM_BYTES: usize = 4;

/// NTT executor: owns twiddle tables and scratch memory on a device.
///
/// Construct via [`NttExec::new`], then call [`forward`]/[`inverse`] with an
/// explicit [`NttPlan`], or use [`forward_auto`]/[`inverse_auto`] for the
/// heuristic-planned convenience path.
///
/// [`forward`]: NttExec::forward
/// [`inverse`]: NttExec::inverse
/// [`forward_auto`]: NttExec::forward_auto
/// [`inverse_auto`]: NttExec::inverse_auto
pub struct NttExec<P: MontyParameters, R: Runtime> {
    client: ComputeClient<R>,
    fwd_twiddles: Vec<Handle>,
    inv_twiddles: Vec<Handle>,
    inv_n_table: Vec<Handle>,
    scratch: Handle,
    limits: DeviceLimits,
    max_log_n: u32,
    _p: core::marker::PhantomData<P>,
}

impl<P: MontyParameters, R: Runtime> NttExec<P, R> {
    /// Create a new executor, precomputing all twiddle tables and allocating
    /// scratch memory on `device`.
    ///
    /// `scratch_bytes`: scratch buffer size in bytes. Pass 0 for the default
    /// (64 MiB). Should be a power of two for clean sub-batching.
    pub fn new(device: &R::Device, scratch_bytes: usize) -> Self {
        let scratch_bytes = if scratch_bytes == 0 {
            DEFAULT_SCRATCH_BYTES
        } else {
            scratch_bytes
        };
        let max_log_n = P::TWO_ADICITY.min(24);
        let client = R::client(device);

        let props = client.properties();
        let limits = DeviceLimits {
            max_shared_mem_bytes: props.hardware.max_shared_memory_size,
            max_threads_per_wg: props.hardware.max_units_per_cube,
            scratch_bytes,
        };

        let mut fwd_twiddles = Vec::with_capacity(max_log_n as usize + 1);
        let mut inv_twiddles = Vec::with_capacity(max_log_n as usize + 1);
        let mut inv_n_table = Vec::with_capacity(max_log_n as usize + 1);

        // Index 0 is unused (no NTT of size 1), but we fill it for indexing ease.
        fwd_twiddles.push(client.empty(PARTIAL_TWIDDLE_LEN * ELEM_BYTES));
        inv_twiddles.push(client.empty(PARTIAL_TWIDDLE_LEN * ELEM_BYTES));
        inv_n_table.push(client.create_from_slice(u32::as_bytes(&[0u32])));

        for log_n in 1..=max_log_n {
            let fwd = build_partial_fwd_twiddles::<P>(log_n);
            let inv = build_partial_inv_twiddles::<P>(log_n);
            let inv_n_val = n_inv::<P>(log_n);

            fwd_twiddles.push(client.create_from_slice(u32::as_bytes(&fwd)));
            inv_twiddles.push(client.create_from_slice(u32::as_bytes(&inv)));
            inv_n_table.push(client.create_from_slice(u32::as_bytes(&[inv_n_val])));
        }

        let scratch_elems = scratch_bytes / ELEM_BYTES;
        let scratch = client.empty(scratch_elems * ELEM_BYTES);

        Self {
            client,
            fwd_twiddles,
            inv_twiddles,
            inv_n_table,
            scratch,
            limits,
            max_log_n,
            _p: core::marker::PhantomData,
        }
    }

    /// Device limits detected at construction time.
    pub fn limits(&self) -> &DeviceLimits {
        &self.limits
    }

    /// Access the underlying compute client (for sync, buffer creation, etc.).
    pub fn client(&self) -> &ComputeClient<R> {
        &self.client
    }

    /// Forward NTT (R->N) using an explicit plan.
    pub fn forward(&self, buf: &Handle, plan: &NttPlan, batch: usize) {
        assert!(
            plan.log_n >= 1 && plan.log_n <= self.max_log_n,
            "log_n={} out of range [1, {}]",
            plan.log_n,
            self.max_log_n
        );
        let tw = &self.fwd_twiddles[plan.log_n as usize];
        self.run_batched(buf, tw, None, plan, batch, Direction::Forward);
    }

    /// Inverse NTT (N->R) using an explicit plan.
    pub fn inverse(&self, buf: &Handle, plan: &NttPlan, batch: usize) {
        assert!(
            plan.log_n >= 1 && plan.log_n <= self.max_log_n,
            "log_n={} out of range [1, {}]",
            plan.log_n,
            self.max_log_n
        );
        let tw = &self.inv_twiddles[plan.log_n as usize];
        let inv_n = &self.inv_n_table[plan.log_n as usize];
        self.run_batched(buf, tw, Some(inv_n), plan, batch, Direction::Inverse);
    }

    /// Forward NTT with automatic heuristic planning.
    pub fn forward_auto(&self, buf: &Handle, log_n: u32, batch: usize) {
        let plan = plan_heuristic(log_n, batch, &self.limits);
        self.forward(buf, &plan, batch);
    }

    /// Inverse NTT with automatic heuristic planning.
    pub fn inverse_auto(&self, buf: &Handle, log_n: u32, batch: usize) {
        let plan = plan_heuristic(log_n, batch, &self.limits);
        self.inverse(buf, &plan, batch);
    }

    fn run_batched(
        &self,
        buf: &Handle,
        tw: &Handle,
        inv_n: Option<&Handle>,
        plan: &NttPlan,
        batch: usize,
        dir: Direction,
    ) {
        let n = 1usize << plan.log_n;

        // Single-pass: in-place, no scratch needed.
        if plan.passes.len() == 1 {
            let pass = &plan.passes[0];
            self.launch_pass(buf, buf, tw, inv_n, plan.log_n, pass, batch, dir);
            return;
        }

        // Multi-pass: sub-batch using scratch for ping-pong.
        assert!(plan.sub_batch > 0, "plan.sub_batch must be > 0");
        let mut remaining = batch;
        let mut offset = 0usize;

        while remaining > 0 {
            let sub_batch = remaining.min(plan.sub_batch);
            self.launch_passes(buf, offset, tw, inv_n, plan, sub_batch, dir);
            remaining -= sub_batch;
            offset += sub_batch * n;
        }
    }

    fn launch_passes(
        &self,
        buf: &Handle,
        buf_offset: usize,
        tw: &Handle,
        inv_n: Option<&Handle>,
        plan: &NttPlan,
        sub_batch: usize,
        dir: Direction,
    ) {
        let n = 1usize << plan.log_n;
        let total = sub_batch * n;
        let num_passes = plan.passes.len();

        for (pass_idx, pass) in plan.passes.iter().enumerate() {
            // Ping-pong: even passes read from user buf, write to scratch.
            //             odd passes read from scratch, write to user buf.
            let (src, dst) = if pass_idx % 2 == 0 {
                (buf, &self.scratch)
            } else {
                (&self.scratch, buf)
            };
            self.launch_pass(src, dst, tw, inv_n, plan.log_n, pass, sub_batch, dir);
        }

        // If odd number of passes, result is in scratch — copy back.
        if num_passes % 2 == 1 {
            self.copy_to_user(buf, buf_offset, total);
        }
    }

    fn launch_pass(
        &self,
        src: &Handle,
        dst: &Handle,
        tw: &Handle,
        inv_n: Option<&Handle>,
        log_n: u32,
        pass: &PassConfig,
        sub_batch: usize,
        dir: Direction,
    ) {
        let n = 1usize << log_n;
        let total = sub_batch * n;
        let n_other = 1usize << (log_n - pass.log_pass);
        let grid_x = (n_other / pass.z_count as usize) as u32;

        unsafe {
            match dir {
                Direction::Forward => {
                    ntt_fwd_pass::launch_unchecked::<P, R>(
                        &self.client,
                        CubeCount::Static(grid_x, sub_batch as u32, 1),
                        CubeDim::new_1d(1u32 << pass.log_wg),
                        ArrayArg::from_raw_parts::<u32>(src, total, 1),
                        ArrayArg::from_raw_parts::<u32>(dst, total, 1),
                        ArrayArg::from_raw_parts::<u32>(tw, PARTIAL_TWIDDLE_LEN, 1),
                        log_n,
                        pass.log_pass,
                        pass.stage_offset,
                        pass.log_wg,
                        pass.z_count,
                    )
                    .unwrap();
                }
                Direction::Inverse => {
                    let inv_n_h = inv_n.unwrap();
                    ntt_inv_pass::launch_unchecked::<P, R>(
                        &self.client,
                        CubeCount::Static(grid_x, sub_batch as u32, 1),
                        CubeDim::new_1d(1u32 << pass.log_wg),
                        ArrayArg::from_raw_parts::<u32>(src, total, 1),
                        ArrayArg::from_raw_parts::<u32>(dst, total, 1),
                        ArrayArg::from_raw_parts::<u32>(tw, PARTIAL_TWIDDLE_LEN, 1),
                        ArrayArg::from_raw_parts::<u32>(inv_n_h, 1, 1),
                        log_n,
                        pass.log_pass,
                        pass.stage_offset,
                        pass.log_wg,
                        pass.z_count,
                    )
                    .unwrap();
                }
            }
        }
    }

    fn copy_to_user(&self, buf: &Handle, _buf_offset: usize, count: usize) {
        let wg_size = 256u32;
        let grid = ((count as u32 + wg_size - 1) / wg_size, 1, 1);

        unsafe {
            copy_kernel::launch_unchecked::<R>(
                &self.client,
                CubeCount::Static(grid.0, 1, 1),
                CubeDim::new_1d(wg_size),
                ArrayArg::from_raw_parts::<u32>(&self.scratch, count, 1),
                ArrayArg::from_raw_parts::<u32>(buf, count, 1),
                count as u32,
            )
            .unwrap();
        }
    }
}

#[derive(Clone, Copy)]
enum Direction {
    Forward,
    Inverse,
}

/// Trivial device-to-device copy kernel.
#[cube(launch_unchecked)]
fn copy_kernel(src: &Array<u32>, dst: &mut Array<u32>, #[comptime] _len: u32) {
    let idx = ABSOLUTE_POS as usize;
    dst[idx] = src[idx];
}
