//! NTT executor: owns device resources and launches kernels according to a plan.

use cubecl::prelude::*;
use cubecl::server::Handle;
use r0_cube::Device;
use r0_field::{ExtField, MontyParameters};

use crate::fwd_pass::ntt_fwd_pass;
use crate::inv_pass::ntt_inv_pass;
use crate::plan::{plan_heuristic, DeviceLimits, NttPlan, PassConfig};
use crate::twiddles::{
    build_partial_fwd_twiddles, build_partial_inv_twiddles, n_inv, PARTIAL_TWIDDLE_LEN,
};

const ELEM_BYTES: usize = 4;

/// Device-resident NTT executor.
///
/// Owns precomputed forward and inverse twiddle tables (one per
/// supported `log_n`) and a scratch buffer used for multi-pass
/// ping-pong. Construct one per `(device, field)` and reuse it across
/// calls — both setup and twiddle upload are nontrivial.
///
/// `P` selects the field
/// ([`BabyBearParameters`](r0_field::BabyBearParameters) or
/// [`KoalaBearParameters`](r0_field::KoalaBearParameters)). `R` selects
/// the cubecl runtime (`CudaRuntime`, `WgpuRuntime`, `CpuRuntime`).
///
/// Run an NTT with [`forward`] or [`inverse`]; both pick a plan
/// internally via the heuristic. For autotuning or fine-grained
/// control, enable the `unstable-planner` feature and use
/// `forward_with_plan` / `inverse_with_plan`.
///
/// [`forward`]: NttExec::forward
/// [`inverse`]: NttExec::inverse
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
    /// Construct an executor on `device`, precomputing twiddle tables for
    /// every supported `log_n` (1..=24, capped by `P::TWO_ADICITY`).
    ///
    /// Multi-pass NTTs ping-pong through the shared scratch buffer owned
    /// by the [`Device`]; size that buffer with
    /// [`Device::acquire_with_scratch`](Device::acquire_with_scratch) /
    /// [`acquire_with_scratch_for`](Device::acquire_with_scratch_for) when
    /// the default 64 MiB isn't enough. Each in-flight polynomial of size
    /// `2^log_n` consumes `4 << log_n` bytes, which sets the per-launch
    /// sub-batch ceiling for multi-pass plans; smaller scratch just means
    /// more launch iterations.
    ///
    /// Setup touches the device (uploads twiddles), so build one and
    /// reuse it.
    pub fn new(device: &Device<R>) -> Self {
        let max_log_n = P::TWO_ADICITY.min(24);
        let client = device.client().clone();

        let props = client.properties();
        let limits = DeviceLimits {
            max_shared_mem_bytes: props.hardware.max_shared_memory_size,
            max_threads_per_wg: props.hardware.max_units_per_cube,
            scratch_bytes: device.scratch_bytes(),
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

        let scratch = device.scratch().clone();

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

    /// Device limits the planner sees (shared memory, max threads per
    /// workgroup, scratch bytes). Gated behind `unstable-planner`;
    /// callers using the heuristic path don't need this.
    #[cfg(feature = "unstable-planner")]
    pub fn limits(&self) -> &DeviceLimits {
        &self.limits
    }

    /// Underlying cubecl compute client. Use for buffer creation
    /// (`create_from_slice`, `empty`), sync, and readback.
    pub fn client(&self) -> &ComputeClient<R> {
        &self.client
    }

    /// Forward NTT (R→N) of `batch` polynomials of size `2^log_n`,
    /// in place on `buf`.
    ///
    /// `buf` holds `batch * 2^log_n` Montgomery-form `u32`s laid out
    /// as `batch` contiguous polynomials in bit-reversed coefficient
    /// order. On return, each polynomial holds its natural-order
    /// evaluations. The plan is built by the internal heuristic.
    ///
    /// # Panics
    ///
    /// Panics if `log_n` is outside `[1, P::TWO_ADICITY.min(24)]`.
    pub fn forward(&self, buf: &Handle, log_n: u32, batch: usize) {
        let plan = plan_heuristic(log_n, batch, &self.limits);
        self.run_forward(buf, &plan, batch);
    }

    /// Inverse NTT (N→R) of `batch` polynomials of size `2^log_n`,
    /// in place on `buf`.
    ///
    /// Reads natural-order evaluations and writes bit-reversed
    /// coefficients (with the `N^{-1}` scaling already applied). The
    /// plan is built by the internal heuristic.
    ///
    /// # Panics
    ///
    /// Panics if `log_n` is outside `[1, P::TWO_ADICITY.min(24)]`.
    pub fn inverse(&self, buf: &Handle, log_n: u32, batch: usize) {
        let plan = plan_heuristic(log_n, batch, &self.limits);
        self.run_inverse(buf, &plan, batch);
    }

    /// Forward NTT for `batch` polynomials of length `2^log_n` whose
    /// elements are in the field `F`, where `F` is the base field `P` or
    /// any extension over `P` laid out transposed (component-major within
    /// each polynomial).
    ///
    /// Equivalent to [`forward(buf, log_n, batch * F::DEGREE)`](Self::forward) —
    /// a degree-`D` extension polynomial of length `N` is bitwise identical
    /// to `D` consecutive base-field polynomials of length `N`, so the NTT
    /// works as-is. This method exists to bind the extension type at the
    /// call site so the type system catches `BabyBear4` fed to a
    /// `KoalaBear` executor before it silently corrupts data.
    ///
    /// # Panics
    /// Panics if `log_n` is outside `[1, P::TWO_ADICITY.min(24)]`.
    pub fn forward_ext<F: ExtField<Base = P>>(
        &self,
        buf: &Handle,
        log_n: u32,
        batch: usize,
    ) {
        self.forward(buf, log_n, batch * F::DEGREE as usize);
    }

    /// Inverse NTT counterpart to [`forward_ext`](Self::forward_ext).
    ///
    /// # Panics
    /// Panics if `log_n` is outside `[1, P::TWO_ADICITY.min(24)]`.
    pub fn inverse_ext<F: ExtField<Base = P>>(
        &self,
        buf: &Handle,
        log_n: u32,
        batch: usize,
    ) {
        self.inverse(buf, log_n, batch * F::DEGREE as usize);
    }

    /// Forward NTT (R→N) using an explicit [`NttPlan`].
    ///
    /// Gated behind `unstable-planner`. Prefer [`forward`] unless you
    /// have a hand-tuned or autotuned plan you want to lock in.
    /// Build plans via [`crate::plan_heuristic`] or
    /// [`crate::enumerate_valid_plans`].
    ///
    /// [`forward`]: NttExec::forward
    #[cfg(feature = "unstable-planner")]
    pub fn forward_with_plan(&self, buf: &Handle, plan: &NttPlan, batch: usize) {
        self.run_forward(buf, plan, batch);
    }

    /// Inverse NTT (N→R) using an explicit [`NttPlan`].
    /// See [`forward_with_plan`] for caveats.
    ///
    /// [`forward_with_plan`]: NttExec::forward_with_plan
    #[cfg(feature = "unstable-planner")]
    pub fn inverse_with_plan(&self, buf: &Handle, plan: &NttPlan, batch: usize) {
        self.run_inverse(buf, plan, batch);
    }

    fn run_forward(&self, buf: &Handle, plan: &NttPlan, batch: usize) {
        assert!(
            plan.log_n >= 1 && plan.log_n <= self.max_log_n,
            "log_n={} out of range [1, {}]",
            plan.log_n,
            self.max_log_n
        );
        let tw = &self.fwd_twiddles[plan.log_n as usize];
        self.run_batched(buf, tw, None, plan, batch, Direction::Forward);
    }

    fn run_inverse(&self, buf: &Handle, plan: &NttPlan, batch: usize) {
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

        // Slice the user buffer down to the current sub-batch so kernels
        // (which index from CUBE_POS_Y * n internally) hit the right rows.
        // Cloning a Handle is shallow — just metadata, no data copy.
        let buf_slice = buf
            .clone()
            .offset_start((buf_offset * ELEM_BYTES) as u64);

        for (pass_idx, pass) in plan.passes.iter().enumerate() {
            // Ping-pong: even passes read from user buf, write to scratch.
            //             odd passes read from scratch, write to user buf.
            let (src, dst) = if pass_idx % 2 == 0 {
                (&buf_slice, &self.scratch)
            } else {
                (&self.scratch, &buf_slice)
            };
            self.launch_pass(src, dst, tw, inv_n, plan.log_n, pass, sub_batch, dir);
        }

        // If odd number of passes, result is in scratch — copy back.
        if num_passes % 2 == 1 {
            self.copy_to_user(&buf_slice, total);
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

    fn copy_to_user(&self, dst: &Handle, count: usize) {
        let wg_size = 256u32;
        let grid = ((count as u32 + wg_size - 1) / wg_size, 1, 1);

        unsafe {
            copy_kernel::launch_unchecked::<R>(
                &self.client,
                CubeCount::Static(grid.0, 1, 1),
                CubeDim::new_1d(wg_size),
                ArrayArg::from_raw_parts::<u32>(&self.scratch, count, 1),
                ArrayArg::from_raw_parts::<u32>(dst, count, 1),
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
