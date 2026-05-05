//! High-level NTT planner: precomputes twiddle tables, manages scratch
//! memory, and exposes a simple `forward`/`inverse` API.

use cubecl::prelude::*;
use cubecl::server::Handle;
use r0_field::MontyParameters;

use crate::fwd_pass::ntt_fwd_pass;
use crate::inv_pass::ntt_inv_pass;
use crate::twiddles::{build_partial_fwd_twiddles, build_partial_inv_twiddles, n_inv};
use crate::PARTIAL_TWIDDLE_LEN;

/// Default scratch buffer size: 64 MiB (16M u32 elements).
const DEFAULT_SCRATCH_ELEMS: usize = 16 * 1024 * 1024;

/// Precomputed NTT state for a specific field and device.
///
/// Holds twiddle tables and a scratch buffer on-device. Call [`forward`] and
/// [`inverse`] to perform transforms. Thread-safe (immutable after construction).
///
/// [`forward`]: NttPlanner::forward
/// [`inverse`]: NttPlanner::inverse
pub struct NttPlanner<P: MontyParameters, R: Runtime> {
    client: ComputeClient<R>,
    /// Forward partial twiddle tables, indexed by log_n (1..=max_log_n).
    fwd_twiddles: Vec<Handle>,
    /// Inverse partial twiddle tables, indexed by log_n (1..=max_log_n).
    inv_twiddles: Vec<Handle>,
    /// inv_n[i] = (2^i)^{-1} mod p in Montgomery form, for i = 0..=max_log_n.
    inv_n_table: Vec<Handle>,
    /// Scratch buffer for ping-pong.
    scratch: Handle,
    /// Scratch capacity in u32 elements.
    scratch_len: usize,
    /// Maximum supported log_n (min of field's TWO_ADICITY and 24).
    max_log_n: u32,
    _p: core::marker::PhantomData<P>,
}

impl<P: MontyParameters, R: Runtime> NttPlanner<P, R> {
    /// Create a new planner, precomputing all twiddle tables and allocating
    /// a scratch buffer on `device`.
    ///
    /// `scratch_elems`: size of scratch buffer in u32 elements. Pass 0 for
    /// the default (64 MiB = 16M elements).
    pub fn new(device: &R::Device, scratch_elems: usize) -> Self {
        let scratch_len = if scratch_elems == 0 {
            DEFAULT_SCRATCH_ELEMS
        } else {
            scratch_elems
        };
        let max_log_n = P::TWO_ADICITY.min(24);
        let client = R::client(device);

        // Precompute twiddle tables for all supported sizes.
        let mut fwd_twiddles = Vec::with_capacity(max_log_n as usize + 1);
        let mut inv_twiddles = Vec::with_capacity(max_log_n as usize + 1);
        let mut inv_n_table = Vec::with_capacity(max_log_n as usize + 1);

        // Index 0 is unused (no NTT of size 1), but we fill it for indexing ease.
        fwd_twiddles.push(client.empty(PARTIAL_TWIDDLE_LEN * 4));
        inv_twiddles.push(client.empty(PARTIAL_TWIDDLE_LEN * 4));
        inv_n_table.push(client.create_from_slice(u32::as_bytes(&[0u32])));

        for log_n in 1..=max_log_n {
            let fwd = build_partial_fwd_twiddles::<P>(log_n);
            let inv = build_partial_inv_twiddles::<P>(log_n);
            let inv_n_val = n_inv::<P>(log_n);

            fwd_twiddles.push(client.create_from_slice(u32::as_bytes(&fwd)));
            inv_twiddles.push(client.create_from_slice(u32::as_bytes(&inv)));
            inv_n_table.push(client.create_from_slice(u32::as_bytes(&[inv_n_val])));
        }

        let scratch = client.empty(scratch_len * 4);

        Self {
            client,
            fwd_twiddles,
            inv_twiddles,
            inv_n_table,
            scratch,
            scratch_len,
            max_log_n,
            _p: core::marker::PhantomData,
        }
    }

    /// Forward NTT (R→N): bit-reversed coefficients in → natural evaluations out.
    ///
    /// `buf`: device buffer containing `batch` polynomials of `2^log_n` u32
    /// elements each, in bit-reversed order. Transformed in-place to natural order.
    ///
    /// # Panics
    /// - If `log_n` exceeds the field's supported range.
    /// - If the scratch buffer cannot hold even one polynomial (`2^log_n > scratch_len`).
    pub fn forward(&self, buf: &Handle, log_n: u32, batch: usize) {
        assert!(log_n >= 1 && log_n <= self.max_log_n, "log_n={log_n} out of range");
        let n = 1usize << log_n;
        if log_n > 10 {
            assert!(n <= self.scratch_len, "scratch too small for log_n={log_n}");
        }

        let tw = &self.fwd_twiddles[log_n as usize];
        self.run_batched(buf, tw, None, log_n, batch, Direction::Forward);
    }

    /// Inverse NTT (N→R): natural evaluations in → bit-reversed coefficients out.
    ///
    /// `buf`: device buffer containing `batch` polynomials of `2^log_n` u32
    /// elements each, in natural order. Transformed in-place to bit-reversed order.
    ///
    /// # Panics
    /// - If `log_n` exceeds the field's supported range.
    /// - If the scratch buffer cannot hold even one polynomial.
    pub fn inverse(&self, buf: &Handle, log_n: u32, batch: usize) {
        assert!(log_n >= 1 && log_n <= self.max_log_n, "log_n={log_n} out of range");
        let n = 1usize << log_n;
        if log_n > 10 {
            assert!(n <= self.scratch_len, "scratch too small for log_n={log_n}");
        }

        let tw = &self.inv_twiddles[log_n as usize];
        let inv_n = &self.inv_n_table[log_n as usize];
        self.run_batched(buf, tw, Some(inv_n), log_n, batch, Direction::Inverse);
    }

    fn run_batched(
        &self,
        buf: &Handle,
        tw: &Handle,
        inv_n: Option<&Handle>,
        log_n: u32,
        batch: usize,
        dir: Direction,
    ) {
        let n = 1usize << log_n;

        // Single-pass: in-place, no scratch needed.
        if log_n <= 10 {
            self.launch_single_pass(buf, tw, inv_n, log_n, batch, dir);
            return;
        }

        // Multi-pass: determine sub-batch size based on scratch capacity.
        let max_sub_batch = self.scratch_len / n;
        let mut remaining = batch;
        let mut offset = 0usize; // in u32 elements

        while remaining > 0 {
            let sub_batch = remaining.min(max_sub_batch);
            self.launch_multi_pass(buf, offset, tw, inv_n, log_n, sub_batch, dir);
            remaining -= sub_batch;
            offset += sub_batch * n;
        }
    }

    fn launch_single_pass(
        &self,
        buf: &Handle,
        tw: &Handle,
        inv_n: Option<&Handle>,
        log_n: u32,
        batch: usize,
        dir: Direction,
    ) {
        let n = 1usize << log_n;
        let total = batch * n;
        let log_wg = pick_log_wg(log_n);

        unsafe {
            match dir {
                Direction::Forward => {
                    ntt_fwd_pass::launch_unchecked::<P, R>(
                        &self.client,
                        CubeCount::Static(1, batch as u32, 1),
                        CubeDim::new_1d(1u32 << log_wg),
                        ArrayArg::from_raw_parts::<u32>(buf, total, 1),
                        ArrayArg::from_raw_parts::<u32>(buf, total, 1),
                        ArrayArg::from_raw_parts::<u32>(tw, PARTIAL_TWIDDLE_LEN, 1),
                        log_n,
                        log_n,
                        0u32,
                        log_wg,
                        1u32,
                    )
                    .unwrap();
                }
                Direction::Inverse => {
                    let inv_n = inv_n.unwrap();
                    ntt_inv_pass::launch_unchecked::<P, R>(
                        &self.client,
                        CubeCount::Static(1, batch as u32, 1),
                        CubeDim::new_1d(1u32 << log_wg),
                        ArrayArg::from_raw_parts::<u32>(buf, total, 1),
                        ArrayArg::from_raw_parts::<u32>(buf, total, 1),
                        ArrayArg::from_raw_parts::<u32>(tw, PARTIAL_TWIDDLE_LEN, 1),
                        ArrayArg::from_raw_parts::<u32>(inv_n, 1, 1),
                        log_n,
                        log_n,
                        0u32,
                        log_wg,
                        1u32,
                    )
                    .unwrap();
                }
            }
        }
    }

    fn launch_multi_pass(
        &self,
        buf: &Handle,
        buf_offset: usize,
        tw: &Handle,
        inv_n: Option<&Handle>,
        log_n: u32,
        sub_batch: usize,
        dir: Direction,
    ) {
        let n = 1usize << log_n;
        let total = sub_batch * n;

        // Determine pass decomposition.
        let (pass_sizes, num_passes) = decompose_passes(log_n);

        // For multi-pass, we ping-pong between the user's buffer (at offset)
        // and the scratch buffer. The user buffer slice starts at buf_offset.
        //
        // Forward: pass k reads from A, writes to B (A and B alternate).
        //   Pass 1: user → scratch
        //   Pass 2: scratch → user
        //   Pass 3: user → scratch
        //
        // Inverse: same ping-pong pattern.
        //
        // After even passes: result in user buffer. After odd passes: result in scratch.
        // If odd number of passes: need to copy scratch → user.

        let z: u32 = 8;
        let mut stage_offset: u32 = 0;

        for pass_idx in 0..num_passes {
            let log_pass = pass_sizes[pass_idx];
            let n_other = n >> log_pass;
            let grid_x = (n_other / z as usize) as u32;
            let log_wg = pick_log_wg(log_pass);

            // Determine source and destination for this pass.
            let (src, dst) = if pass_idx % 2 == 0 {
                (buf, &self.scratch) // even pass: user → scratch
            } else {
                (&self.scratch, buf) // odd pass: scratch → user
            };

            unsafe {
                match dir {
                    Direction::Forward => {
                        ntt_fwd_pass::launch_unchecked::<P, R>(
                            &self.client,
                            CubeCount::Static(grid_x, sub_batch as u32, 1),
                            CubeDim::new_1d(1u32 << log_wg),
                            ArrayArg::from_raw_parts::<u32>(src, total, 1),
                            ArrayArg::from_raw_parts::<u32>(dst, total, 1),
                            ArrayArg::from_raw_parts::<u32>(tw, PARTIAL_TWIDDLE_LEN, 1),
                            log_n,
                            log_pass,
                            stage_offset,
                            log_wg,
                            z,
                        )
                        .unwrap();
                    }
                    Direction::Inverse => {
                        let inv_n_h = inv_n.unwrap();
                        ntt_inv_pass::launch_unchecked::<P, R>(
                            &self.client,
                            CubeCount::Static(grid_x, sub_batch as u32, 1),
                            CubeDim::new_1d(1u32 << log_wg),
                            ArrayArg::from_raw_parts::<u32>(src, total, 1),
                            ArrayArg::from_raw_parts::<u32>(dst, total, 1),
                            ArrayArg::from_raw_parts::<u32>(tw, PARTIAL_TWIDDLE_LEN, 1),
                            ArrayArg::from_raw_parts::<u32>(inv_n_h, 1, 1),
                            log_n,
                            log_pass,
                            stage_offset,
                            log_wg,
                            z,
                        )
                        .unwrap();
                    }
                }
            }

            stage_offset += log_pass;
        }

        // If odd number of passes, result is in scratch — copy back to user.
        if num_passes % 2 == 1 {
            self.copy_to_user(buf, buf_offset, total);
        }
    }

    /// Device-to-device copy from scratch buffer into user buffer at the given
    /// element offset. Uses a trivial copy kernel.
    fn copy_to_user(&self, buf: &Handle, buf_offset: usize, count: usize) {
        // For now, use a simple element-wise copy kernel.
        let total = count;
        let wg_size = 256u32;
        let grid = ((total as u32 + wg_size - 1) / wg_size, 1, 1);

        unsafe {
            copy_kernel::launch_unchecked::<R>(
                &self.client,
                CubeCount::Static(grid.0, 1, 1),
                CubeDim::new_1d(wg_size),
                ArrayArg::from_raw_parts::<u32>(&self.scratch, total, 1),
                ArrayArg::from_raw_parts::<u32>(buf, total, 1),
                total as u32,
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

/// Decompose log_n into pass sizes. Returns (sizes, count).
fn decompose_passes(log_n: u32) -> ([u32; 4], usize) {
    let mut sizes = [0u32; 4];
    if log_n <= 10 {
        sizes[0] = log_n;
        (sizes, 1)
    } else if log_n <= 20 {
        let half = log_n / 2;
        sizes[0] = half;
        sizes[1] = log_n - half;
        (sizes, 2)
    } else {
        let third = log_n / 3;
        let rem = log_n % 3;
        sizes[0] = third + if rem > 0 { 1 } else { 0 };
        sizes[1] = third + if rem > 1 { 1 } else { 0 };
        sizes[2] = third;
        (sizes, 3)
    }
}

fn pick_log_wg(log_pass: u32) -> u32 {
    log_pass.saturating_sub(1).min(8)
}

/// Trivial device-to-device copy kernel.
#[cube(launch_unchecked)]
fn copy_kernel(src: &Array<u32>, dst: &mut Array<u32>, #[comptime] _len: u32) {
    let idx = ABSOLUTE_POS as usize;
    dst[idx] = src[idx];
}
