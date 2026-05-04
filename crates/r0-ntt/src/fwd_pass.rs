//! Unified forward-NTT pass kernel.
//!
//! A single `#[cube]` kernel handles every pass of a multi-pass forward
//! NTT. The host calls it once per pass with different
//! `(log_pass, stage_offset)` pairs that tile the `log_n` stages.
//!
//! Non-final passes (`stage_offset + log_pass < log_n`) store results
//! in a transposed layout (into a separate output buffer) so the next
//! pass loads contiguously. The final pass stores contiguously and may
//! write back to the same buffer it read from (in-place safe).
//!
//! Each workgroup processes `z_count` independent chunks of
//! `2^log_pass` elements, amortizing twiddle loads and barrier cost.
//!
//! Grid: `(N / (N_pass * z_count), batch_size, 1)`.

use cubecl::prelude::*;
use r0_field::{monty_add, monty_mul, monty_sub, MontyParameters};

/// Single forward-NTT pass: CT-DIT butterflies for `log_pass` stages
/// starting at global stage `stage_offset`.
///
/// - `input`: source buffer (read-only for this pass).
/// - `output`: destination buffer. For non-final passes this MUST be a
///   separate allocation from `input` (transposed store races with
///   concurrent reads otherwise). For the final pass (no transpose),
///   `output` may alias `input`.
/// - `log_n`: total transform size (N = 2^log_n).
/// - `log_pass`: number of stages this pass handles (N_pass = 2^log_pass).
/// - `stage_offset`: global stage index where this pass begins.
/// - `log_wg`: workgroup size exponent (wg_size = 2^log_wg).
/// - `z_count`: chunks per workgroup (shared mem = z_count * N_pass).
#[cube(launch_unchecked)]
pub fn ntt_fwd_pass<P: MontyParameters>(
    input: &Array<u32>,
    output: &mut Array<u32>,
    twiddles: &Array<u32>,
    #[comptime] log_n: u32,
    #[comptime] log_pass: u32,
    #[comptime] stage_offset: u32,
    #[comptime] log_wg: u32,
    #[comptime] z_count: u32,
) {
    let n = comptime!(1usize << log_n);
    let n_pass = comptime!(1usize << log_pass);
    let n_other = comptime!(1usize << (log_n - log_pass));
    let wg_size = comptime!(1usize << log_wg);
    let loads_per_thread = comptime!(1usize << (log_pass - log_wg));
    let butt_per_thread = comptime!(1usize << (log_pass - 1 - log_wg));
    let z = comptime!(z_count as usize);
    let transpose_out = comptime!(stage_offset + log_pass < log_n);

    let super_block = CUBE_POS_X as usize;
    let batch_offset = CUBE_POS_Y as usize * n;
    let mut shared = SharedMemory::<u32>::new(z * n_pass);
    let tid = UNIT_POS as usize;

    // -- Load ----------------------------------------------------------
    //
    // All passes load contiguously from `input`. First pass reads from
    // the original layout; subsequent passes read from the transposed
    // layout written by the previous pass into this buffer.
    #[unroll]
    for zi in 0..z {
        let slab_base = batch_offset + (super_block * z + zi) * n_pass;
        #[unroll]
        for k in 0..loads_per_thread {
            let local_idx = tid + k * wg_size;
            shared[zi * n_pass + local_idx] = input[slab_base + local_idx];
        }
    }
    sync_cube();

    // -- Butterfly stages ----------------------------------------------
    #[unroll]
    for s in 0..log_pass {
        let d = comptime!(1usize << s);
        let mask_d = comptime!((1usize << s) - 1);
        let log_two_d = comptime!(s + 1);

        let inner_step = comptime!(1usize << (log_n - s - 1));
        let outer_step = comptime!(1usize << (log_n - stage_offset - s - 1));

        #[unroll]
        for k in 0..butt_per_thread {
            let butt_idx = tid + k * wg_size;
            let group = butt_idx >> s;
            let j = butt_idx & mask_d;
            let i_lo = (group << log_two_d) | j;
            let i_hi = i_lo + d;
            let j_inner = j * inner_step;

            #[unroll]
            for zi in 0..z {
                let tw = if comptime!(stage_offset == 0) {
                    twiddles[j_inner]
                } else {
                    let wg_pos = super_block * z + zi;
                    twiddles[wg_pos * outer_step + j_inner]
                };

                let base = zi * n_pass;
                let a = shared[base + i_lo];
                let b = shared[base + i_hi];
                let t = monty_mul::<P>(tw, b);
                shared[base + i_lo] = monty_add::<P>(a, t);
                shared[base + i_hi] = monty_sub::<P>(a, t);
            }
        }
        sync_cube();
    }

    // -- Store ---------------------------------------------------------
    if comptime!(transpose_out) {
        // Non-final pass: tiled transposed store into `output`.
        //
        // Remap thread->element so Z adjacent threads write to Z
        // adjacent global addresses:
        //   flat = L * Z + zi  (assigned round-robin across threads)
        // Within a 32-thread warp, groups of Z write consecutively.
        //
        // SAFETY: `output` must be a separate buffer from `input`.
        // The transposed write pattern aliases other workgroups' read
        // regions, so in-place operation would race.
        let stores_per_thread = comptime!((z_count as usize) * (1usize << (log_pass - log_wg)));
        let z_mask = comptime!((z_count as usize) - 1);
        let log_z = comptime!(if z_count <= 1 { 0u32 } else { 31 - (z_count as u32).leading_zeros() });

        #[unroll]
        for iter in 0..stores_per_thread {
            let flat = tid + iter * wg_size;
            let local_idx = flat >> log_z;
            let zi = flat & z_mask;
            let chunk_id = super_block * z + zi;
            output[batch_offset + local_idx * n_other + chunk_id] =
                shared[zi * n_pass + local_idx];
        }
    } else {
        // Final pass: contiguous store into `output`.
        // Safe to alias input (reads and writes target the same positions).
        #[unroll]
        for zi in 0..z {
            let slab_base = batch_offset + (super_block * z + zi) * n_pass;
            #[unroll]
            for k in 0..loads_per_thread {
                let local_idx = tid + k * wg_size;
                output[slab_base + local_idx] = shared[zi * n_pass + local_idx];
            }
        }
    }
}
