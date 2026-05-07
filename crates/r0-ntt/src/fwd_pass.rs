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

use crate::pass_common::reconstruct_twiddle;

/// Single forward-NTT pass: CT-DIT butterflies for `log_pass` stages
/// starting at global stage `stage_offset`.
///
/// - `input`: source buffer (read-only for this pass).
/// - `output`: destination buffer. For non-final passes this MUST be a
///   separate allocation from `input` (transposed store races with
///   concurrent reads otherwise). For the final pass (no transpose),
///   `output` may alias `input`.
/// - `partial_twiddles`: windowed twiddle table (NUM_WINDOWS * WINDOW_SIZE
///   = 3072 entries). Built by `build_partial_fwd_twiddles`.
/// - `log_n`: total transform size (N = 2^log_n).
/// - `log_pass`: number of stages this pass handles (N_pass = 2^log_pass).
/// - `stage_offset`: global stage index where this pass begins.
/// - `log_wg`: workgroup size exponent (wg_size = 2^log_wg).
/// - `z_count`: chunks per workgroup (shared mem = z_count * N_pass).
#[cube(launch_unchecked)]
pub fn ntt_fwd_pass<P: MontyParameters>(
    input: &Array<u32>,
    output: &mut Array<u32>,
    partial_twiddles: &Array<u32>,
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

    // log_remaining = stages after this pass = log_n - stage_offset - log_pass.
    // For middle passes this is > 0; for the final pass it's 0.
    let log_remaining = comptime!(log_n - stage_offset - log_pass);

    // Number of windows needed: ceil((log_n - 1) / 10). Max exponent is
    // N/2 - 1 = 2^(log_n-1) - 1, which has log_n-1 bits.
    let num_windows = comptime!(((log_n + 8) / 10).max(1));

    let super_block = CUBE_POS_X as usize;
    let batch_offset = CUBE_POS_Y as usize * n;
    let mut shared = SharedMemory::<u32>::new(z * n_pass);
    let tid = UNIT_POS as usize;

    // -- Load NTT data -------------------------------------------------
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

        // Twiddle exponent building blocks:
        // inner_step: contribution per local butterfly j
        // outer_step: contribution per effective workgroup position
        let inner_step = comptime!(1u32 << (log_n - 1 - s));
        let outer_step = comptime!(1u32 << (log_n - 1 - stage_offset - s));

        #[unroll]
        for k in 0..butt_per_thread {
            let butt_idx = tid + k * wg_size;
            let group = butt_idx >> s;
            let j = butt_idx & mask_d;
            let i_lo = (group << log_two_d) | j;
            let i_hi = i_lo + d;
            let j_inner = j as u32 * inner_step;

            #[unroll]
            for zi in 0..z {
                let tw_exp = if comptime!(stage_offset == 0) {
                    // First pass: only local j contributes.
                    j_inner
                } else {
                    // Non-first pass: wg contributes, shifted by log_remaining
                    // to account for prior transposes.
                    let wg_pos = (super_block * z + zi) as u32;
                    let effective_wg = wg_pos >> log_remaining;
                    effective_wg * outer_step + j_inner
                };

                let tw = reconstruct_twiddle::<P>(partial_twiddles, tw_exp, num_windows);

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

    // -- Store (always transposed) ----------------------------------------
    //
    // Tiled transposed store: output[local_idx * N_other + chunk_id].
    // For multi-pass: creates the layout the next pass loads contiguously.
    // For the final pass: this IS the natural output order (since
    // N_other = N/N_pass and the position wg + j*N_other = natural position).
    // For single-pass (N_other=1): degenerates to contiguous (identity).
    //
    // Z adjacent threads write to Z adjacent addresses for coalescing.
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
}
