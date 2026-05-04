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

/// Reconstruct a twiddle factor w^k from the windowed partial twiddle table.
///
/// `partial_twiddles` has layout [NUM_WINDOWS][WINDOW_SIZE] (5 windows of 64).
/// Decomposes `k` into 6-bit windows and multiplies the corresponding entries.
#[cube]
fn reconstruct_twiddle<P: MontyParameters>(
    partial_twiddles: &Array<u32>,
    k: u32,
    #[comptime] num_windows: u32,
) -> u32 {
    let lg_window = comptime!(6u32);
    let window_mask = comptime!((1u32 << 6) - 1);
    let window_size = comptime!(64usize);

    // Start with identity (partial[0][0] = 1 in monty form).
    let mut acc = partial_twiddles[0usize]; // monty(1)

    #[unroll]
    for w in 0..num_windows {
        let k_w = (k >> (w * lg_window)) & window_mask;
        let idx = comptime!(w as usize) * window_size + k_w as usize;
        let entry = partial_twiddles[idx];
        if k_w != 0u32 {
            acc = monty_mul::<P>(acc, entry);
        }
    }
    acc
}

/// Single forward-NTT pass: CT-DIT butterflies for `log_pass` stages
/// starting at global stage `stage_offset`.
///
/// - `input`: source buffer (read-only for this pass).
/// - `output`: destination buffer. For non-final passes this MUST be a
///   separate allocation from `input` (transposed store races with
///   concurrent reads otherwise). For the final pass (no transpose),
///   `output` may alias `input`.
/// - `partial_twiddles`: windowed twiddle table (NUM_WINDOWS * WINDOW_SIZE
///   = 320 entries). Built by `build_partial_fwd_twiddles`.
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
    let transpose_out = comptime!(stage_offset + log_pass < log_n);

    // log_remaining = stages after this pass = log_n - stage_offset - log_pass.
    // For middle passes this is > 0; for the final pass it's 0.
    let log_remaining = comptime!(log_n - stage_offset - log_pass);

    // Number of windows needed for reconstruction (could be fewer for
    // small log_n, but 5 covers all cases up to 30-bit exponents).
    let num_windows = comptime!(5u32);

    let super_block = CUBE_POS_X as usize;
    let batch_offset = CUBE_POS_Y as usize * n;
    let mut shared = SharedMemory::<u32>::new(z * n_pass);
    let tid = UNIT_POS as usize;

    // -- Load ----------------------------------------------------------
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

    // -- Store ---------------------------------------------------------
    if comptime!(transpose_out) {
        // Non-final pass: tiled transposed store into `output`.
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
