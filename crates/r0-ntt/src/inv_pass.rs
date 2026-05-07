//! Unified inverse-NTT pass kernel.
//!
//! Mirrors `fwd_pass.rs` (forward) but uses GS-DIF butterflies with
//! descending stride and inverse twiddles. The N^{-1} scaling is
//! applied at load time on the first pass (stage_offset == 0).
//!
//! The twiddle wg_pos logic is REVERSED relative to the forward:
//! - First inverse pass (stage_offset == 0): wg_pos contributes
//!   (high-stride stages cross chunks).
//! - Last inverse pass (stage_offset + log_pass == log_n): wg_pos
//!   doesn't contribute (low-stride stages are within a chunk).

use cubecl::prelude::*;
use r0_field::{monty_add, monty_mul, monty_sub, MontyParameters};

use crate::pass_common::reconstruct_twiddle;

/// Single inverse-NTT pass: GS-DIF butterflies for `log_pass` stages
/// starting at global inverse stage `stage_offset`, with descending
/// stride.
///
/// - `input`: source buffer (read-only for this pass).
/// - `output`: destination buffer. Must be separate from `input` for
///   non-final passes (transposed store). May alias for the final pass.
/// - `partial_twiddles`: windowed inverse twiddle table (3072 entries).
///   Built by `build_partial_inv_twiddles`.
/// - `inv_n`: single-element buffer holding N^{-1} in Montgomery form.
///   Only used when `stage_offset == 0`.
/// - `log_n`, `log_pass`, `stage_offset`, `log_wg`, `z_count`: same
///   semantics as `ntt_fwd_pass`.
#[cube(launch_unchecked)]
pub fn ntt_inv_pass<P: MontyParameters>(
    input: &Array<u32>,
    output: &mut Array<u32>,
    partial_twiddles: &Array<u32>,
    inv_n: &Array<u32>,
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
    let is_first_pass = comptime!(stage_offset == 0);

    let log_remaining = comptime!(log_n - stage_offset - log_pass);
    let num_windows = comptime!(((log_n + 8) / 10).max(1));

    let super_block = CUBE_POS_X as usize;
    let batch_offset = CUBE_POS_Y as usize * n;
    let mut shared = SharedMemory::<u32>::new(z * n_pass);
    let tid = UNIT_POS as usize;

    // -- Load (always transposed / tiled gather) -----------------------
    //
    // Reads input[local_idx * N_other + chunk_id] — the dual of the
    // forward kernel's transposed store. Z adjacent threads (adjacent
    // chunk_ids) read from Z adjacent addresses for coalescing.
    // For single-pass (N_other=1): degenerates to contiguous (identity).
    //
    // N^{-1} scaling is applied on the first pass.
    let z_mask = comptime!((z_count as usize) - 1);
    let log_z = comptime!(if z_count <= 1 { 0u32 } else { 31 - (z_count as u32).leading_zeros() });

    if comptime!(is_first_pass) {
        let n_inv_value = inv_n[0];
        let total_elems = comptime!(z * n_pass);
        let elems_per_thread = comptime!(total_elems / wg_size);
        #[unroll]
        for iter in 0..elems_per_thread {
            let flat = tid + iter * wg_size;
            let local_idx = flat >> log_z;
            let zi = flat & z_mask;
            let chunk_id = super_block * z + zi;
            shared[zi * n_pass + local_idx] =
                monty_mul::<P>(input[batch_offset + local_idx * n_other + chunk_id], n_inv_value);
        }
    } else {
        let total_elems = comptime!(z * n_pass);
        let elems_per_thread = comptime!(total_elems / wg_size);
        #[unroll]
        for iter in 0..elems_per_thread {
            let flat = tid + iter * wg_size;
            let local_idx = flat >> log_z;
            let zi = flat & z_mask;
            let chunk_id = super_block * z + zi;
            shared[zi * n_pass + local_idx] =
                input[batch_offset + local_idx * n_other + chunk_id];
        }
    }
    sync_cube();

    // -- Butterfly stages (GS-DIF, descending stride) ------------------
    #[unroll]
    for s in 0..log_pass {
        let log_d = comptime!(log_pass - 1 - s);
        let d = comptime!(1usize << (log_pass - 1 - s));
        let mask_d = comptime!((1usize << (log_pass - 1 - s)) - 1);
        let log_two_d = comptime!(log_pass - s);

        // Twiddle exponent for GS-DIF inverse stage `s` of this pass:
        //   First pass (high-stride): inner_step = 2^(log_n - log_pass + s),
        //                             wg contributes at outer_step = 2^s.
        //   Non-first passes (low-stride, after a prior transpose):
        //     inner_step = 2^(stage_offset + s); wg contribution (middle
        //     passes only) shares that same stride.
        let inner_step = comptime!(if stage_offset == 0 {
            1u32 << (log_n - log_pass + s)
        } else {
            1u32 << (stage_offset + s)
        });

        #[unroll]
        for k in 0..butt_per_thread {
            let butt_idx = tid + k * wg_size;
            let group = butt_idx >> log_d;
            let j = butt_idx & mask_d;
            let i_lo = (group << log_two_d) | j;
            let i_hi = i_lo + d;
            let j_inner = j as u32 * inner_step;

            #[unroll]
            for zi in 0..z {
                let tw_exp = if comptime!(is_first_pass) {
                    // First inverse pass: data in original order, wg
                    // contributes directly (no prior transpose).
                    let outer_step = comptime!(1u32 << s);
                    let wg_pos = (super_block * z + zi) as u32;
                    wg_pos * outer_step + j_inner
                } else if comptime!(stage_offset + log_pass == log_n) {
                    // Final inverse pass: no wg contribution.
                    j_inner
                } else {
                    // Middle inverse pass: wg shifted by log_remaining to
                    // undo prior transposes. Outer/inner share inner_step.
                    let wg_pos = (super_block * z + zi) as u32;
                    let effective_wg = wg_pos >> log_remaining;
                    effective_wg * inner_step + j_inner
                };

                let tw = reconstruct_twiddle::<P>(partial_twiddles, tw_exp, num_windows);

                let base = zi * n_pass;
                let a = shared[base + i_lo];
                let b = shared[base + i_hi];
                // GS-DIF: (a, b, w) -> (a + b, (a - b) * w)
                let sum = monty_add::<P>(a, b);
                let diff = monty_sub::<P>(a, b);
                let dt = monty_mul::<P>(diff, tw);
                shared[base + i_lo] = sum;
                shared[base + i_hi] = dt;
            }
        }
        sync_cube();
    }

    // -- Store (always contiguous) ----------------------------------------
    //
    // The dual of the forward's always-transposed-store. Each workgroup
    // writes its N_pass results to a contiguous block. For single-pass
    // (N_other=1) this is identical to the forward's transposed store
    // (both degenerate to sequential). For multi-pass, the next pass's
    // transposed load will gather the correct elements from this layout.
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
