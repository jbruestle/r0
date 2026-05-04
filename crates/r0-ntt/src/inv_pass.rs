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

/// Single inverse-NTT pass: GS-DIF butterflies for `log_pass` stages
/// starting at global inverse stage `stage_offset`, with descending
/// stride.
///
/// - `input`: source buffer (read-only for this pass).
/// - `output`: destination buffer. Must be separate from `input` for
///   non-final passes (transposed store). May alias for the final pass.
/// - `inv_twiddles`: inverse twiddle table (w^{-1} powers).
/// - `inv_n`: single-element buffer holding N^{-1} in Montgomery form.
///   Only used when `stage_offset == 0`.
/// - `log_n`, `log_pass`, `stage_offset`, `log_wg`, `z_count`: same
///   semantics as `ntt_fwd_pass`.
#[cube(launch_unchecked)]
pub fn ntt_inv_pass<P: MontyParameters>(
    input: &Array<u32>,
    output: &mut Array<u32>,
    inv_twiddles: &Array<u32>,
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
    let transpose_out = comptime!(stage_offset + log_pass < log_n);
    let is_first_pass = comptime!(stage_offset == 0);

    let super_block = CUBE_POS_X as usize;
    let batch_offset = CUBE_POS_Y as usize * n;
    let mut shared = SharedMemory::<u32>::new(z * n_pass);
    let tid = UNIT_POS as usize;

    // -- Load ----------------------------------------------------------
    //
    // First pass pre-multiplies by N^{-1} (folding the scaling into the
    // load to avoid a separate pass).
    if comptime!(is_first_pass) {
        let n_inv_value = inv_n[0];
        #[unroll]
        for zi in 0..z {
            let slab_base = batch_offset + (super_block * z + zi) * n_pass;
            #[unroll]
            for k in 0..loads_per_thread {
                let local_idx = tid + k * wg_size;
                shared[zi * n_pass + local_idx] =
                    monty_mul::<P>(input[slab_base + local_idx], n_inv_value);
            }
        }
    } else {
        #[unroll]
        for zi in 0..z {
            let slab_base = batch_offset + (super_block * z + zi) * n_pass;
            #[unroll]
            for k in 0..loads_per_thread {
                let local_idx = tid + k * wg_size;
                shared[zi * n_pass + local_idx] = input[slab_base + local_idx];
            }
        }
    }
    sync_cube();

    // -- Butterfly stages (GS-DIF, descending stride) ------------------
    //
    // At pass-local stage s, stride = 2^(log_pass - 1 - s) (descending).
    #[unroll]
    for s in 0..log_pass {
        let log_d = comptime!(log_pass - 1 - s);
        let d = comptime!(1usize << (log_pass - 1 - s));
        let mask_d = comptime!((1usize << (log_pass - 1 - s)) - 1);
        let log_two_d = comptime!(log_pass - s);

        // Twiddle: inv_twiddles[j_global * 2^(stage_offset + s)]
        //
        // For the first pass (stage_offset == 0):
        //   j_global = wg_pos + j * N_other
        //   tw = inv_twiddles[wg_pos * 2^s + j * 2^(log_n - log_pass + s)]
        //
        // For the last pass (stage_offset + log_pass == log_n):
        //   j_global = j (chunk base is a multiple of d_global)
        //   tw = inv_twiddles[j * 2^(stage_offset + s)]
        let inner_step = comptime!(if stage_offset == 0 {
            1usize << (log_n - log_pass + s)
        } else {
            1usize << (stage_offset + s)
        });
        let outer_step = comptime!(1usize << s);

        #[unroll]
        for k in 0..butt_per_thread {
            let butt_idx = tid + k * wg_size;
            let group = butt_idx >> log_d;
            let j = butt_idx & mask_d;
            let i_lo = (group << log_two_d) | j;
            let i_hi = i_lo + d;
            let j_inner = j * inner_step;

            #[unroll]
            for zi in 0..z {
                let tw = if comptime!(is_first_pass) {
                    let wg_pos = super_block * z + zi;
                    inv_twiddles[wg_pos * outer_step + j_inner]
                } else {
                    inv_twiddles[j_inner]
                };

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

    // -- Store ---------------------------------------------------------
    if comptime!(transpose_out) {
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
