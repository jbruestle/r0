//! Two-pass inverse NTT for `log_n ∈ [11, 20]`.
//!
//! The inverse iterates GS-DIF stages in descending-stride order:
//!
//! - **`intt_pass1`**: strided `N2`-slab at base `i_low`, stride `N1`.
//!   Applies stages with original stride `N/2..N1` (pass-1-local
//!   descending stride `N2/2..1`). Also does the `×N⁻¹` pre-mult at
//!   the load step.
//! - **`intt_pass2`**: contiguous `N1`-chunk at offset `block_id*N1`.
//!   Applies stages with original stride `N1/2..1` (pass-2-local
//!   descending stride `N1/2..1`).
//!
//! These kernels use the original (non-transposed) memory layout and
//! operate in-place safely (each workgroup reads and writes the same
//! positions).
//!
//! Grid-Y batching is supported via `CUBE_POS_Y`.

use cubecl::prelude::*;
use r0_field::{monty_add, monty_mul, monty_sub, MontyParameters};

/// Inverse pass 1: strided-slab GS-DIF.
#[cube(launch_unchecked)]
pub fn intt_pass1<P: MontyParameters>(
    data: &mut Array<u32>,
    inv_twiddles: &Array<u32>,
    inv_n: &Array<u32>,
    #[comptime] log_n: u32,
    #[comptime] log_n1: u32,
    #[comptime] log_wg: u32,
) {
    let n = comptime!(1usize << log_n);
    let n1 = comptime!(1usize << log_n1);
    let n2 = comptime!(1usize << (log_n - log_n1));
    let log_n2 = comptime!(log_n - log_n1);
    let wg_size = comptime!(1usize << log_wg);
    let loads_per_thread = comptime!(1usize << (log_n - log_n1 - log_wg));
    let butt_per_thread = comptime!(1usize << (log_n - log_n1 - 1 - log_wg));

    let batch_offset = CUBE_POS_Y as usize * n;
    let i_low = CUBE_POS_X as usize;
    let mut shared = SharedMemory::<u32>::new(n2);
    let tid = UNIT_POS as usize;
    let n_inv_value = inv_n[0];

    #[unroll]
    for k in 0..loads_per_thread {
        let local_j = tid + k * wg_size;
        shared[local_j] = monty_mul::<P>(data[batch_offset + i_low + local_j * n1], n_inv_value);
    }
    sync_cube();

    #[unroll]
    for s_p1 in 0..log_n2 {
        let log_d = comptime!(log_n2 - 1 - s_p1);
        let d = comptime!(1usize << (log_n2 - 1 - s_p1));
        let mask_d = comptime!((1usize << (log_n2 - 1 - s_p1)) - 1);
        let log_two_d = comptime!(log_n2 - s_p1);
        let outer_step = comptime!(1usize << s_p1);
        let inner_step = comptime!(1usize << (s_p1 + log_n1));
        let i_low_term = i_low * outer_step;

        #[unroll]
        for k in 0..butt_per_thread {
            let butt_idx = tid + k * wg_size;
            let group = butt_idx >> log_d;
            let j = butt_idx & mask_d;
            let i_lo = (group << log_two_d) | j;
            let i_hi = i_lo + d;

            let tw_idx = i_low_term + j * inner_step;
            let tw = inv_twiddles[tw_idx];
            let a = shared[i_lo];
            let b = shared[i_hi];
            let sum = monty_add::<P>(a, b);
            let diff = monty_sub::<P>(a, b);
            let dt = monty_mul::<P>(diff, tw);
            shared[i_lo] = sum;
            shared[i_hi] = dt;
        }
        sync_cube();
    }

    #[unroll]
    for k in 0..loads_per_thread {
        let local_j = tid + k * wg_size;
        data[batch_offset + i_low + local_j * n1] = shared[local_j];
    }
}

/// Inverse pass 2: contiguous-chunk GS-DIF.
#[cube(launch_unchecked)]
pub fn intt_pass2<P: MontyParameters>(
    data: &mut Array<u32>,
    inv_twiddles: &Array<u32>,
    #[comptime] log_n: u32,
    #[comptime] log_n1: u32,
    #[comptime] log_wg: u32,
) {
    let n = comptime!(1usize << log_n);
    let n1 = comptime!(1usize << log_n1);
    let log_n2 = comptime!(log_n - log_n1);
    let wg_size = comptime!(1usize << log_wg);
    let loads_per_thread = comptime!(1usize << (log_n1 - log_wg));
    let butt_per_thread = comptime!(1usize << (log_n1 - 1 - log_wg));

    let block_id = CUBE_POS_X as usize;
    let batch_offset = CUBE_POS_Y as usize * n;
    let block_offset = batch_offset + block_id * n1;
    let mut shared = SharedMemory::<u32>::new(n1);
    let tid = UNIT_POS as usize;

    #[unroll]
    for k in 0..loads_per_thread {
        let local_idx = tid + k * wg_size;
        shared[local_idx] = data[block_offset + local_idx];
    }
    sync_cube();

    #[unroll]
    for s_p2 in 0..log_n1 {
        let log_d = comptime!(log_n1 - 1 - s_p2);
        let d = comptime!(1usize << (log_n1 - 1 - s_p2));
        let mask_d = comptime!((1usize << (log_n1 - 1 - s_p2)) - 1);
        let log_two_d = comptime!(log_n1 - s_p2);
        let tw_step = comptime!(1usize << (log_n2 + s_p2));

        #[unroll]
        for k in 0..butt_per_thread {
            let butt_idx = tid + k * wg_size;
            let group = butt_idx >> log_d;
            let j = butt_idx & mask_d;
            let i_lo = (group << log_two_d) | j;
            let i_hi = i_lo + d;

            let tw = inv_twiddles[j * tw_step];
            let a = shared[i_lo];
            let b = shared[i_hi];
            let sum = monty_add::<P>(a, b);
            let diff = monty_sub::<P>(a, b);
            let dt = monty_mul::<P>(diff, tw);
            shared[i_lo] = sum;
            shared[i_hi] = dt;
        }
        sync_cube();
    }

    #[unroll]
    for k in 0..loads_per_thread {
        let local_idx = tid + k * wg_size;
        data[block_offset + local_idx] = shared[local_idx];
    }
}
