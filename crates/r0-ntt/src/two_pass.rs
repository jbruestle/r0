//! Two-pass NTT (forward + inverse) for `log_n ∈ [11, 20]`.
//!
//! # Forward
//!
//! Splits the `log_n` CT-DIT stages into a low-stride prefix (pass 1)
//! and a high-stride suffix (pass 2), each running through workgroup-
//! shared memory:
//!
//! - **`ntt_pass1`**: each workgroup processes one contiguous
//!   `N1 = 2^log_n1` chunk. Applies stages `0..log_n1` (strides
//!   `1..N1/2`). Workgroup count = `N2 = N/N1`.
//! - **`ntt_pass2`**: each workgroup processes one strided
//!   `N2 = 2^log_n2` slab at base `i_low`, stride `N1`. Applies stages
//!   `log_n1..log_n` (strides `N1..N/2`) in pass-2-local form
//!   `1..N2/2`. Workgroup count = `N1`.
//!
//! No twist factor is needed between passes — with bit-reversed input
//! `data[i] = a_{bit_rev_N(i)}`, each pass-1 chunk is naturally the
//! bit-reversed-N1 form of a stride-N2 sub-sequence of `a`, so pass 1
//! produces partial DFTs that line up exactly for pass 2 to combine
//! using the original N-point CT-DIT twiddle indices.
//!
//! # Inverse
//!
//! Mirror of forward. The inverse iterates GS-DIF stages in
//! descending-stride order, so pass 1 (first in pipeline) handles the
//! high-stride stages, pass 2 the low-stride stages:
//!
//! - **`intt_pass1`**: strided `N2`-slab at base `i_low`, stride `N1`.
//!   Applies stages with original stride `N/2..N1` (pass-1-local
//!   descending stride `N2/2..1`). Also does the `×N⁻¹` pre-mult at
//!   the load step.
//! - **`intt_pass2`**: contiguous `N1`-chunk at offset `block_id*N1`.
//!   Applies stages with original stride `N1/2..1` (pass-2-local
//!   descending stride `N1/2..1`).
//!
//! Bound constraints (host-side, not asserted in kernel):
//! - `log_n1 + log_n2 == log_n`, both ≥ 1.
//! - For each pass, the kernel's `log_wg` must satisfy
//!   `log_wg ≤ log_pass_size − 1`.

use cubecl::prelude::*;
use r0_field::{monty_add, monty_mul, monty_sub, MontyParameters};

/// Pass 1: contiguous-chunk CT-DIT on `2^log_n1` elements per workgroup.
/// Workgroup index = chunk id, range `[0, N2)`. In-place on `data`.
#[cube(launch_unchecked)]
pub fn ntt_pass1<P: MontyParameters>(
    data: &mut Array<u32>,
    twiddles: &Array<u32>,
    #[comptime] log_n: u32,
    #[comptime] log_n1: u32,
    #[comptime] log_wg: u32,
) {
    let n1 = comptime!(1usize << log_n1);
    let wg_size = comptime!(1usize << log_wg);
    let loads_per_thread = comptime!(1usize << (log_n1 - log_wg));
    let butt_per_thread = comptime!(1usize << (log_n1 - 1 - log_wg));

    let block_id = CUBE_POS as usize;
    let block_offset = block_id * n1;
    let mut shared = SharedMemory::<u32>::new(n1);
    let tid = UNIT_POS as usize;

    // -- Load: contiguous slice from data --
    #[unroll]
    for k in 0..loads_per_thread {
        let local_idx = tid + k * wg_size;
        shared[local_idx] = data[block_offset + local_idx];
    }
    sync_cube();

    // -- Stages s = 0..log_n1, stride 2^s --
    #[unroll]
    for s in 0..log_n1 {
        let d = comptime!(1usize << s);
        let mask_d = comptime!((1usize << s) - 1);
        let log_two_d = comptime!(s + 1);
        // Twiddle stride in the global N-point flat table (same as the
        // `2^(log_n − s − 1)` formula used by `ntt_monolithic`).
        let tw_step = comptime!(1usize << (log_n - s - 1));

        #[unroll]
        for k in 0..butt_per_thread {
            let butt_idx = tid + k * wg_size;
            let group = butt_idx >> s;
            let j = butt_idx & mask_d;
            let i_lo = (group << log_two_d) | j;
            let i_hi = i_lo + d;

            let tw = twiddles[j * tw_step];
            let a = shared[i_lo];
            let b = shared[i_hi];
            let t = monty_mul::<P>(tw, b);
            shared[i_lo] = monty_add::<P>(a, t);
            shared[i_hi] = monty_sub::<P>(a, t);
        }
        sync_cube();
    }

    // -- Store --
    #[unroll]
    for k in 0..loads_per_thread {
        let local_idx = tid + k * wg_size;
        data[block_offset + local_idx] = shared[local_idx];
    }
}

/// Pass 2: strided-slab CT-DIT on `2^log_n2 = 2^(log_n - log_n1)`
/// elements per workgroup. Workgroup index = `i_low ∈ [0, N1)`. The
/// slab is `{ data[i_low + j*N1] : j ∈ 0..N2 }`. In-place on `data`.
///
/// At pass-2-local stage `s' ∈ [0, log_n2)`, butterfly `j ∈ [0, 2^s')`,
/// the global twiddle index is `i_low * outer_step + j * inner_step`
/// where:
/// - `outer_step = 2^(log_n2 - s' - 1)` (per workgroup, per stage)
/// - `inner_step = 2^(log_n  - s' - 1)` (per stage)
///
/// — exactly the original-N CT-DIT twiddle for stage `log_n1 + s'`,
/// butterfly `i_low + j*N1`.
#[cube(launch_unchecked)]
pub fn ntt_pass2<P: MontyParameters>(
    data: &mut Array<u32>,
    twiddles: &Array<u32>,
    #[comptime] log_n: u32,
    #[comptime] log_n1: u32,
    #[comptime] log_wg: u32,
) {
    let n1 = comptime!(1usize << log_n1);
    let n2 = comptime!(1usize << (log_n - log_n1));
    let log_n2 = comptime!(log_n - log_n1);
    let wg_size = comptime!(1usize << log_wg);
    let loads_per_thread = comptime!(1usize << (log_n - log_n1 - log_wg));
    let butt_per_thread = comptime!(1usize << (log_n - log_n1 - 1 - log_wg));

    let i_low = CUBE_POS as usize;
    let mut shared = SharedMemory::<u32>::new(n2);
    let tid = UNIT_POS as usize;

    // -- Load: strided slab `data[i_low + j*N1]` for j ∈ 0..N2 --
    #[unroll]
    for k in 0..loads_per_thread {
        let local_j = tid + k * wg_size;
        shared[local_j] = data[i_low + local_j * n1];
    }
    sync_cube();

    // -- Stages s' = 0..log_n2, stride 2^s' --
    #[unroll]
    for s_prime in 0..log_n2 {
        let d = comptime!(1usize << s_prime);
        let mask_d = comptime!((1usize << s_prime) - 1);
        let log_two_d = comptime!(s_prime + 1);
        let outer_step = comptime!(1usize << (log_n - log_n1 - s_prime - 1));
        let inner_step = comptime!(1usize << (log_n - s_prime - 1));
        // Cache the per-workgroup, per-stage offset into the twiddle
        // table — invariant across all butterflies within this stage.
        let i_low_term = i_low * outer_step;

        #[unroll]
        for k in 0..butt_per_thread {
            let butt_idx = tid + k * wg_size;
            let group = butt_idx >> s_prime;
            let j = butt_idx & mask_d;
            let i_lo = (group << log_two_d) | j;
            let i_hi = i_lo + d;

            let tw_idx = i_low_term + j * inner_step;
            let tw = twiddles[tw_idx];
            let a = shared[i_lo];
            let b = shared[i_hi];
            let t = monty_mul::<P>(tw, b);
            shared[i_lo] = monty_add::<P>(a, t);
            shared[i_hi] = monty_sub::<P>(a, t);
        }
        sync_cube();
    }

    // -- Store: strided write back --
    #[unroll]
    for k in 0..loads_per_thread {
        let local_j = tid + k * wg_size;
        data[i_low + local_j * n1] = shared[local_j];
    }
}

/// Inverse pass 1: strided-slab GS-DIF, handles the FIRST `log_n2`
/// stages of the descending-stride iteration (original strides
/// `N/2 .. N1`). Workgroup index = `i_low ∈ [0, N1)`. Slab is
/// `{ data[i_low + j*N1] : j ∈ 0..N2 }`. Also does the `×N⁻¹`
/// pre-multiplication at the load step.
///
/// Twiddle index for pass-1-local stage `s_p1` ∈ `[0, log_n2)`,
/// butterfly `j ∈ [0, 2^(log_n2 − 1 − s_p1))`:
///
/// ```text
///     tw_idx = i_low * 2^s_p1 + j * 2^(s_p1 + log_n1)
/// ```
///
/// — exactly the original-N inverse CT-DIT twiddle index `j_orig *
/// 2^s_prime` for `j_orig = i_low + j*N1`, `s_prime = s_p1`.
#[cube(launch_unchecked)]
pub fn intt_pass1<P: MontyParameters>(
    data: &mut Array<u32>,
    inv_twiddles: &Array<u32>,
    inv_n: &Array<u32>,
    #[comptime] log_n: u32,
    #[comptime] log_n1: u32,
    #[comptime] log_wg: u32,
) {
    let n1 = comptime!(1usize << log_n1);
    let n2 = comptime!(1usize << (log_n - log_n1));
    let log_n2 = comptime!(log_n - log_n1);
    let wg_size = comptime!(1usize << log_wg);
    let loads_per_thread = comptime!(1usize << (log_n - log_n1 - log_wg));
    let butt_per_thread = comptime!(1usize << (log_n - log_n1 - 1 - log_wg));

    let i_low = CUBE_POS as usize;
    let mut shared = SharedMemory::<u32>::new(n2);
    let tid = UNIT_POS as usize;
    let n_inv_value = inv_n[0];

    // -- Load: strided slab, pre-multiplied by N⁻¹ --
    #[unroll]
    for k in 0..loads_per_thread {
        let local_j = tid + k * wg_size;
        shared[local_j] = monty_mul::<P>(data[i_low + local_j * n1], n_inv_value);
    }
    sync_cube();

    // -- Stages s_p1 ∈ [0, log_n2), local stride descending: 2^(log_n2 − 1 − s_p1) --
    #[unroll]
    for s_p1 in 0..log_n2 {
        let log_d = comptime!(log_n2 - 1 - s_p1);
        let d = comptime!(1usize << (log_n2 - 1 - s_p1));
        let mask_d = comptime!((1usize << (log_n2 - 1 - s_p1)) - 1);
        let log_two_d = comptime!(log_n2 - s_p1);
        // Twiddle index: i_low * outer_step + j * inner_step.
        // outer_step = 2^s_p1 (per workgroup, per stage).
        // inner_step = 2^(s_p1 + log_n1) (per stage).
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
            // GS-DIF: (a, b, ω) → (a + b, (a − b) · ω).
            let sum = monty_add::<P>(a, b);
            let diff = monty_sub::<P>(a, b);
            let dt = monty_mul::<P>(diff, tw);
            shared[i_lo] = sum;
            shared[i_hi] = dt;
        }
        sync_cube();
    }

    // -- Store: strided write back --
    #[unroll]
    for k in 0..loads_per_thread {
        let local_j = tid + k * wg_size;
        data[i_low + local_j * n1] = shared[local_j];
    }
}

/// Inverse pass 2: contiguous-chunk GS-DIF, handles the LAST `log_n1`
/// stages of the descending-stride iteration (original strides
/// `N1/2 .. 1`). Workgroup index = chunk id `∈ [0, N2)`. Chunk is
/// `data[block_id*N1 .. (block_id+1)*N1]`.
///
/// Twiddle index for pass-2-local stage `s_p2` ∈ `[0, log_n1)`,
/// butterfly `j ∈ [0, 2^(log_n1 − 1 − s_p2))`:
///
/// ```text
///     tw_idx = j * 2^(log_n2 + s_p2)
/// ```
///
/// — independent of `block_id`, since these stages are entirely within
/// each chunk.
#[cube(launch_unchecked)]
pub fn intt_pass2<P: MontyParameters>(
    data: &mut Array<u32>,
    inv_twiddles: &Array<u32>,
    #[comptime] log_n: u32,
    #[comptime] log_n1: u32,
    #[comptime] log_wg: u32,
) {
    let n1 = comptime!(1usize << log_n1);
    let log_n2 = comptime!(log_n - log_n1);
    let wg_size = comptime!(1usize << log_wg);
    let loads_per_thread = comptime!(1usize << (log_n1 - log_wg));
    let butt_per_thread = comptime!(1usize << (log_n1 - 1 - log_wg));

    let block_id = CUBE_POS as usize;
    let block_offset = block_id * n1;
    let mut shared = SharedMemory::<u32>::new(n1);
    let tid = UNIT_POS as usize;

    // -- Load: contiguous slice from data --
    #[unroll]
    for k in 0..loads_per_thread {
        let local_idx = tid + k * wg_size;
        shared[local_idx] = data[block_offset + local_idx];
    }
    sync_cube();

    // -- Stages s_p2 ∈ [0, log_n1), local stride descending: 2^(log_n1 − 1 − s_p2) --
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

    // -- Store --
    #[unroll]
    for k in 0..loads_per_thread {
        let local_idx = tid + k * wg_size;
        data[block_offset + local_idx] = shared[local_idx];
    }
}
