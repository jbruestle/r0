//! Monolithic NTT kernels (forward + inverse) for `log_n ≤ 10`.
//!
//! A single workgroup processes the entire 2^log_n-point transform
//! through workgroup-shared memory. Both kernels are **in-place**:
//! the `data` buffer is loaded into shared memory, butterflied through
//! all log_n stages, then written back. sppark uses the same pattern
//! (`fr_t* d_inout`).
//!
//! Forward (`ntt_monolithic`): CT-DIT butterfly `(a + ω·b, a − ω·b)`,
//! ascending stride, bit-reversed coefficients in → natural-order
//! evaluations out (`DESIGN.md` §7).
//!
//! Inverse (`ntt_monolithic_inverse`): GS-DIF butterfly
//! `(a + b, (a − b) · ω)`, descending stride, inverse twiddles, with
//! the `×N⁻¹` scaling folded into the load step. Natural-order
//! evaluations in → bit-reversed coefficients out.
//!
//! Per-stage twiddles are read from flat `[ω^0..ω^(N/2−1)]` (or
//! inverse) tables built on the host.

use cubecl::prelude::*;
use r0_field::{monty_add, monty_mul, monty_sub, MontyParameters};

#[cube(launch_unchecked)]
pub fn ntt_monolithic<P: MontyParameters>(
    data: &mut Array<u32>,
    twiddles: &Array<u32>,
    #[comptime] log_n: u32,
    #[comptime] log_wg: u32,
) {
    // Comptime-derived sizes — all `usize` so they flow into Array
    // indexing without further casts.
    //
    // Constraints (enforced at the launch site, not asserted in
    // kernel): log_n >= 1, log_wg <= log_n - 1, workgroup count = 1.
    let n = comptime!(1usize << log_n);
    let wg_size = comptime!(1usize << log_wg);
    let loads_per_thread = comptime!(1usize << (log_n - log_wg));
    let butt_per_thread = comptime!(1usize << (log_n - 1 - log_wg));

    let mut shared = SharedMemory::<u32>::new(n);
    let tid = UNIT_POS as usize;

    // -- Load: each thread pulls `loads_per_thread` strided elements --
    #[unroll]
    for k in 0..loads_per_thread {
        let idx = tid + k * wg_size;
        shared[idx] = data[idx];
    }
    sync_cube();

    // -- Stages: s = 0..log_n, stride d = 2^s --
    #[unroll]
    for s in 0..log_n {
        let d = comptime!(1usize << s);
        let mask_d = comptime!((1usize << s) - 1);
        let log_two_d = comptime!(s + 1);
        // Twiddle stride: stage s wants ω^{j · 2^(log_n − s − 1)} for
        // butterfly j ∈ 0..d. Flat table holds ω^0..ω^(N/2−1).
        let tw_step = comptime!(1usize << (log_n - s - 1));

        #[unroll]
        for k in 0..butt_per_thread {
            let butt_idx = tid + k * wg_size;
            // `butt_idx ∈ 0..N/2`. Decompose into (group, j) pair where
            // each group spans `2d` consecutive shared-memory slots and
            // j ∈ 0..d is the offset within the group's lower half.
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

    // -- Store: shared → data (in-place) --
    #[unroll]
    for k in 0..loads_per_thread {
        let idx = tid + k * wg_size;
        data[idx] = shared[idx];
    }
}

/// Monolithic inverse NTT: natural-order evaluations in, bit-reversed
/// coefficients out (the reverse of `ntt_monolithic`'s direction).
///
/// Implementation: GS-DIF butterfly `(a + b, (a − b) · ω⁻¹)` with stages
/// running in **descending stride** (`N/2, …, 1`). Inverse twiddles are
/// passed as a separate flat table (see `build_inv_twiddles`); the
/// `×N⁻¹` scaling is folded into the load step by pre-multiplying each
/// input value with `inv_n[0]`.
///
/// `inv_n` is a single-element `Array<u32>` so the host can supply
/// `N⁻¹ mod p` (in Montgomery form) at runtime — comptime computation
/// of field constants would require trait constants in `comptime!()`,
/// which cubecl 0.9 handles awkwardly.
#[cube(launch_unchecked)]
pub fn ntt_monolithic_inverse<P: MontyParameters>(
    data: &mut Array<u32>,
    inv_twiddles: &Array<u32>,
    inv_n: &Array<u32>,
    #[comptime] log_n: u32,
    #[comptime] log_wg: u32,
) {
    let n = comptime!(1usize << log_n);
    let wg_size = comptime!(1usize << log_wg);
    let loads_per_thread = comptime!(1usize << (log_n - log_wg));
    let butt_per_thread = comptime!(1usize << (log_n - 1 - log_wg));

    let mut shared = SharedMemory::<u32>::new(n);
    let tid = UNIT_POS as usize;
    let n_inv_value = inv_n[0];

    // -- Load: pull strided elements, pre-multiplied by N⁻¹ --
    #[unroll]
    for k in 0..loads_per_thread {
        let idx = tid + k * wg_size;
        shared[idx] = monty_mul::<P>(data[idx], n_inv_value);
    }
    sync_cube();

    // -- Stages: s = 0..log_n, stride d_s = 2^(log_n − 1 − s) (descending) --
    #[unroll]
    for s in 0..log_n {
        let log_d = comptime!(log_n - 1 - s);
        let d = comptime!(1usize << (log_n - 1 - s));
        let mask_d = comptime!((1usize << (log_n - 1 - s)) - 1);
        let log_two_d = comptime!(log_n - s);
        // Twiddle stride at stage s: 2^s. Flat table holds ω^{-0}..ω^{-(N/2-1)}.
        let tw_step = comptime!(1usize << s);

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
            // GS-DIF butterfly: (a, b, ω) → (a + b, (a − b) · ω).
            let sum = monty_add::<P>(a, b);
            let diff = monty_sub::<P>(a, b);
            let dt = monty_mul::<P>(diff, tw);
            shared[i_lo] = sum;
            shared[i_hi] = dt;
        }
        sync_cube();
    }

    // -- Store: shared → data (in-place) --
    #[unroll]
    for k in 0..loads_per_thread {
        let idx = tid + k * wg_size;
        data[idx] = shared[idx];
    }
}
