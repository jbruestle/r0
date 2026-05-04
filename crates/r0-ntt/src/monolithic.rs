//! Monolithic forward NTT kernel for `log_n ≤ 10`.
//!
//! Single workgroup processes the entire 2^log_n-point transform through
//! workgroup-shared memory. Stages run with stride `2^s` ascending from
//! `s=0` (stride 1) to `s=log_n-1` (stride N/2), which is the iteration
//! pattern that consumes bit-reversed coefficients and produces
//! natural-order evaluations — sppark's RN convention, our canonical
//! convention (`DESIGN.md` §7).
//!
//! Per-stage twiddles are read from a flat `[ω^0, …, ω^(N/2−1)]` table
//! built on the host. Stage `s` uses twiddles at stride `2^(log_n−s−1)`.

use cubecl::prelude::*;
use r0_field::{monty_add, monty_mul, monty_sub, MontyParameters};

#[cube(launch_unchecked)]
pub fn ntt_monolithic<P: MontyParameters>(
    input: &Array<u32>,
    twiddles: &Array<u32>,
    output: &mut Array<u32>,
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
        shared[idx] = input[idx];
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

    // -- Store: shared → output --
    #[unroll]
    for k in 0..loads_per_thread {
        let idx = tid + k * wg_size;
        output[idx] = shared[idx];
    }
}
