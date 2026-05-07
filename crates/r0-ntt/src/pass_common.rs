//! Helpers shared by the forward and inverse pass kernels.

use cubecl::prelude::*;
use r0_field::{monty_mul, MontyParameters};

/// Reconstruct a twiddle factor `w^k` from the windowed partial twiddle table.
///
/// `partial_twiddles` has layout `[NUM_WINDOWS][WINDOW_SIZE]` (3 windows of
/// 1024). Decomposes `k` into 10-bit windows and multiplies the corresponding
/// entries. Window 0's first read seeds the accumulator; remaining windows
/// each contribute one `monty_mul`. For `log_n <= 20` only `num_windows = 2`
/// is needed (single multiply per twiddle); `log_n` 21..=24 needs all three.
#[cube]
pub(crate) fn reconstruct_twiddle<P: MontyParameters>(
    partial_twiddles: &Array<u32>,
    k: u32,
    #[comptime] num_windows: u32,
) -> u32 {
    let lg_window = comptime!(10u32);
    let window_mask = comptime!((1u32 << 10) - 1);
    let window_size = comptime!(1024usize);

    // Start with window 0's entry (avoids an extra multiply vs starting at 1).
    let k_0 = k & window_mask;
    let mut acc = partial_twiddles[k_0 as usize];

    // Remaining windows: branchless multiply. partial[w][0] = monty(1) = identity.
    // The 12 KiB table fits in L1 cache, so global reads are fast.
    #[unroll]
    for w in 1..num_windows {
        let k_w = (k >> (w * lg_window)) & window_mask;
        let idx = comptime!(w as usize) * window_size + k_w as usize;
        acc = monty_mul::<P>(acc, partial_twiddles[idx]);
    }
    acc
}
