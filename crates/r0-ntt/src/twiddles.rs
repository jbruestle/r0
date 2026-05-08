//! Host-side twiddle precomputation, bit-reversal, and field-elem helpers.

use r0_field::{MontyField, MontyParameters};

/// Square-and-multiply exponentiation in `MontyField<P>`.
fn pow_field<P: MontyParameters>(mut base: MontyField<P>, mut exp: u32) -> MontyField<P> {
    let mut acc = MontyField::<P>::from_canonical(1);
    while exp > 0 {
        if exp & 1 == 1 {
            acc = acc * base;
        }
        base = base * base;
        exp >>= 1;
    }
    acc
}

/// Build a flat forward-twiddle table `[w^0, w^1, ..., w^(N/2 - 1)]` in
/// Montgomery form, where `w` is a primitive `N`-th root of unity
/// (`N = 2^log_n`). For `log_n == 0` returns an empty vector.
///
/// Reference implementation kept around to cross-check the partial
/// (windowed) tables — the kernels never need the flat form.
#[cfg(test)]
fn build_fwd_twiddles<P: MontyParameters>(log_n: u32) -> Vec<u32> {
    if log_n == 0 {
        return Vec::new();
    }
    let half = 1usize << (log_n - 1);
    let omega = MontyField::<P>::from_canonical(P::TWO_ADIC_GENERATORS[log_n as usize]);
    let mut out = Vec::with_capacity(half);
    let mut current = MontyField::<P>::from_canonical(1);
    for _ in 0..half {
        out.push(current.raw());
        current = current * omega;
    }
    out
}

/// Build a flat inverse-twiddle table `[w^{-0}, w^{-1}, ..., w^{-(N/2-1)}]`
/// in Montgomery form. Companion reference impl to [`build_fwd_twiddles`].
#[cfg(test)]
fn build_inv_twiddles<P: MontyParameters>(log_n: u32) -> Vec<u32> {
    if log_n == 0 {
        return Vec::new();
    }
    let half = 1usize << (log_n - 1);
    let omega = MontyField::<P>::from_canonical(P::TWO_ADIC_GENERATORS[log_n as usize]);
    // w^{-1} = w^{N - 1} since w^N = 1.
    let inv_omega = pow_field::<P>(omega, (1u32 << log_n) - 1);
    let mut out = Vec::with_capacity(half);
    let mut current = MontyField::<P>::from_canonical(1);
    for _ in 0..half {
        out.push(current.raw());
        current = current * inv_omega;
    }
    out
}

/// `N^{-1} mod p` in Montgomery form, where `N = 2^log_n`.
/// Used by the inverse NTT kernel as a load-time scaling factor.
pub fn n_inv<P: MontyParameters>(log_n: u32) -> u32 {
    let n = 1u32 << log_n;
    let n_field = MontyField::<P>::from_canonical(n);
    // Fermat: N^{-1} = N^{p-2} mod p.
    pow_field::<P>(n_field, P::PRIME - 2).raw()
}

// -- Windowed partial twiddle tables --
//
// Instead of storing N/2 twiddle factors (2 MiB for log_n=20), we store a
// small partial table of NUM_WINDOWS * WINDOW_SIZE entries and reconstruct
// any w^k on-the-fly via at most NUM_WINDOWS-1 multiplications.
//
// Layout: flat [window_0[0..1024], window_1[0..1024], window_2[0..1024]]
// partial[w][i] = omega^(i * 2^(w * LG_WINDOW))

/// Window size exponent for partial twiddle tables.
pub const LG_WINDOW: u32 = 10;
/// Window size (number of entries per window).
pub const WINDOW_SIZE: usize = 1 << LG_WINDOW; // 1024
/// Number of windows. Covers up to 30 bits of exponent (sufficient for
/// BabyBear S=27 and KoalaBear S=24).
pub const NUM_WINDOWS: usize = 3;
/// Total number of entries in a partial twiddle table.
pub const PARTIAL_TWIDDLE_LEN: usize = NUM_WINDOWS * WINDOW_SIZE; // 3072

/// Build a forward partial twiddle table for the given `log_n`.
///
/// Returns a flat `Vec<u32>` of length `NUM_WINDOWS * WINDOW_SIZE` (3072).
/// Entry `[w * WINDOW_SIZE + i]` = omega^(i * 2^(w * LG_WINDOW)) in
/// Montgomery form, where omega is a primitive 2^log_n-th root of unity.
pub fn build_partial_fwd_twiddles<P: MontyParameters>(log_n: u32) -> Vec<u32> {
    assert!(log_n >= 1);
    let omega = MontyField::<P>::from_canonical(P::TWO_ADIC_GENERATORS[log_n as usize]);
    build_partial_table::<P>(omega, log_n)
}

/// Build an inverse partial twiddle table for the given `log_n`.
///
/// Same layout as forward, but uses omega^{-1} as the base root.
pub fn build_partial_inv_twiddles<P: MontyParameters>(log_n: u32) -> Vec<u32> {
    assert!(log_n >= 1);
    let omega = MontyField::<P>::from_canonical(P::TWO_ADIC_GENERATORS[log_n as usize]);
    let inv_omega = pow_field::<P>(omega, (1u32 << log_n) - 1);
    build_partial_table::<P>(inv_omega, log_n)
}

fn build_partial_table<P: MontyParameters>(omega: MontyField<P>, _log_n: u32) -> Vec<u32> {
    let mut out = vec![0u32; PARTIAL_TWIDDLE_LEN];
    // Window 0: partial[0][i] = omega^i
    let mut base = omega; // omega^(2^0) = omega
    for w in 0..NUM_WINDOWS {
        // base = omega^(2^(w * LG_WINDOW))
        let mut current = MontyField::<P>::from_canonical(1);
        for i in 0..WINDOW_SIZE {
            out[w * WINDOW_SIZE + i] = current.raw();
            current = current * base;
        }
        // Advance base: square LG_WINDOW times to get omega^(2^((w+1)*LG_WINDOW))
        for _ in 0..LG_WINDOW {
            base = base * base;
        }
    }
    out
}

/// In-place bit-reversal permutation of a power-of-two-length slice.
///
/// In this crate's convention, NTT coefficients live in bit-reversed
/// memory order while evaluations live in natural memory order. Apply
/// this function to map between them: before [`NttExec::forward`] to
/// place coefficients in bit-reversed order, or after
/// [`NttExec::inverse`] to read the recovered coefficients in natural
/// order.
///
/// # Panics
///
/// Panics if `data.len()` is not a power of two.
///
/// # Example
///
/// ```
/// use r0_ntt::bit_reverse_in_place;
/// let mut v = [0u32, 1, 2, 3, 4, 5, 6, 7];
/// bit_reverse_in_place(&mut v);
/// assert_eq!(v, [0, 4, 2, 6, 1, 5, 3, 7]);
/// ```
///
/// [`NttExec::forward`]: crate::NttExec::forward
/// [`NttExec::inverse`]: crate::NttExec::inverse
pub fn bit_reverse_in_place<T: Copy>(data: &mut [T]) {
    let n = data.len();
    if n <= 1 {
        return;
    }
    let log_n = n.trailing_zeros();
    assert_eq!(1usize << log_n, n, "bit_reverse requires power-of-two length");
    for i in 0..n {
        let j = (i as u32).reverse_bits() >> (32 - log_n);
        let j = j as usize;
        if i < j {
            data.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use r0_field::BabyBearParameters;

    #[test]
    fn twiddles_omega_n_is_one() {
        for log_n in 1..=10u32 {
            let omega = MontyField::<BabyBearParameters>::from_canonical(
                BabyBearParameters::TWO_ADIC_GENERATORS[log_n as usize],
            );
            let mut x = omega;
            for _ in 0..log_n {
                x = x * x;
            }
            assert_eq!(
                x,
                MontyField::<BabyBearParameters>::from_canonical(1),
                "w^N != 1 for log_n = {log_n}"
            );
        }
    }

    #[test]
    fn twiddles_first_two_correct() {
        // tw[0] = 1; tw[1] = w.
        for log_n in 1..=10u32 {
            let tw = build_fwd_twiddles::<BabyBearParameters>(log_n);
            let one = MontyField::<BabyBearParameters>::from_canonical(1);
            assert_eq!(MontyField::<BabyBearParameters>::from_raw(tw[0]), one);
            if tw.len() >= 2 {
                let omega = MontyField::<BabyBearParameters>::from_canonical(
                    BabyBearParameters::TWO_ADIC_GENERATORS[log_n as usize],
                );
                assert_eq!(MontyField::<BabyBearParameters>::from_raw(tw[1]), omega);
            }
        }
    }

    #[test]
    fn bit_reverse_involutive() {
        let mut x: Vec<u32> = (0..1024u32).collect();
        let original = x.clone();
        bit_reverse_in_place(&mut x);
        bit_reverse_in_place(&mut x);
        assert_eq!(x, original);
    }

    /// Verify that reconstructing w^k from the partial table matches the
    /// flat twiddle table for all k in [0, N/2).
    #[test]
    fn partial_twiddles_reconstruct_matches_flat() {
        for log_n in [10u32, 14, 20] {
            let flat = build_fwd_twiddles::<BabyBearParameters>(log_n);
            let partial = build_partial_fwd_twiddles::<BabyBearParameters>(log_n);
            let half_n = 1usize << (log_n - 1);

            for k in 0..half_n {
                let reconstructed = reconstruct_twiddle::<BabyBearParameters>(&partial, k as u32);
                assert_eq!(
                    reconstructed, flat[k],
                    "mismatch at k={k}, log_n={log_n}"
                );
            }
        }
    }

    /// Verify inverse partial twiddles reconstruct correctly.
    #[test]
    fn partial_inv_twiddles_reconstruct_matches_flat() {
        for log_n in [10u32, 14, 20] {
            let flat = build_inv_twiddles::<BabyBearParameters>(log_n);
            let partial = build_partial_inv_twiddles::<BabyBearParameters>(log_n);
            let half_n = 1usize << (log_n - 1);

            for k in 0..half_n {
                let reconstructed = reconstruct_twiddle::<BabyBearParameters>(&partial, k as u32);
                assert_eq!(
                    reconstructed, flat[k],
                    "inv mismatch at k={k}, log_n={log_n}"
                );
            }
        }
    }
}

/// Host-side reconstruction of `w^k` from a partial twiddle table.
/// Mirrors what the kernel's [`crate::pass_common::reconstruct_twiddle`]
/// computes on-device; used to sanity-check that path.
#[cfg(test)]
fn reconstruct_twiddle<P: MontyParameters>(partial: &[u32], k: u32) -> u32 {
    let k_0 = (k & (WINDOW_SIZE as u32 - 1)) as usize;
    let mut acc = MontyField::<P>::from_raw(partial[k_0]);
    for w in 1..NUM_WINDOWS {
        let k_w = ((k >> (w as u32 * LG_WINDOW)) & (WINDOW_SIZE as u32 - 1)) as usize;
        let entry = MontyField::<P>::from_raw(partial[w * WINDOW_SIZE + k_w]);
        acc = acc * entry;
    }
    acc.raw()
}
