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
pub fn build_fwd_twiddles<P: MontyParameters>(log_n: u32) -> Vec<u32> {
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
/// in Montgomery form. Used by the GS-DIF inverse kernel.
pub fn build_inv_twiddles<P: MontyParameters>(log_n: u32) -> Vec<u32> {
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

/// In-place bit-reversal permutation of a power-of-two-sized slice.
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
}
