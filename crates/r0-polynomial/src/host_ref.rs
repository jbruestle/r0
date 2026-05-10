//! Serial host reference for division by `(x − z)`.
//!
//! Pure-Rust Horner-form synthetic division, used by tests as the
//! oracle the cube path checks against. The output convention matches
//! `PolyDivExec::div_by_x_minus_z` (`rotate=true`): quotient at
//! positions `[0..n−1]`, remainder at `[n−1]`.
//!
//! Operates on a generic `Field` (an opaque type satisfying the
//! [`HostField`] trait). Impls for all five `r0-field` instances ship
//! with this module so integration tests in this crate can use them
//! without running into orphan-rule issues.

use r0_field::{
    BabyBear4, BabyBear5, BabyBearParameters, KoalaBear4, KoalaBearParameters, MontyField,
};

/// Minimal field surface the reference needs.
pub trait HostField: Copy {
    fn zero() -> Self;
    fn add(a: Self, b: Self) -> Self;
    fn mul(a: Self, b: Self) -> Self;
}

// Impls for the five `r0-field` instances. Each delegates to the type's
// host-side operator overloads.

impl HostField for MontyField<BabyBearParameters> {
    fn zero() -> Self {
        Self::ZERO
    }
    fn add(a: Self, b: Self) -> Self {
        a + b
    }
    fn mul(a: Self, b: Self) -> Self {
        a * b
    }
}

impl HostField for MontyField<KoalaBearParameters> {
    fn zero() -> Self {
        Self::ZERO
    }
    fn add(a: Self, b: Self) -> Self {
        a + b
    }
    fn mul(a: Self, b: Self) -> Self {
        a * b
    }
}

impl HostField for BabyBear4 {
    fn zero() -> Self {
        Self::ZERO
    }
    fn add(a: Self, b: Self) -> Self {
        a + b
    }
    fn mul(a: Self, b: Self) -> Self {
        a * b
    }
}

impl HostField for KoalaBear4 {
    fn zero() -> Self {
        Self::ZERO
    }
    fn add(a: Self, b: Self) -> Self {
        a + b
    }
    fn mul(a: Self, b: Self) -> Self {
        a * b
    }
}

impl HostField for BabyBear5 {
    fn zero() -> Self {
        Self::ZERO
    }
    fn add(a: Self, b: Self) -> Self {
        a + b
    }
    fn mul(a: Self, b: Self) -> Self {
        a * b
    }
}

/// In-place serial division by `(x − z)` for one polynomial of length `n`.
/// Writes `rotate=true` output back into `coeffs`.
pub fn div_by_x_minus_z_serial<F: HostField>(coeffs: &mut [F], z: F) {
    let n = coeffs.len();
    if n == 0 {
        return;
    }
    // Horner-form synthetic division, descending degree:
    //   Q ← 0
    //   for c in (a_{n-1}, …, a_0):
    //     Q ← z · Q + c
    //     emit Q  → b_{n-2}, …, b_0, r
    //
    // We need the *original* `a_k` for the read at step k; stash a copy
    // (cheap — n is at most 2^24).
    let original = coeffs.to_vec();
    let mut q = F::zero();
    for k in 0..n {
        let c = original[n - 1 - k];
        q = F::add(F::mul(z, q), c);
        // scan position k → output index (n - 1 - k)
        coeffs[n - 1 - k] = q;
    }
}
