//! Generic 31-bit Montgomery field, single-sourced via cubecl `#[cube]`.
//!
//! Every arithmetic op below has one definition that compiles to host
//! Rust, CUDA, WGSL, and the cubecl CPU backend. Operator overloads on
//! [`MontyField`] forward to those same `#[cube]` functions.
//!
//! # cubecl 0.9 quirks accommodated
//!
//! - `u32::mul_hi` panics on host (its host impl is `unexpanded!()`).
//!   We bridge it with [`mul_hi_u32`], which has a real host body *plus*
//!   a hand-written `expand` companion module that delegates to cubecl's
//!   `__expand_mul_hi`. The cubecl macro rewrites a call `foo(a, b)`
//!   into `foo::expand(scope, a, b)`, so this single name resolves in
//!   both contexts (function in value namespace, module in type
//!   namespace).
//! - cubecl 0.9 has no `From<u64> for ConstantValue`, so we never form
//!   u64 literals or u64 trait constants. All u64 in the body comes
//!   from a runtime `as u64` cast on a u32 local — done only inside
//!   [`mul_hi_u32`]'s host body, never inside `#[cube]` IR.
//! - `[profile.dev] overflow-checks = false` is set workspace-wide so
//!   plain `*`/`+`/`-` wrap consistently between host Rust and cube IR.

use core::marker::PhantomData;
use cubecl::prelude::*;

/// Compile-time description of a 31-bit Montgomery prime field.
///
/// Implement on a zero-sized marker type to define a new field; pair
/// with [`MontyField<Self>`] for elements. Two implementations are
/// provided, [`crate::BabyBearParameters`] and
/// [`crate::KoalaBearParameters`]; you should rarely need to define
/// your own.
///
/// Constants are not validated against each other — they must be
/// mutually consistent. The cross-check against Plonky3 is the test
/// suite's job.
pub trait MontyParameters: Copy + Clone + Default + Send + Sync + 'static {
    /// The prime modulus `p`. Must satisfy `p < 2^31` (so `2p < 2^32`).
    const PRIME: u32;

    /// `-p^{-1} mod 2^32` (additive form). Plonky3's positive-form
    /// `MONTY_MU = +p^{-1} mod 2^32`; ours is `2^32 - that`.
    const MU: u32;

    /// `R^2 mod p` where `R = 2^32`. Used to lift canonical → Montgomery.
    const R2: u32;

    /// 2-adicity: `s` such that `2^s | p - 1` and `2^(s+1) ∤ p - 1`.
    const TWO_ADICITY: u32;

    /// `TWO_ADIC_GENERATORS[k]` is a primitive `2^k`-th root of unity in
    /// **canonical (non-Montgomery) form**, for `k ∈ 0..=TWO_ADICITY`.
    /// Index 0 is `1`. Matches Plonky3's `TwoAdicData::TWO_ADIC_GENERATORS`
    /// values verbatim — convert with [`MontyField::from_canonical`] at use.
    const TWO_ADIC_GENERATORS: &'static [u32];
}

/// A field element of the prime field defined by `P`.
///
/// Internally a single `u32` holding the value in Montgomery form
/// (`x · 2^32 mod p`, reduced to `[0, p)`). Build from a canonical
/// integer via [`from_canonical`](Self::from_canonical) and read the
/// canonical representative back via
/// [`to_canonical`](Self::to_canonical). Add/sub/mul/neg work as
/// operator overloads on the host; inside `#[cube]` kernels, the same
/// arithmetic is available as the free [`monty_add()`], [`monty_sub()`],
/// [`monty_mul()`], [`monty_neg()`] functions on the underlying `u32`.
#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq, Default)]
#[repr(transparent)]
pub struct MontyField<P: MontyParameters> {
    raw: u32,
    _p: PhantomData<P>,
}

impl<P: MontyParameters> MontyField<P> {
    /// The additive identity (`0` in canonical form).
    pub const ZERO: Self = Self {
        raw: 0,
        _p: PhantomData,
    };

    /// Wrap an already-Montgomery-form `u32` directly. Debug-asserts
    /// `raw < p`. Most callers want [`from_canonical`](Self::from_canonical).
    #[inline]
    pub const fn from_raw(raw: u32) -> Self {
        debug_assert!(raw < P::PRIME);
        Self { raw, _p: PhantomData }
    }

    /// Construct from a canonical (non-Montgomery) `u32`. Reduces `mod p`.
    #[inline]
    pub fn from_canonical(x: u32) -> Self {
        Self {
            raw: monty_mul::<P>(x % P::PRIME, P::R2),
            _p: PhantomData,
        }
    }

    /// Inverse of [`from_canonical`](Self::from_canonical): returns the
    /// canonical (non-Montgomery) `u32` representative in `[0, p)`.
    #[inline]
    pub fn to_canonical(self) -> u32 {
        // `monty_reduce_split(0, self.raw)` = `self.raw · R^{-1} mod p`.
        monty_reduce_split::<P>(0, self.raw)
    }

    /// Underlying Montgomery-form `u32`. Use to bridge to kernel input
    /// buffers (e.g. `client.create_from_slice` of raw `u32`s).
    #[inline]
    pub const fn raw(self) -> u32 {
        self.raw
    }
}

// ---- Operator overloads forward to the #[cube] free functions ----

impl<P: MontyParameters> core::ops::Add for MontyField<P> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self { raw: monty_add::<P>(self.raw, rhs.raw), _p: PhantomData }
    }
}

impl<P: MontyParameters> core::ops::Sub for MontyField<P> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self { raw: monty_sub::<P>(self.raw, rhs.raw), _p: PhantomData }
    }
}

impl<P: MontyParameters> core::ops::Mul for MontyField<P> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self { raw: monty_mul::<P>(self.raw, rhs.raw), _p: PhantomData }
    }
}

impl<P: MontyParameters> core::ops::Neg for MontyField<P> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self { raw: monty_neg::<P>(self.raw), _p: PhantomData }
    }
}

// ---- mul_hi bridge: host body + sibling `expand` module ----
//
// `u32::mul_hi` from cubecl-core panics if called from regular Rust
// (its host impl is `unexpanded!()`). We need a name that works in
// both contexts. cubecl-macros rewrites `foo(a, b)` inside a `#[cube]`
// body into `foo::expand(scope, a, b)`, resolving the call through the
// type namespace. So we pair a free `fn mul_hi_u32` (value namespace,
// real host body) with a sibling `mod mul_hi_u32` (type namespace,
// containing the IR-builder `expand`).

/// `(a · b) >> 32` — the high half of a 32-bit multiply.
///
/// Bridge for cubecl 0.9 quirks (see this module's source): pairs a
/// real host body with a sibling `mul_hi_u32::expand` module so the
/// same name resolves both in regular Rust and inside `#[cube]`
/// kernels. You should rarely call this directly — [`monty_mul()`]
/// already invokes it as part of Montgomery reduction.
#[inline]
#[allow(clippy::cast_possible_truncation)]
pub fn mul_hi_u32(a: u32, b: u32) -> u32 {
    (((a as u64) * (b as u64)) >> 32) as u32
}

#[allow(non_snake_case)]
pub mod mul_hi_u32 {
    use cubecl::ir::Scope;
    use cubecl::prelude::*;

    pub fn expand(
        scope: &mut Scope,
        a: ExpandElementTyped<u32>,
        b: ExpandElementTyped<u32>,
    ) -> ExpandElementTyped<u32> {
        u32::__expand_mul_hi(scope, a, b)
    }
}

// ---- Free `#[cube]` operations — single source for host AND cube ----
//
// All four operate on raw Montgomery `u32` values; inputs and outputs
// are in `[0, p)`. They're the kernel-side counterpart to the operator
// overloads on `MontyField<P>` — host code can use either form, but
// `#[cube]` bodies must call these directly.

/// `(a + b) mod p` on raw Montgomery `u32` values.
#[cube]
pub fn monty_add<P: MontyParameters>(a: u32, b: u32) -> u32 {
    let s = a + b;
    if s >= P::PRIME { s - P::PRIME } else { s }
}

/// `(a - b) mod p` on raw Montgomery `u32` values.
#[cube]
pub fn monty_sub<P: MontyParameters>(a: u32, b: u32) -> u32 {
    if a >= b { a - b } else { (a + P::PRIME) - b }
}

/// `-a mod p` on a raw Montgomery `u32` value.
#[cube]
pub fn monty_neg<P: MontyParameters>(a: u32) -> u32 {
    let neg = P::PRIME - a;
    if neg >= P::PRIME { neg - P::PRIME } else { neg }
}

/// Montgomery multiplication: `(a · b · R^{-1}) mod p` with `R = 2^32`.
///
/// Computes the full 64-bit product as `(hi, lo)` and feeds it to
/// [`monty_reduce_split()`]. On native CUDA `mul_hi` lowers to a single
/// `mul.hi.u32`; on WGSL it emulates via a schoolbook split (~10 ops).
#[cube]
pub fn monty_mul<P: MontyParameters>(a: u32, b: u32) -> u32 {
    let lo = a * b;                  // (a·b) mod 2^32 (wrapping)
    let hi = mul_hi_u32(a, b);       // (a·b) >> 32
    monty_reduce_split::<P>(hi, lo)
}

/// Montgomery reduction on a 64-bit value passed as `(hi, lo)`.
///
/// Returns `((hi << 32) | lo) · R^{-1} mod p` reduced to `[0, p)`.
/// Exposed for kernels that accumulate `u64` intermediates manually
/// before reducing; for a plain `a · b` reduction, use [`monty_mul()`].
#[cube]
pub fn monty_reduce_split<P: MontyParameters>(hi: u32, lo: u32) -> u32 {
    // Step 1: t such that lo + t·p ≡ 0 (mod 2^32), via additive MU.
    let t = lo * P::MU;
    // Step 2: high half of t·p.
    let u_hi = mul_hi_u32(t, P::PRIME);
    // Step 3: branchless carry. `lo + (-lo mod 2^32)` is 0 if lo == 0
    // and 2^32 otherwise, so the top bit of `lo | (0 - lo)` is the
    // carry bit. Avoids the `if … { 0u32 } else { 1u32 }` literal-in-
    // branch issue with cubecl 0.9.
    let neg_lo = 0u32 - lo;
    let carry = (lo | neg_lo) >> 31;
    // Step 4: combine. For 31-bit primes, hi < 2^30 and u_hi < 2^31, so
    // hi + u_hi + carry < 2^32 — no overflow.
    let result = hi + u_hi + carry;
    // Step 5: bring into [0, p). Output is in [0, 2p); for our primes
    // 2p < 2^32 so a single subtract suffices.
    if result >= P::PRIME { result - P::PRIME } else { result }
}
