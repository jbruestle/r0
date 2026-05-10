//! Degree-4 binomial extension `F_p[X] / (X^4 - W)`.
//!
//! `Ext4<P>` is a single struct that serves both as the host-side
//! wrapper (operator overloads, [`from_canonical`](Ext4::from_canonical),
//! [`to_canonical`](Ext4::to_canonical)) and as the cubecl `CubeType` flowing
//! through `#[cube]` kernels. The four limbs `c0..c3` are the components of
//! `c0 + c1·X + c2·X^2 + c3·X^3` in Montgomery form (one base-field reduction
//! per limb, never four-limb-wide).
//!
//! Multiplication is a textbook 16-mul schoolbook with the wraparound terms
//! (degrees 4..6 of the unreduced product) folded in via a single Montgomery
//! multiply by `W_MONT`. We do not do Karatsuba here — the overhead matters
//! less than the readability and cubecl emits the same reductions either way.
//!
//! See [`BinomialExt4Parameters`] for the parameter trait, and the
//! per-base-field instances in [`crate::baby_bear`] / [`crate::koala_bear`].

use core::marker::PhantomData;

use cubecl::prelude::*;

use crate::ext::ExtField;
use crate::monty::{monty_add, monty_mul, monty_neg, monty_sub, MontyField, MontyParameters};

/// Parameters for a degree-4 binomial extension over a base prime field.
pub trait BinomialExt4Parameters: 'static + Copy + Clone + Default + Send + Sync {
    /// Base prime field this extension sits over.
    type Base: MontyParameters;

    /// Non-residue `W` in the irreducible polynomial `X^4 - W`, in
    /// **canonical** (non-Montgomery) form. Plonky3's
    /// `BinomialExtensionData<4>::W` for the same base is the cross-check.
    const W: u32;

    /// `W` in Montgomery form, derived from [`W`](Self::W) and
    /// [`Base::PRIME`](MontyParameters::PRIME). Don't override unless you
    /// want to hand-precompute and cross-check.
    const W_MONT: u32 =
        ((Self::W as u64) << 32).rem_euclid(<Self::Base as MontyParameters>::PRIME as u64) as u32;
}

/// Element of the degree-4 binomial extension `F_p[X] / (X^4 - W)`.
///
/// Limbs are in Montgomery form (`x · 2^32 mod p`, reduced to `[0, p)`).
/// On host, treat as a value type with the usual operator overloads. In
/// `#[cube]` bodies this is the `CubeType` flowing through; the
/// arithmetic is also exposed as the free [`ext4_add()`], [`ext4_sub()`],
/// [`ext4_mul()`], [`ext4_neg()`] functions for code that prefers them.
#[derive(CubeType, Copy, Clone, Debug, Default, Eq, Hash, PartialEq)]
pub struct Ext4<P: BinomialExt4Parameters> {
    /// Coefficient of `X^0`.
    pub c0: u32,
    /// Coefficient of `X^1`.
    pub c1: u32,
    /// Coefficient of `X^2`.
    pub c2: u32,
    /// Coefficient of `X^3`.
    pub c3: u32,
    #[cube(comptime)]
    _p: PhantomData<P>,
}

impl<P: BinomialExt4Parameters> Ext4<P> {
    /// Additive identity.
    pub const ZERO: Self = Self {
        c0: 0,
        c1: 0,
        c2: 0,
        c3: 0,
        _p: PhantomData,
    };

    /// Wrap pre-Montgomery raw limbs directly. Each must be `< p`.
    #[inline]
    pub const fn from_raw(raw: [u32; 4]) -> Self {
        Self {
            c0: raw[0],
            c1: raw[1],
            c2: raw[2],
            c3: raw[3],
            _p: PhantomData,
        }
    }

    /// Construct from canonical (non-Montgomery) limbs. Each is reduced
    /// `mod p` and lifted to Montgomery form.
    #[inline]
    pub fn from_canonical(limbs: [u32; 4]) -> Self {
        Self {
            c0: MontyField::<P::Base>::from_canonical(limbs[0]).raw(),
            c1: MontyField::<P::Base>::from_canonical(limbs[1]).raw(),
            c2: MontyField::<P::Base>::from_canonical(limbs[2]).raw(),
            c3: MontyField::<P::Base>::from_canonical(limbs[3]).raw(),
            _p: PhantomData,
        }
    }

    /// Inverse of [`from_canonical`](Self::from_canonical).
    #[inline]
    pub fn to_canonical(self) -> [u32; 4] {
        [
            MontyField::<P::Base>::from_raw(self.c0).to_canonical(),
            MontyField::<P::Base>::from_raw(self.c1).to_canonical(),
            MontyField::<P::Base>::from_raw(self.c2).to_canonical(),
            MontyField::<P::Base>::from_raw(self.c3).to_canonical(),
        ]
    }

    /// The four raw Montgomery-form limbs.
    #[inline]
    pub const fn raw(self) -> [u32; 4] {
        [self.c0, self.c1, self.c2, self.c3]
    }
}

// ---- Host operator overloads forward to free `#[cube]` fns ----

impl<P: BinomialExt4Parameters> core::ops::Add for Ext4<P> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        ext4_add::<P>(self, rhs)
    }
}

impl<P: BinomialExt4Parameters> core::ops::Sub for Ext4<P> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        ext4_sub::<P>(self, rhs)
    }
}

impl<P: BinomialExt4Parameters> core::ops::Mul for Ext4<P> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        ext4_mul::<P>(self, rhs)
    }
}

impl<P: BinomialExt4Parameters> core::ops::Neg for Ext4<P> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        ext4_neg::<P>(self)
    }
}

// ---- Free #[cube] ops (single source for host AND cube) ----

/// Componentwise addition.
#[cube]
pub fn ext4_add<P: BinomialExt4Parameters>(a: Ext4<P>, b: Ext4<P>) -> Ext4<P> {
    Ext4::<P> {
        c0: monty_add::<P::Base>(a.c0, b.c0),
        c1: monty_add::<P::Base>(a.c1, b.c1),
        c2: monty_add::<P::Base>(a.c2, b.c2),
        c3: monty_add::<P::Base>(a.c3, b.c3),
        _p: PhantomData,
    }
}

/// Componentwise subtraction.
#[cube]
pub fn ext4_sub<P: BinomialExt4Parameters>(a: Ext4<P>, b: Ext4<P>) -> Ext4<P> {
    Ext4::<P> {
        c0: monty_sub::<P::Base>(a.c0, b.c0),
        c1: monty_sub::<P::Base>(a.c1, b.c1),
        c2: monty_sub::<P::Base>(a.c2, b.c2),
        c3: monty_sub::<P::Base>(a.c3, b.c3),
        _p: PhantomData,
    }
}

/// Componentwise negation.
#[cube]
pub fn ext4_neg<P: BinomialExt4Parameters>(a: Ext4<P>) -> Ext4<P> {
    Ext4::<P> {
        c0: monty_neg::<P::Base>(a.c0),
        c1: monty_neg::<P::Base>(a.c1),
        c2: monty_neg::<P::Base>(a.c2),
        c3: monty_neg::<P::Base>(a.c3),
        _p: PhantomData,
    }
}

/// Multiplication mod `X^4 - W`.
///
/// Schoolbook 16-mul: form the unreduced degree-6 product, fold the
/// degree-4..6 wraparound terms back via a single Montgomery multiply by
/// `W_MONT`. All arithmetic is base-field Montgomery.
#[cube]
pub fn ext4_mul<P: BinomialExt4Parameters>(a: Ext4<P>, b: Ext4<P>) -> Ext4<P> {
    let a0_b0 = monty_mul::<P::Base>(a.c0, b.c0);
    let a0_b1 = monty_mul::<P::Base>(a.c0, b.c1);
    let a0_b2 = monty_mul::<P::Base>(a.c0, b.c2);
    let a0_b3 = monty_mul::<P::Base>(a.c0, b.c3);

    let a1_b0 = monty_mul::<P::Base>(a.c1, b.c0);
    let a1_b1 = monty_mul::<P::Base>(a.c1, b.c1);
    let a1_b2 = monty_mul::<P::Base>(a.c1, b.c2);
    let a1_b3 = monty_mul::<P::Base>(a.c1, b.c3);

    let a2_b0 = monty_mul::<P::Base>(a.c2, b.c0);
    let a2_b1 = monty_mul::<P::Base>(a.c2, b.c1);
    let a2_b2 = monty_mul::<P::Base>(a.c2, b.c2);
    let a2_b3 = monty_mul::<P::Base>(a.c2, b.c3);

    let a3_b0 = monty_mul::<P::Base>(a.c3, b.c0);
    let a3_b1 = monty_mul::<P::Base>(a.c3, b.c1);
    let a3_b2 = monty_mul::<P::Base>(a.c3, b.c2);
    let a3_b3 = monty_mul::<P::Base>(a.c3, b.c3);

    // Wraparound coefficients (X^4 → W, X^5 → W·X, X^6 → W·X^2):
    //   degree-4: a1·b3 + a2·b2 + a3·b1   → folds into c0
    //   degree-5: a2·b3 + a3·b2           → folds into c1
    //   degree-6: a3·b3                   → folds into c2
    let w4 = monty_add::<P::Base>(monty_add::<P::Base>(a1_b3, a2_b2), a3_b1);
    let w5 = monty_add::<P::Base>(a2_b3, a3_b2);
    let w6 = a3_b3;

    let w4_w = monty_mul::<P::Base>(w4, P::W_MONT);
    let w5_w = monty_mul::<P::Base>(w5, P::W_MONT);
    let w6_w = monty_mul::<P::Base>(w6, P::W_MONT);

    let c0 = monty_add::<P::Base>(a0_b0, w4_w);
    let c1 = monty_add::<P::Base>(monty_add::<P::Base>(a0_b1, a1_b0), w5_w);
    let c2 = monty_add::<P::Base>(
        monty_add::<P::Base>(monty_add::<P::Base>(a0_b2, a1_b1), a2_b0),
        w6_w,
    );
    let c3 = monty_add::<P::Base>(
        monty_add::<P::Base>(monty_add::<P::Base>(a0_b3, a1_b2), a2_b1),
        a3_b0,
    );

    Ext4::<P> {
        c0,
        c1,
        c2,
        c3,
        _p: PhantomData,
    }
}

// ---- ExtField impl ----

#[cube]
impl<P: BinomialExt4Parameters> ExtField for Ext4<P> {
    type Base = P::Base;
    const DEGREE: u32 = 4;

    fn add(a: Ext4<P>, b: Ext4<P>) -> Ext4<P> {
        ext4_add::<P>(a, b)
    }

    fn sub(a: Ext4<P>, b: Ext4<P>) -> Ext4<P> {
        ext4_sub::<P>(a, b)
    }

    fn mul(a: Ext4<P>, b: Ext4<P>) -> Ext4<P> {
        ext4_mul::<P>(a, b)
    }

    fn neg(a: Ext4<P>) -> Ext4<P> {
        ext4_neg::<P>(a)
    }

    fn zero() -> Ext4<P> {
        Ext4::<P> {
            c0: 0u32,
            c1: 0u32,
            c2: 0u32,
            c3: 0u32,
            _p: PhantomData,
        }
    }

    fn one() -> Ext4<P> {
        Ext4::<P> {
            c0: P::Base::MONT_ONE,
            c1: 0u32,
            c2: 0u32,
            c3: 0u32,
            _p: PhantomData,
        }
    }

    fn from_base_raw(x: u32) -> Ext4<P> {
        Ext4::<P> {
            c0: x,
            c1: 0u32,
            c2: 0u32,
            c3: 0u32,
            _p: PhantomData,
        }
    }

    fn load(arr: &Array<u32>, base: u32, i: u32, n: u32) -> Ext4<P> {
        Ext4::<P> {
            c0: arr[(base + i) as usize],
            c1: arr[(base + n + i) as usize],
            c2: arr[(base + 2 * n + i) as usize],
            c3: arr[(base + 3 * n + i) as usize],
            _p: PhantomData,
        }
    }

    fn store(arr: &mut Array<u32>, base: u32, i: u32, n: u32, v: Ext4<P>) {
        arr[(base + i) as usize] = v.c0;
        arr[(base + n + i) as usize] = v.c1;
        arr[(base + 2 * n + i) as usize] = v.c2;
        arr[(base + 3 * n + i) as usize] = v.c3;
    }
}
