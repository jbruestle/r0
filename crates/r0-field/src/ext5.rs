//! Degree-5 binomial extension `F_p[X] / (X^5 - W)`.
//!
//! Same shape as [`crate::ext4`], extended to five components with the
//! degree-5 wraparound `X^5 → W, X^6 → W·X, X^7 → W·X^2, X^8 → W·X^3`.
//! Plonky3 only defines `BinomialExtensionData<5>` for BabyBear (`W = 2`)
//! — KoalaBear has no degree-5 binomial extension because `gcd(5, p_KB - 1) = 1`,
//! so every element is already a fifth power.

use core::marker::PhantomData;

use cubecl::prelude::*;

use crate::ext::ExtField;
use crate::monty::{monty_add, monty_mul, monty_neg, monty_sub, MontyField, MontyParameters};

/// Parameters for a degree-5 binomial extension over a base prime field.
pub trait BinomialExt5Parameters: 'static + Copy + Clone + Default + Send + Sync {
    /// Base prime field this extension sits over.
    type Base: MontyParameters;

    /// Non-residue `W` in the irreducible polynomial `X^5 - W`, in
    /// **canonical** (non-Montgomery) form.
    const W: u32;

    /// `W` in Montgomery form, derived from [`W`](Self::W) and
    /// [`Base::PRIME`](MontyParameters::PRIME).
    const W_MONT: u32 =
        ((Self::W as u64) << 32).rem_euclid(<Self::Base as MontyParameters>::PRIME as u64) as u32;
}

/// Element of the degree-5 binomial extension `F_p[X] / (X^5 - W)`.
#[derive(CubeType, Copy, Clone, Debug, Default, Eq, Hash, PartialEq)]
pub struct Ext5<P: BinomialExt5Parameters> {
    pub c0: u32,
    pub c1: u32,
    pub c2: u32,
    pub c3: u32,
    pub c4: u32,
    #[cube(comptime)]
    _p: PhantomData<P>,
}

impl<P: BinomialExt5Parameters> Ext5<P> {
    pub const ZERO: Self = Self {
        c0: 0,
        c1: 0,
        c2: 0,
        c3: 0,
        c4: 0,
        _p: PhantomData,
    };

    #[inline]
    pub const fn from_raw(raw: [u32; 5]) -> Self {
        Self {
            c0: raw[0],
            c1: raw[1],
            c2: raw[2],
            c3: raw[3],
            c4: raw[4],
            _p: PhantomData,
        }
    }

    #[inline]
    pub fn from_canonical(limbs: [u32; 5]) -> Self {
        Self {
            c0: MontyField::<P::Base>::from_canonical(limbs[0]).raw(),
            c1: MontyField::<P::Base>::from_canonical(limbs[1]).raw(),
            c2: MontyField::<P::Base>::from_canonical(limbs[2]).raw(),
            c3: MontyField::<P::Base>::from_canonical(limbs[3]).raw(),
            c4: MontyField::<P::Base>::from_canonical(limbs[4]).raw(),
            _p: PhantomData,
        }
    }

    #[inline]
    pub fn to_canonical(self) -> [u32; 5] {
        [
            MontyField::<P::Base>::from_raw(self.c0).to_canonical(),
            MontyField::<P::Base>::from_raw(self.c1).to_canonical(),
            MontyField::<P::Base>::from_raw(self.c2).to_canonical(),
            MontyField::<P::Base>::from_raw(self.c3).to_canonical(),
            MontyField::<P::Base>::from_raw(self.c4).to_canonical(),
        ]
    }

    #[inline]
    pub const fn raw(self) -> [u32; 5] {
        [self.c0, self.c1, self.c2, self.c3, self.c4]
    }
}

// ---- Host operator overloads ----

impl<P: BinomialExt5Parameters> core::ops::Add for Ext5<P> {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        ext5_add::<P>(self, rhs)
    }
}

impl<P: BinomialExt5Parameters> core::ops::Sub for Ext5<P> {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        ext5_sub::<P>(self, rhs)
    }
}

impl<P: BinomialExt5Parameters> core::ops::Mul for Ext5<P> {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        ext5_mul::<P>(self, rhs)
    }
}

impl<P: BinomialExt5Parameters> core::ops::Neg for Ext5<P> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        ext5_neg::<P>(self)
    }
}

// ---- Free `#[cube]` ops ----

/// `#[cube]`-callable constructor for [`Ext5<P>`] from raw Montgomery-form
/// limbs. The host [`Ext5::from_raw`] is `const fn` (not callable from
/// cube IR) and the struct's `_p: PhantomData<P>` is private; this free
/// function is the bridge for downstream `#[cube]` kernels.
#[cube]
pub fn ext5_from_raws<P: BinomialExt5Parameters>(
    c0: u32,
    c1: u32,
    c2: u32,
    c3: u32,
    c4: u32,
) -> Ext5<P> {
    Ext5::<P> {
        c0,
        c1,
        c2,
        c3,
        c4,
        _p: PhantomData,
    }
}

#[cube]
pub fn ext5_add<P: BinomialExt5Parameters>(a: Ext5<P>, b: Ext5<P>) -> Ext5<P> {
    Ext5::<P> {
        c0: monty_add::<P::Base>(a.c0, b.c0),
        c1: monty_add::<P::Base>(a.c1, b.c1),
        c2: monty_add::<P::Base>(a.c2, b.c2),
        c3: monty_add::<P::Base>(a.c3, b.c3),
        c4: monty_add::<P::Base>(a.c4, b.c4),
        _p: PhantomData,
    }
}

#[cube]
pub fn ext5_sub<P: BinomialExt5Parameters>(a: Ext5<P>, b: Ext5<P>) -> Ext5<P> {
    Ext5::<P> {
        c0: monty_sub::<P::Base>(a.c0, b.c0),
        c1: monty_sub::<P::Base>(a.c1, b.c1),
        c2: monty_sub::<P::Base>(a.c2, b.c2),
        c3: monty_sub::<P::Base>(a.c3, b.c3),
        c4: monty_sub::<P::Base>(a.c4, b.c4),
        _p: PhantomData,
    }
}

#[cube]
pub fn ext5_neg<P: BinomialExt5Parameters>(a: Ext5<P>) -> Ext5<P> {
    Ext5::<P> {
        c0: monty_neg::<P::Base>(a.c0),
        c1: monty_neg::<P::Base>(a.c1),
        c2: monty_neg::<P::Base>(a.c2),
        c3: monty_neg::<P::Base>(a.c3),
        c4: monty_neg::<P::Base>(a.c4),
        _p: PhantomData,
    }
}

/// Multiplication mod `X^5 - W` (schoolbook 25-mul).
#[cube]
pub fn ext5_mul<P: BinomialExt5Parameters>(a: Ext5<P>, b: Ext5<P>) -> Ext5<P> {
    let a0_b0 = monty_mul::<P::Base>(a.c0, b.c0);
    let a0_b1 = monty_mul::<P::Base>(a.c0, b.c1);
    let a0_b2 = monty_mul::<P::Base>(a.c0, b.c2);
    let a0_b3 = monty_mul::<P::Base>(a.c0, b.c3);
    let a0_b4 = monty_mul::<P::Base>(a.c0, b.c4);

    let a1_b0 = monty_mul::<P::Base>(a.c1, b.c0);
    let a1_b1 = monty_mul::<P::Base>(a.c1, b.c1);
    let a1_b2 = monty_mul::<P::Base>(a.c1, b.c2);
    let a1_b3 = monty_mul::<P::Base>(a.c1, b.c3);
    let a1_b4 = monty_mul::<P::Base>(a.c1, b.c4);

    let a2_b0 = monty_mul::<P::Base>(a.c2, b.c0);
    let a2_b1 = monty_mul::<P::Base>(a.c2, b.c1);
    let a2_b2 = monty_mul::<P::Base>(a.c2, b.c2);
    let a2_b3 = monty_mul::<P::Base>(a.c2, b.c3);
    let a2_b4 = monty_mul::<P::Base>(a.c2, b.c4);

    let a3_b0 = monty_mul::<P::Base>(a.c3, b.c0);
    let a3_b1 = monty_mul::<P::Base>(a.c3, b.c1);
    let a3_b2 = monty_mul::<P::Base>(a.c3, b.c2);
    let a3_b3 = monty_mul::<P::Base>(a.c3, b.c3);
    let a3_b4 = monty_mul::<P::Base>(a.c3, b.c4);

    let a4_b0 = monty_mul::<P::Base>(a.c4, b.c0);
    let a4_b1 = monty_mul::<P::Base>(a.c4, b.c1);
    let a4_b2 = monty_mul::<P::Base>(a.c4, b.c2);
    let a4_b3 = monty_mul::<P::Base>(a.c4, b.c3);
    let a4_b4 = monty_mul::<P::Base>(a.c4, b.c4);

    // Wraparound coefficients (X^5..X^8, each multiplied by W → c0..c3):
    //   degree-5: a1·b4 + a2·b3 + a3·b2 + a4·b1
    //   degree-6: a2·b4 + a3·b3 + a4·b2
    //   degree-7: a3·b4 + a4·b3
    //   degree-8: a4·b4
    let w5 = monty_add::<P::Base>(
        monty_add::<P::Base>(monty_add::<P::Base>(a1_b4, a2_b3), a3_b2),
        a4_b1,
    );
    let w6 = monty_add::<P::Base>(monty_add::<P::Base>(a2_b4, a3_b3), a4_b2);
    let w7 = monty_add::<P::Base>(a3_b4, a4_b3);
    let w8 = a4_b4;

    let w5_w = monty_mul::<P::Base>(w5, P::W_MONT);
    let w6_w = monty_mul::<P::Base>(w6, P::W_MONT);
    let w7_w = monty_mul::<P::Base>(w7, P::W_MONT);
    let w8_w = monty_mul::<P::Base>(w8, P::W_MONT);

    let c0 = monty_add::<P::Base>(a0_b0, w5_w);
    let c1 = monty_add::<P::Base>(monty_add::<P::Base>(a0_b1, a1_b0), w6_w);
    let c2 = monty_add::<P::Base>(
        monty_add::<P::Base>(monty_add::<P::Base>(a0_b2, a1_b1), a2_b0),
        w7_w,
    );
    let c3 = monty_add::<P::Base>(
        monty_add::<P::Base>(
            monty_add::<P::Base>(monty_add::<P::Base>(a0_b3, a1_b2), a2_b1),
            a3_b0,
        ),
        w8_w,
    );
    let c4 = monty_add::<P::Base>(
        monty_add::<P::Base>(
            monty_add::<P::Base>(monty_add::<P::Base>(a0_b4, a1_b3), a2_b2),
            a3_b1,
        ),
        a4_b0,
    );

    Ext5::<P> {
        c0,
        c1,
        c2,
        c3,
        c4,
        _p: PhantomData,
    }
}

// ---- ExtField impl ----

#[cube]
impl<P: BinomialExt5Parameters> ExtField for Ext5<P> {
    type Base = P::Base;
    const DEGREE: u32 = 5;

    fn add(a: Ext5<P>, b: Ext5<P>) -> Ext5<P> {
        ext5_add::<P>(a, b)
    }

    fn sub(a: Ext5<P>, b: Ext5<P>) -> Ext5<P> {
        ext5_sub::<P>(a, b)
    }

    fn mul(a: Ext5<P>, b: Ext5<P>) -> Ext5<P> {
        ext5_mul::<P>(a, b)
    }

    fn neg(a: Ext5<P>) -> Ext5<P> {
        ext5_neg::<P>(a)
    }

    fn zero() -> Ext5<P> {
        Ext5::<P> {
            c0: 0u32,
            c1: 0u32,
            c2: 0u32,
            c3: 0u32,
            c4: 0u32,
            _p: PhantomData,
        }
    }

    fn one() -> Ext5<P> {
        Ext5::<P> {
            c0: P::Base::MONT_ONE,
            c1: 0u32,
            c2: 0u32,
            c3: 0u32,
            c4: 0u32,
            _p: PhantomData,
        }
    }

    fn from_base_raw(x: u32) -> Ext5<P> {
        Ext5::<P> {
            c0: x,
            c1: 0u32,
            c2: 0u32,
            c3: 0u32,
            c4: 0u32,
            _p: PhantomData,
        }
    }

    fn load(arr: &Array<u32>, base: u32, i: u32, n: u32) -> Ext5<P> {
        Ext5::<P> {
            c0: arr[(base + i) as usize],
            c1: arr[(base + n + i) as usize],
            c2: arr[(base + 2 * n + i) as usize],
            c3: arr[(base + 3 * n + i) as usize],
            c4: arr[(base + 4 * n + i) as usize],
            _p: PhantomData,
        }
    }

    fn store(arr: &mut Array<u32>, base: u32, i: u32, n: u32, v: Ext5<P>) {
        arr[(base + i) as usize] = v.c0;
        arr[(base + n + i) as usize] = v.c1;
        arr[(base + 2 * n + i) as usize] = v.c2;
        arr[(base + 3 * n + i) as usize] = v.c3;
        arr[(base + 4 * n + i) as usize] = v.c4;
    }
}
