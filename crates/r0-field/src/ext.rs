//! Field-shape abstraction layer.
//!
//! Most code in `r0-field` operates on the base prime field directly via
//! [`MontyParameters`] + free `monty_*` `#[cube] fn`s. That's the right
//! shape for kernels that natively take base-field elements (e.g. NTT).
//!
//! Other kernels — polynomial division, evaluation, folding — care about
//! the polynomial structure but not whether the *coefficients* are in the
//! base field or in an extension. [`ExtField`] is the abstraction those
//! kernels are generic over: a `#[cube] trait` carrying degree, arithmetic,
//! and load/store helpers honoring the transposed memory layout
//! (component `c` of element `i` lives at offset `c·N + i` in a buffer of
//! `N` logical elements).
//!
//! [`BaseElem<P>`] implements `ExtField` with `DEGREE = 1`, letting a
//! single `<F: ExtField>` kernel cover both base-field and extension-field
//! polynomials. It's a `u32` newtype tagged by `P` for type identity; the
//! generated code is identical to operating on bare `u32`s.
//!
//! See [`crate::ext4`] and [`crate::ext5`] for the binomial extensions.

use core::marker::PhantomData;

use cubecl::prelude::*;

use crate::monty::{monty_add, monty_mul, monty_neg, monty_sub, MontyParameters};

/// Common shape for kernels that are generic over a base or extension field.
///
/// All implementations agree on:
///
/// - [`DEGREE`](Self::DEGREE) — number of base-field limbs per element.
/// - Arithmetic ops — [`add`](Self::add), [`sub`](Self::sub),
///   [`mul`](Self::mul), [`neg`](Self::neg), [`zero`](Self::zero),
///   [`one`](Self::one).
/// - Promotion — [`from_base_raw`](Self::from_base_raw) lifts a single
///   Montgomery-form `u32` into the field (extension elements get zero in
///   the higher components).
/// - Memory I/O — [`load`](Self::load) / [`store`](Self::store) read and
///   write a single logical element at index `i` from a transposed-layout
///   buffer of `n` logical elements, starting at `u32` offset `base`. For
///   `DEGREE = 1` this collapses to `arr[base + i]`; for `DEGREE = D` it
///   touches `D` u32s at stride `n`.
///
/// Implemented by [`BaseElem`] (degree 1), [`crate::Ext4`] (degree 4),
/// and [`crate::Ext5`] (degree 5).
#[cube]
pub trait ExtField: CubeType + Copy + Clone + Sized + Send + Sync + 'static {
    /// Base prime field. Lets callers constrain `F: ExtField<Base = P>`
    /// to bind an extension type to a specific base — used by e.g.
    /// `NttExec::forward_ext` to refuse a `BabyBear4` element fed to a
    /// `KoalaBear` executor.
    type Base: MontyParameters;

    /// Number of base-field limbs per element.
    const DEGREE: u32;

    /// Field addition.
    fn add(a: Self, b: Self) -> Self;
    /// Field subtraction.
    fn sub(a: Self, b: Self) -> Self;
    /// Field multiplication.
    fn mul(a: Self, b: Self) -> Self;
    /// Field negation.
    fn neg(a: Self) -> Self;
    /// Additive identity.
    fn zero() -> Self;
    /// Multiplicative identity.
    fn one() -> Self;

    /// Promote a base-field value (in Montgomery form) to a field element.
    /// For extensions, fills the higher components with zero.
    fn from_base_raw(x: u32) -> Self;

    /// Read element `i` from a transposed-layout buffer of `n` logical
    /// elements starting at u32 offset `base`. Reads `DEGREE` u32s at
    /// stride `n`.
    fn load(arr: &Array<u32>, base: u32, i: u32, n: u32) -> Self;

    /// Write element `i` into a transposed-layout buffer; inverse of
    /// [`load`](Self::load).
    fn store(arr: &mut Array<u32>, base: u32, i: u32, n: u32, v: Self);
}

// ---------------------------------------------------------------------------
// BaseElem<P> — degree-1 ExtField wrapping the base field.
// ---------------------------------------------------------------------------

/// A base-field element tagged by its parameters, used for `<F: ExtField>`
/// dispatch over the base field.
///
/// Carries no runtime overhead: `size_of::<BaseElem<P>>() == 4` and every
/// op delegates to a `monty_*` free function. This exists only so that a
/// generic `<F: ExtField>` kernel can also operate on plain base-field
/// polynomials.
#[derive(CubeType, Copy, Clone, Debug, Default, Eq, Hash, PartialEq)]
pub struct BaseElem<P: MontyParameters> {
    /// Raw Montgomery-form value. Same shape as [`MontyField::raw()`].
    ///
    /// [`MontyField::raw()`]: crate::MontyField::raw
    pub raw: u32,
    #[cube(comptime)]
    _p: PhantomData<P>,
}

impl<P: MontyParameters> BaseElem<P> {
    /// Wrap an already-Montgomery-form `u32` directly.
    #[inline]
    pub const fn from_raw(raw: u32) -> Self {
        Self {
            raw,
            _p: PhantomData,
        }
    }
}

/// `#[cube]`-callable constructor for [`BaseElem<P>`] from a raw
/// Montgomery-form `u32`. The host [`BaseElem::from_raw`] is `const fn`
/// (not callable from cube IR) and the struct's `_p: PhantomData<P>` is
/// private; this free function is the bridge for downstream `#[cube]`
/// kernels (e.g. `r0-polynomial`'s `PairScanLayout::unpack`).
#[cube]
pub fn base_elem_from_raw<P: MontyParameters>(raw: u32) -> BaseElem<P> {
    BaseElem::<P> {
        raw,
        _p: PhantomData,
    }
}

#[cube]
impl<P: MontyParameters> ExtField for BaseElem<P> {
    type Base = P;
    const DEGREE: u32 = 1;

    fn add(a: BaseElem<P>, b: BaseElem<P>) -> BaseElem<P> {
        BaseElem::<P> {
            raw: monty_add::<P>(a.raw, b.raw),
            _p: PhantomData,
        }
    }

    fn sub(a: BaseElem<P>, b: BaseElem<P>) -> BaseElem<P> {
        BaseElem::<P> {
            raw: monty_sub::<P>(a.raw, b.raw),
            _p: PhantomData,
        }
    }

    fn mul(a: BaseElem<P>, b: BaseElem<P>) -> BaseElem<P> {
        BaseElem::<P> {
            raw: monty_mul::<P>(a.raw, b.raw),
            _p: PhantomData,
        }
    }

    fn neg(a: BaseElem<P>) -> BaseElem<P> {
        BaseElem::<P> {
            raw: monty_neg::<P>(a.raw),
            _p: PhantomData,
        }
    }

    fn zero() -> BaseElem<P> {
        BaseElem::<P> {
            raw: 0u32,
            _p: PhantomData,
        }
    }

    fn one() -> BaseElem<P> {
        BaseElem::<P> {
            raw: P::MONT_ONE,
            _p: PhantomData,
        }
    }

    fn from_base_raw(x: u32) -> BaseElem<P> {
        BaseElem::<P> {
            raw: x,
            _p: PhantomData,
        }
    }

    fn load(arr: &Array<u32>, base: u32, i: u32, _n: u32) -> BaseElem<P> {
        BaseElem::<P> {
            raw: arr[(base + i) as usize],
            _p: PhantomData,
        }
    }

    fn store(arr: &mut Array<u32>, base: u32, i: u32, _n: u32, v: BaseElem<P>) {
        arr[(base + i) as usize] = v.raw;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BabyBearParameters, KoalaBearParameters};

    #[test]
    fn base_elem_is_zero_cost() {
        assert_eq!(core::mem::size_of::<BaseElem<BabyBearParameters>>(), 4);
        assert_eq!(core::mem::align_of::<BaseElem<BabyBearParameters>>(), 4);
        assert_eq!(core::mem::size_of::<BaseElem<KoalaBearParameters>>(), 4);
    }
}
