//! 31-bit Montgomery prime fields and their binomial extensions.
//!
//! # Base fields
//!
//! [`BabyBear`] (`p = 2^31 - 2^27 + 1`, 2-adicity 27) and [`KoalaBear`]
//! (`p = 2^31 - 2^24 + 1`, 2-adicity 24) are [`MontyField<P>`] specialized
//! to a zero-sized marker type implementing [`MontyParameters`]. An
//! element is one `u32` in Montgomery form (`x · 2^32 mod p`), reduced
//! to `[0, p)`.
//!
//! Host code uses the operator overloads on [`MontyField<P>`] (`+`, `-`,
//! `*`, unary `-`) plus [`MontyField::from_canonical`] and
//! [`MontyField::to_canonical`]:
//!
//! ```
//! use r0_field::BabyBear;
//! let a = BabyBear::from_canonical(7);
//! let b = BabyBear::from_canonical(11);
//! assert_eq!((a * b).to_canonical(), 77);
//! ```
//!
//! Inside `#[cube]` kernels, call the free [`monty_add()`], [`monty_sub()`],
//! [`monty_mul()`], [`monty_neg()`] functions on raw `u32` Montgomery
//! values. The same `#[cube]` source compiles to host Rust, CUDA, WGSL,
//! and the cubecl CPU backend.
//!
//! # Extension fields
//!
//! Three binomial extensions ship: [`BabyBear4`] (`X^4 - 11`), [`KoalaBear4`]
//! (`X^4 - 3`), and [`BabyBear5`] (`X^5 - 2`) — values matching Plonky3's
//! `BinomialExtensionData<D>` for the same base fields. KoalaBear has no
//! degree-5 binomial extension since `gcd(5, p_KB - 1) = 1`.
//!
//! [`Ext4<P>`] / [`Ext5<P>`] are single structs that double as host
//! wrappers (operator overloads, `from_canonical([u32; D])`) and as the
//! `CubeType` flowing through `#[cube]` kernels. Polynomials of an
//! extension element are stored **transposed**: component `c` of element
//! `i` at offset `c·N + i` in a buffer of `N` logical elements. This
//! layout makes a length-`N` extension polynomial bitwise identical to
//! `D` length-`N` base polynomials, which is why `r0-ntt`'s
//! `forward_ext` / `inverse_ext` need no new kernel.
//!
//! # Generic-over-field code
//!
//! [`ExtField`] is the `#[cube] trait` for code that wants to be generic
//! over the inner field — polynomial division, evaluation, folding, and
//! anything else that cares about polynomial structure but not whether
//! the coefficients are base-field or extension. [`BaseElem<P>`]
//! implements [`ExtField`] with `DEGREE = 1`, so a single
//! `<F: ExtField>` kernel covers both cases.
//!
//! # Correctness
//!
//! Base-field constants and the extensions' `W` cross-check against
//! Plonky3 in `tests/ext_p3_oracle.rs`; arithmetic round-trips via
//! property tests there and in `tests/cube_smoke.rs` /
//! `tests/ext_cube_smoke.rs` for the cube path on every backend.

mod monty;

pub use monty::{
    monty_add, monty_mul, monty_neg, monty_reduce_split, monty_sub, mul_hi_u32, MontyField,
    MontyParameters,
};

mod baby_bear;
mod koala_bear;

pub use baby_bear::{
    BabyBear, BabyBear4, BabyBear4Parameters, BabyBear5, BabyBear5Parameters, BabyBearParameters,
};
pub use koala_bear::{KoalaBear, KoalaBear4, KoalaBear4Parameters, KoalaBearParameters};

mod ext;
pub use ext::{BaseElem, ExtField};

mod ext4;
pub use ext4::{ext4_add, ext4_mul, ext4_neg, ext4_sub, BinomialExt4Parameters, Ext4};

mod ext5;
pub use ext5::{ext5_add, ext5_mul, ext5_neg, ext5_sub, BinomialExt5Parameters, Ext5};
