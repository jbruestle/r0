//! 31-bit Montgomery prime fields for batched arithmetic on GPU and CPU.
//!
//! Two fields are exposed: [`BabyBear`] (`p = 2^31 - 2^27 + 1`, 2-adicity
//! 27) and [`KoalaBear`] (`p = 2^31 - 2^24 + 1`, 2-adicity 24). Both are
//! [`MontyField<P>`] specialized to a zero-sized marker type implementing
//! [`MontyParameters`]. A field element is a single `u32` in Montgomery
//! form (`x · 2^32 mod p`), reduced to `[0, p)`.
//!
//! # Two ways to use it
//!
//! Host code uses the operator overloads on [`MontyField<P>`] (`+`, `-`,
//! `*`, unary `-`) plus [`MontyField::from_canonical`] and
//! [`MontyField::to_canonical`] to enter and leave Montgomery form:
//!
//! ```
//! use r0_field::BabyBear;
//! let a = BabyBear::from_canonical(7);
//! let b = BabyBear::from_canonical(11);
//! assert_eq!((a * b).to_canonical(), 77);
//! ```
//!
//! Inside `#[cube]` kernels, call the free [`monty_add()`], [`monty_sub()`],
//! [`monty_mul()`], [`monty_neg()`] functions directly on raw `u32`
//! Montgomery values. The same `#[cube]` source compiles to host Rust,
//! CUDA, WGSL, and the cubecl CPU backend — write the kernel once,
//! pick the runtime at launch.
//!
//! # Correctness
//!
//! Field constants match Plonky3's `BabyBear` and `KoalaBear` impls
//! bit-for-bit. The test suite property-tests every operation against
//! `p3-baby-bear` / `p3-koala-bear`, so divergence is caught on CI.

mod monty;

pub use monty::{
    monty_add, monty_mul, monty_neg, monty_reduce_split, monty_sub, mul_hi_u32, MontyField,
    MontyParameters,
};

mod baby_bear;
mod koala_bear;

pub use baby_bear::{BabyBear, BabyBearParameters};
pub use koala_bear::{KoalaBear, KoalaBearParameters};
