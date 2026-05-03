//! 31-bit Montgomery prime fields for r0-ntt.
//!
//! Two fields are exposed: [`BabyBear`] (`p = 2^31 - 2^27 + 1`, 2-adicity 27)
//! and [`KoalaBear`] (`p = 2^31 - 2^24 + 1`, 2-adicity 24). Both are
//! generic instantiations of [`MontyField<P>`] over a marker type
//! implementing [`MontyParameters`].

mod monty;

pub use monty::{
    monty_add, monty_mul, monty_neg, monty_reduce_split, monty_sub, mul_hi_u32, MontyField,
    MontyParameters,
};

mod baby_bear;
mod koala_bear;

pub use baby_bear::{BabyBear, BabyBearParameters};
pub use koala_bear::{KoalaBear, KoalaBearParameters};
