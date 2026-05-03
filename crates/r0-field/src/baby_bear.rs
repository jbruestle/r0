//! BabyBear: `p = 2^31 - 2^27 + 1 = 0x78000001`. 2-adicity 27.
//!
//! Constants below match Plonky3's `BabyBearParameters` and `TwoAdicData`
//! impl bit-for-bit (see `~/src/Plonky3/baby-bear/src/baby_bear.rs:18-51`),
//! enabling direct cross-checks against `p3-baby-bear` in tests.

use crate::monty::{MontyField, MontyParameters};

#[derive(Copy, Clone, Default, Debug, Eq, Hash, PartialEq)]
pub struct BabyBearParameters;

impl MontyParameters for BabyBearParameters {
    const PRIME: u32 = 0x78000001;
    /// `-p^{-1} mod 2^32` (additive form). Plonky3's positive-form
    /// `MONTY_MU = 0x88000001`; this is `2^32 - that = 0x77FFFFFF`.
    const MU: u32 = 0x77ffffff;
    const R2: u32 = 0x45dddde3;
    const TWO_ADICITY: u32 = 27;
    const TWO_ADIC_GENERATORS: &'static [u32] = &[
        0x1, 0x78000000, 0x67055c21, 0x5ee99486, 0x0bb4c4e4, 0x2d4cc4da, 0x669d6090, 0x17b56c64,
        0x67456167, 0x688442f9, 0x145e952d, 0x4fe61226, 0x4c734715, 0x11c33e2a, 0x62c3d2b1,
        0x77cad399, 0x54c131f4, 0x4cabd6a6, 0x5cf5713f, 0x3e9430e8, 0x0ba067a3, 0x18adc27d,
        0x21fd55bc, 0x4b859b3d, 0x3bd57996, 0x4483d85a, 0x3a26eef8, 0x1a427a41,
    ];
}

pub type BabyBear = MontyField<BabyBearParameters>;
