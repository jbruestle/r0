//! KoalaBear: `p = 2^31 - 2^24 + 1 = 0x7f000001`. 2-adicity 24.
//!
//! Constants below match Plonky3's `KoalaBearParameters` and `TwoAdicData`
//! impl bit-for-bit (see `~/src/Plonky3/koala-bear/src/koala_bear.rs:21-76`),
//! enabling direct cross-checks against `p3-koala-bear` in tests.

use crate::ext4::{BinomialExt4Parameters, Ext4};
use crate::monty::{MontyField, MontyParameters};

/// Marker type carrying [`MontyParameters`] for the KoalaBear field.
/// Pair with [`MontyField`] to get [`KoalaBear`].
#[derive(Copy, Clone, Default, Debug, Eq, Hash, PartialEq)]
pub struct KoalaBearParameters;

impl MontyParameters for KoalaBearParameters {
    const PRIME: u32 = 0x7f000001;
    /// `-p^{-1} mod 2^32` (additive form). Plonky3's positive-form
    /// `MONTY_MU = 0x81000001`; this is `2^32 - that = 0x7EFFFFFF`.
    const MU: u32 = 0x7effffff;
    const R2: u32 = 0x17f7efe4;
    const MONT_ONE: u32 = (((1u64 << 32) % Self::PRIME as u64) as u32);
    const TWO_ADICITY: u32 = 24;
    const TWO_ADIC_GENERATORS: &'static [u32] = &[
        0x1, 0x7f000000, 0x7e010002, 0x6832fe4a, 0x08dbd69c, 0x0a28f031, 0x5c4a5b99, 0x29b75a80,
        0x17668b8a, 0x27ad539b, 0x334d48c7, 0x7744959c, 0x768fc6fa, 0x303964b2, 0x3e687d4d,
        0x45a60e61, 0x6e2f4d7a, 0x163bd499, 0x6c4a8a45, 0x143ef899, 0x514ddcad, 0x484ef19b,
        0x205d63c3, 0x68e7dd49, 0x6ac49f88,
    ];
}

/// KoalaBear field element: shorthand for [`MontyField<KoalaBearParameters>`].
pub type KoalaBear = MontyField<KoalaBearParameters>;

/// Degree-4 binomial extension parameters: `F_p[X] / (X^4 - 3)`. Matches
/// Plonky3's `<KoalaBearParameters as BinomialExtensionData<4>>::W = 3`.
///
/// (KoalaBear has no degree-5 binomial extension: `gcd(5, p_KB - 1) = 1`,
/// so every element of `F_{p_KB}` is already a fifth power.)
#[derive(Copy, Clone, Default, Debug, Eq, Hash, PartialEq)]
pub struct KoalaBear4Parameters;

impl BinomialExt4Parameters for KoalaBear4Parameters {
    type Base = KoalaBearParameters;
    const W: u32 = 3;
}

/// KoalaBear^4: shorthand for [`Ext4<KoalaBear4Parameters>`].
pub type KoalaBear4 = Ext4<KoalaBear4Parameters>;
