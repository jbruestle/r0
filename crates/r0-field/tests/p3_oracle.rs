//! Plonky3 oracle: cross-checks every Montgomery field op against the
//! published `p3-baby-bear` / `p3-koala-bear` impls. If our `MontyField`
//! ever disagrees with Plonky3 on `add`/`sub`/`mul`/`neg` for a random
//! input, this test fails.

use p3_field::PrimeField32;
use proptest::prelude::*;

use r0_field::{BabyBear, KoalaBear, MontyField, MontyParameters};

// Generate a `u32` strictly less than P::PRIME — what `MontyField`
// claims its inputs are bounded by.
fn arb_field<P: MontyParameters>() -> impl Strategy<Value = u32> {
    0..P::PRIME
}

/// Round-trip: `MontyField::from_canonical(x).to_canonical() == x % p`.
fn check_roundtrip<P: MontyParameters>(x: u32) {
    let f = MontyField::<P>::from_canonical(x);
    assert_eq!(f.to_canonical(), x % P::PRIME);
}

/// Compare a binary op against Plonky3 by canonical-form values.
macro_rules! check_binop_oracle {
    ($a:expr, $b:expr, $p3_ty:ty, $our_ty:ty, $op:tt) => {{
        let a = $a;
        let b = $b;
        let our = (<$our_ty>::from_canonical(a) $op <$our_ty>::from_canonical(b)).to_canonical();
        let p3 = (<$p3_ty>::new(a) $op <$p3_ty>::new(b)).as_canonical_u32();
        assert_eq!(our, p3, "op `{}` disagreed: a={a:#x} b={b:#x} our={our:#x} p3={p3:#x}", stringify!($op));
    }};
}

macro_rules! check_neg_oracle {
    ($a:expr, $p3_ty:ty, $our_ty:ty) => {{
        let a = $a;
        let our = (-<$our_ty>::from_canonical(a)).to_canonical();
        let p3 = (-<$p3_ty>::new(a)).as_canonical_u32();
        assert_eq!(our, p3, "neg disagreed: a={a:#x} our={our:#x} p3={p3:#x}");
    }};
}

// ---- BabyBear ----

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn bb_roundtrip(x in any::<u32>()) {
        check_roundtrip::<r0_field::BabyBearParameters>(x);
    }

    #[test]
    fn bb_add(a in arb_field::<r0_field::BabyBearParameters>(),
              b in arb_field::<r0_field::BabyBearParameters>()) {
        check_binop_oracle!(a, b, p3_baby_bear::BabyBear, BabyBear, +);
    }

    #[test]
    fn bb_sub(a in arb_field::<r0_field::BabyBearParameters>(),
              b in arb_field::<r0_field::BabyBearParameters>()) {
        check_binop_oracle!(a, b, p3_baby_bear::BabyBear, BabyBear, -);
    }

    #[test]
    fn bb_mul(a in arb_field::<r0_field::BabyBearParameters>(),
              b in arb_field::<r0_field::BabyBearParameters>()) {
        check_binop_oracle!(a, b, p3_baby_bear::BabyBear, BabyBear, *);
    }

    #[test]
    fn bb_neg(a in arb_field::<r0_field::BabyBearParameters>()) {
        check_neg_oracle!(a, p3_baby_bear::BabyBear, BabyBear);
    }
}

// ---- KoalaBear ----

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn kb_roundtrip(x in any::<u32>()) {
        check_roundtrip::<r0_field::KoalaBearParameters>(x);
    }

    #[test]
    fn kb_add(a in arb_field::<r0_field::KoalaBearParameters>(),
              b in arb_field::<r0_field::KoalaBearParameters>()) {
        check_binop_oracle!(a, b, p3_koala_bear::KoalaBear, KoalaBear, +);
    }

    #[test]
    fn kb_sub(a in arb_field::<r0_field::KoalaBearParameters>(),
              b in arb_field::<r0_field::KoalaBearParameters>()) {
        check_binop_oracle!(a, b, p3_koala_bear::KoalaBear, KoalaBear, -);
    }

    #[test]
    fn kb_mul(a in arb_field::<r0_field::KoalaBearParameters>(),
              b in arb_field::<r0_field::KoalaBearParameters>()) {
        check_binop_oracle!(a, b, p3_koala_bear::KoalaBear, KoalaBear, *);
    }

    #[test]
    fn kb_neg(a in arb_field::<r0_field::KoalaBearParameters>()) {
        check_neg_oracle!(a, p3_koala_bear::KoalaBear, KoalaBear);
    }
}

// ---- Quick fixed-input sanity checks (don't need proptest to fail) ----

#[test]
fn bb_zero_one_identity() {
    let zero = BabyBear::from_canonical(0);
    let one = BabyBear::from_canonical(1);
    assert_eq!(zero.to_canonical(), 0);
    assert_eq!(one.to_canonical(), 1);
    assert_eq!((one + one).to_canonical(), 2);
    assert_eq!((zero - one).to_canonical(), 0x78000000);
    assert_eq!((-one).to_canonical(), 0x78000000);
    let big = BabyBear::from_canonical(0x78000000);
    assert_eq!((big + one).to_canonical(), 0);
}

#[test]
fn kb_zero_one_identity() {
    let zero = KoalaBear::from_canonical(0);
    let one = KoalaBear::from_canonical(1);
    assert_eq!(zero.to_canonical(), 0);
    assert_eq!(one.to_canonical(), 1);
    assert_eq!((one + one).to_canonical(), 2);
    assert_eq!((zero - one).to_canonical(), 0x7f000000);
    assert_eq!((-one).to_canonical(), 0x7f000000);
    let big = KoalaBear::from_canonical(0x7f000000);
    assert_eq!((big + one).to_canonical(), 0);
}
