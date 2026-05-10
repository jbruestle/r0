//! Cross-check our `Ext4<…>` / `Ext5<…>` host arithmetic against Plonky3's
//! `BinomialExtensionField<…, D>` for the same `(base, W)` pair.
//!
//! Plonky3's binomial extension has an established correctness story
//! (Frobenius tests, two-adic-extension tests, etc.), so we treat it as
//! the oracle and bit-for-bit compare the canonical limbs of every op.

use p3_baby_bear::BabyBear as P3BabyBear;
use p3_field::extension::BinomialExtensionField as P3Ext;
use p3_field::{BasedVectorSpace, PrimeField32};
use p3_koala_bear::KoalaBear as P3KoalaBear;
use proptest::prelude::*;

use r0_field::{
    BabyBear4, BabyBear4Parameters, BabyBear5, BabyBear5Parameters, BabyBearParameters,
    BinomialExt4Parameters, BinomialExt5Parameters, KoalaBear4, KoalaBear4Parameters,
    KoalaBearParameters, MontyField, MontyParameters,
};

// ===========================================================================
// Glue: convert canonical-form limb arrays between our and Plonky3's types.
// ===========================================================================

fn p3_bb4_from(limbs: [u32; 4]) -> P3Ext<P3BabyBear, 4> {
    P3Ext::<P3BabyBear, 4>::new(limbs.map(P3BabyBear::new))
}

fn p3_bb4_canonical(x: P3Ext<P3BabyBear, 4>) -> [u32; 4] {
    let s: &[P3BabyBear] = x.as_basis_coefficients_slice();
    [
        s[0].as_canonical_u32(),
        s[1].as_canonical_u32(),
        s[2].as_canonical_u32(),
        s[3].as_canonical_u32(),
    ]
}

fn p3_kb4_from(limbs: [u32; 4]) -> P3Ext<P3KoalaBear, 4> {
    P3Ext::<P3KoalaBear, 4>::new(limbs.map(P3KoalaBear::new))
}

fn p3_kb4_canonical(x: P3Ext<P3KoalaBear, 4>) -> [u32; 4] {
    let s: &[P3KoalaBear] = x.as_basis_coefficients_slice();
    [
        s[0].as_canonical_u32(),
        s[1].as_canonical_u32(),
        s[2].as_canonical_u32(),
        s[3].as_canonical_u32(),
    ]
}

fn p3_bb5_from(limbs: [u32; 5]) -> P3Ext<P3BabyBear, 5> {
    P3Ext::<P3BabyBear, 5>::new(limbs.map(P3BabyBear::new))
}

fn p3_bb5_canonical(x: P3Ext<P3BabyBear, 5>) -> [u32; 5] {
    let s: &[P3BabyBear] = x.as_basis_coefficients_slice();
    [
        s[0].as_canonical_u32(),
        s[1].as_canonical_u32(),
        s[2].as_canonical_u32(),
        s[3].as_canonical_u32(),
        s[4].as_canonical_u32(),
    ]
}

// ===========================================================================
// Strategy helpers — random canonical limbs in [0, p).
// ===========================================================================

fn limb_strategy<P: MontyParameters>() -> impl Strategy<Value = u32> {
    0u32..P::PRIME
}

fn limbs4<P: MontyParameters>() -> impl Strategy<Value = [u32; 4]> {
    [
        limb_strategy::<P>(),
        limb_strategy::<P>(),
        limb_strategy::<P>(),
        limb_strategy::<P>(),
    ]
}

fn limbs5<P: MontyParameters>() -> impl Strategy<Value = [u32; 5]> {
    [
        limb_strategy::<P>(),
        limb_strategy::<P>(),
        limb_strategy::<P>(),
        limb_strategy::<P>(),
        limb_strategy::<P>(),
    ]
}

// ===========================================================================
// Sanity asserts (non-proptest) — run on every test invocation.
// ===========================================================================

#[test]
fn w_mont_matches_independent_computation_bb4() {
    // Independent Montgomery-form lifting via MontyField, then bit-equality.
    let expected = MontyField::<BabyBearParameters>::from_canonical(BabyBear4Parameters::W).raw();
    assert_eq!(BabyBear4Parameters::W_MONT, expected);
}

#[test]
fn w_mont_matches_independent_computation_kb4() {
    let expected = MontyField::<KoalaBearParameters>::from_canonical(KoalaBear4Parameters::W).raw();
    assert_eq!(KoalaBear4Parameters::W_MONT, expected);
}

#[test]
fn w_mont_matches_independent_computation_bb5() {
    let expected = MontyField::<BabyBearParameters>::from_canonical(BabyBear5Parameters::W).raw();
    assert_eq!(BabyBear5Parameters::W_MONT, expected);
}

#[test]
fn one_round_trips_via_canonical_bb4() {
    let one = BabyBear4::from_canonical([1, 0, 0, 0]);
    assert_eq!(one.to_canonical(), [1, 0, 0, 0]);
}

#[test]
fn one_round_trips_via_canonical_kb4() {
    let one = KoalaBear4::from_canonical([1, 0, 0, 0]);
    assert_eq!(one.to_canonical(), [1, 0, 0, 0]);
}

#[test]
fn one_round_trips_via_canonical_bb5() {
    let one = BabyBear5::from_canonical([1, 0, 0, 0, 0]);
    assert_eq!(one.to_canonical(), [1, 0, 0, 0, 0]);
}

// ===========================================================================
// proptest oracles vs Plonky3 — covers add, sub, mul, neg, distributivity.
// ===========================================================================

proptest! {
    #![proptest_config(ProptestConfig { cases: 256, .. ProptestConfig::default() })]

    // --- BabyBear^4 (W=11) ---------------------------------------------------

    #[test]
    fn bb4_round_trip(limbs in limbs4::<BabyBearParameters>()) {
        prop_assert_eq!(BabyBear4::from_canonical(limbs).to_canonical(), limbs);
    }

    #[test]
    fn bb4_add_matches_p3(a in limbs4::<BabyBearParameters>(), b in limbs4::<BabyBearParameters>()) {
        let ours = (BabyBear4::from_canonical(a) + BabyBear4::from_canonical(b)).to_canonical();
        let p3 = p3_bb4_canonical(p3_bb4_from(a) + p3_bb4_from(b));
        prop_assert_eq!(ours, p3);
    }

    #[test]
    fn bb4_sub_matches_p3(a in limbs4::<BabyBearParameters>(), b in limbs4::<BabyBearParameters>()) {
        let ours = (BabyBear4::from_canonical(a) - BabyBear4::from_canonical(b)).to_canonical();
        let p3 = p3_bb4_canonical(p3_bb4_from(a) - p3_bb4_from(b));
        prop_assert_eq!(ours, p3);
    }

    #[test]
    fn bb4_mul_matches_p3(a in limbs4::<BabyBearParameters>(), b in limbs4::<BabyBearParameters>()) {
        let ours = (BabyBear4::from_canonical(a) * BabyBear4::from_canonical(b)).to_canonical();
        let p3 = p3_bb4_canonical(p3_bb4_from(a) * p3_bb4_from(b));
        prop_assert_eq!(ours, p3);
    }

    #[test]
    fn bb4_neg_matches_p3(a in limbs4::<BabyBearParameters>()) {
        let ours = (-BabyBear4::from_canonical(a)).to_canonical();
        let p3 = p3_bb4_canonical(-p3_bb4_from(a));
        prop_assert_eq!(ours, p3);
    }

    #[test]
    fn bb4_a_plus_neg_a_is_zero(a in limbs4::<BabyBearParameters>()) {
        let f = BabyBear4::from_canonical(a);
        prop_assert_eq!((f + (-f)).to_canonical(), [0, 0, 0, 0]);
    }

    #[test]
    fn bb4_a_times_zero_is_zero(a in limbs4::<BabyBearParameters>()) {
        let f = BabyBear4::from_canonical(a);
        let zero = BabyBear4::from_canonical([0, 0, 0, 0]);
        prop_assert_eq!((f * zero).to_canonical(), [0, 0, 0, 0]);
    }

    #[test]
    fn bb4_a_times_one_is_a(a in limbs4::<BabyBearParameters>()) {
        let f = BabyBear4::from_canonical(a);
        let one = BabyBear4::from_canonical([1, 0, 0, 0]);
        prop_assert_eq!((f * one).to_canonical(), a);
    }

    #[test]
    fn bb4_distributivity(
        a in limbs4::<BabyBearParameters>(),
        b in limbs4::<BabyBearParameters>(),
        c in limbs4::<BabyBearParameters>(),
    ) {
        let af = BabyBear4::from_canonical(a);
        let bf = BabyBear4::from_canonical(b);
        let cf = BabyBear4::from_canonical(c);
        let lhs = ((af + bf) * cf).to_canonical();
        let rhs = (af * cf + bf * cf).to_canonical();
        prop_assert_eq!(lhs, rhs);
    }

    // --- KoalaBear^4 (W=3) ---------------------------------------------------

    #[test]
    fn kb4_round_trip(limbs in limbs4::<KoalaBearParameters>()) {
        prop_assert_eq!(KoalaBear4::from_canonical(limbs).to_canonical(), limbs);
    }

    #[test]
    fn kb4_add_matches_p3(a in limbs4::<KoalaBearParameters>(), b in limbs4::<KoalaBearParameters>()) {
        let ours = (KoalaBear4::from_canonical(a) + KoalaBear4::from_canonical(b)).to_canonical();
        let p3 = p3_kb4_canonical(p3_kb4_from(a) + p3_kb4_from(b));
        prop_assert_eq!(ours, p3);
    }

    #[test]
    fn kb4_mul_matches_p3(a in limbs4::<KoalaBearParameters>(), b in limbs4::<KoalaBearParameters>()) {
        let ours = (KoalaBear4::from_canonical(a) * KoalaBear4::from_canonical(b)).to_canonical();
        let p3 = p3_kb4_canonical(p3_kb4_from(a) * p3_kb4_from(b));
        prop_assert_eq!(ours, p3);
    }

    #[test]
    fn kb4_neg_matches_p3(a in limbs4::<KoalaBearParameters>()) {
        let ours = (-KoalaBear4::from_canonical(a)).to_canonical();
        let p3 = p3_kb4_canonical(-p3_kb4_from(a));
        prop_assert_eq!(ours, p3);
    }

    #[test]
    fn kb4_distributivity(
        a in limbs4::<KoalaBearParameters>(),
        b in limbs4::<KoalaBearParameters>(),
        c in limbs4::<KoalaBearParameters>(),
    ) {
        let af = KoalaBear4::from_canonical(a);
        let bf = KoalaBear4::from_canonical(b);
        let cf = KoalaBear4::from_canonical(c);
        prop_assert_eq!(((af + bf) * cf).to_canonical(), (af * cf + bf * cf).to_canonical());
    }

    // --- BabyBear^5 (W=2) ----------------------------------------------------

    #[test]
    fn bb5_round_trip(limbs in limbs5::<BabyBearParameters>()) {
        prop_assert_eq!(BabyBear5::from_canonical(limbs).to_canonical(), limbs);
    }

    #[test]
    fn bb5_add_matches_p3(a in limbs5::<BabyBearParameters>(), b in limbs5::<BabyBearParameters>()) {
        let ours = (BabyBear5::from_canonical(a) + BabyBear5::from_canonical(b)).to_canonical();
        let p3 = p3_bb5_canonical(p3_bb5_from(a) + p3_bb5_from(b));
        prop_assert_eq!(ours, p3);
    }

    #[test]
    fn bb5_mul_matches_p3(a in limbs5::<BabyBearParameters>(), b in limbs5::<BabyBearParameters>()) {
        let ours = (BabyBear5::from_canonical(a) * BabyBear5::from_canonical(b)).to_canonical();
        let p3 = p3_bb5_canonical(p3_bb5_from(a) * p3_bb5_from(b));
        prop_assert_eq!(ours, p3);
    }

    #[test]
    fn bb5_neg_matches_p3(a in limbs5::<BabyBearParameters>()) {
        let ours = (-BabyBear5::from_canonical(a)).to_canonical();
        let p3 = p3_bb5_canonical(-p3_bb5_from(a));
        prop_assert_eq!(ours, p3);
    }

    #[test]
    fn bb5_distributivity(
        a in limbs5::<BabyBearParameters>(),
        b in limbs5::<BabyBearParameters>(),
        c in limbs5::<BabyBearParameters>(),
    ) {
        let af = BabyBear5::from_canonical(a);
        let bf = BabyBear5::from_canonical(b);
        let cf = BabyBear5::from_canonical(c);
        prop_assert_eq!(((af + bf) * cf).to_canonical(), (af * cf + bf * cf).to_canonical());
    }
}
