//! Cross-check: the history-of-16+r partial-round formulation
//! (`host_permute_via_history`) must agree with the naive serial
//! (`host_permute`) on every input. Validates the partial-round weight
//! and offset precomputation.

use r0_field::KoalaBear;
use r0_poseidon1::{host_permute, host_permute_via_history};

fn make_input(seed: u32) -> [KoalaBear; 16] {
    core::array::from_fn(|i| {
        // Deterministic, decorrelates well across seeds and slots.
        let v = (seed.wrapping_mul(0x9E37_79B1)).wrapping_add((i as u32).wrapping_mul(0xC2B2_AE3D));
        KoalaBear::from_canonical(v)
    })
}

fn assert_state_eq(actual: &[KoalaBear; 16], expected: &[KoalaBear; 16], label: &str) {
    let actual_canon: [u32; 16] = core::array::from_fn(|i| actual[i].to_canonical());
    let expected_canon: [u32; 16] = core::array::from_fn(|i| expected[i].to_canonical());
    assert_eq!(actual_canon, expected_canon, "{label}");
}

#[test]
fn history_matches_naive_on_zero_to_fifteen() {
    let mut a: [KoalaBear; 16] =
        core::array::from_fn(|i| KoalaBear::from_canonical(i as u32));
    let mut b = a;
    host_permute(&mut a);
    host_permute_via_history(&mut b);
    assert_state_eq(&b, &a, "history-formulation disagrees with naive on [0..15]");
}

#[test]
fn history_matches_naive_random_inputs() {
    for seed in 0u32..32 {
        let mut a = make_input(seed);
        let mut b = a;
        host_permute(&mut a);
        host_permute_via_history(&mut b);
        assert_state_eq(
            &b,
            &a,
            &format!("history-formulation disagrees with naive on seed {seed}"),
        );
    }
}
