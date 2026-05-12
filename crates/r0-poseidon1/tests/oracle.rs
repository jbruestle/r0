//! Plonky3 oracle: input `[0..15]` → known output vector for KB16
//! Poseidon1. Matches `default_koalabear_poseidon1_16` from p3-koala-bear
//! and the in-tree leanMultisig test vector. This is the load-bearing
//! correctness check for the round constants, MDS column, and round
//! structure.

use r0_field::KoalaBear;
use r0_poseidon1::host_permute;

const EXPECTED_ZERO_TO_FIFTEEN: [u32; 16] = [
    610090613, 935319874, 1893335292, 796792199, 356405232, 552237741, 55134556, 1215104204,
    1823723405, 1133298033, 1780633798, 1453946561, 710069176, 1128629550, 1917333254, 1175481618,
];

#[test]
fn host_permute_zero_to_fifteen() {
    let mut state: [KoalaBear; 16] =
        core::array::from_fn(|i| KoalaBear::from_canonical(i as u32));
    host_permute(&mut state);
    let actual: [u32; 16] = core::array::from_fn(|i| state[i].to_canonical());
    assert_eq!(
        actual, EXPECTED_ZERO_TO_FIFTEEN,
        "host_permute disagrees with the Plonky3 oracle vector"
    );
}
