//! Cross-check: host FFT MDS path agrees with naive matvec on every
//! input. Validates the lambda eigenvalue derivation (and therefore that
//! the W twiddle constants and INV_16 are right).

use r0_field::KoalaBear;
use r0_poseidon1::{host_mds_fft, mds_naive};

fn make_input(seed: u32) -> [KoalaBear; 16] {
    core::array::from_fn(|i| {
        let v = (seed.wrapping_mul(0x9E37_79B1)).wrapping_add((i as u32).wrapping_mul(0xC2B2_AE3D));
        KoalaBear::from_canonical(v)
    })
}

#[test]
fn fft_mds_matches_naive() {
    for seed in 0u32..32 {
        let input = make_input(seed);
        let mut a = input;
        let mut b = input;
        mds_naive(&mut a);
        host_mds_fft(&mut b);
        for i in 0..16 {
            assert_eq!(
                a[i].to_canonical(),
                b[i].to_canonical(),
                "FFT MDS disagrees with naive at slot {i}, seed {seed}"
            );
        }
    }
}

#[test]
fn fft_mds_on_unit_basis() {
    // For input e_0 = [1, 0, ..., 0], MDS · e_0 should give the first
    // column of MDS, which is MDS_CIRC_COL_CANONICAL.
    let mut state: [KoalaBear; 16] = core::array::from_fn(|i| {
        if i == 0 { KoalaBear::from_canonical(1) } else { KoalaBear::from_canonical(0) }
    });
    host_mds_fft(&mut state);
    let expected = r0_poseidon1::MDS_CIRC_COL_CANONICAL;
    let actual: [u32; 16] = core::array::from_fn(|i| state[i].to_canonical());
    assert_eq!(actual, expected, "FFT MDS on e_0 should give MDS first column");
}
