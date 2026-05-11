//! Correctness tests against Plonky3's Radix2Dit oracle.
//!
//! Spot-checks at critical log_n boundaries, batch-size sweep at
//! log_n=20, and a roundtrip to cover inverse.

use cubecl::prelude::*;

use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_field::{PrimeField32, TwoAdicField};

use r0_cube::{Device, Runtime};
use r0_field::{BabyBearParameters, MontyField, MontyParameters};
use r0_ntt::{bit_reverse_in_place, NttExec};

trait FieldBridge: MontyParameters {
    type P3: PrimeField32 + TwoAdicField + Copy;
    fn p3_from_canonical(x: u32) -> Self::P3;
}

impl FieldBridge for BabyBearParameters {
    type P3 = p3_baby_bear::BabyBear;
    fn p3_from_canonical(x: u32) -> Self::P3 { p3_baby_bear::BabyBear::new(x) }
}

fn pseudo_random_canonical<P: MontyParameters>(n: usize, batch_idx: usize) -> Vec<u32> {
    (0..n as u32)
        .map(|i| {
            let seed = i.wrapping_mul(0x9E3779B1)
                ^ (batch_idx as u32).wrapping_mul(0x517CC1B7)
                ^ 0xDEADBEEF;
            seed % P::PRIME
        })
        .collect()
}

fn check_forward<P: FieldBridge>(log_n: u32, batch: usize) {
    let n = 1usize << log_n;
    let device = Device::<Runtime>::acquire();
    let exec = NttExec::<P, Runtime>::new(&device);
    let client = device.client();

    let mut all_input = Vec::with_capacity(batch * n);
    let mut expected = Vec::with_capacity(batch * n);

    for b in 0..batch {
        let canonical = pseudo_random_canonical::<P>(n, b);
        let mut field: Vec<MontyField<P>> = canonical.iter().map(|&v| MontyField::<P>::from_canonical(v)).collect();
        bit_reverse_in_place(&mut field);
        all_input.extend(field.iter().map(|f| f.raw()));

        let p3_in: Vec<P::P3> = canonical.iter().map(|&v| P::p3_from_canonical(v)).collect();
        let p3_out = Radix2Dit::<P::P3>::default().dft(p3_in);
        expected.extend(p3_out.iter().map(|f| f.as_canonical_u32()));
    }

    let buf = client.create_from_slice(u32::as_bytes(&all_input));
    exec.forward(&buf, log_n, batch);

    let bytes = client.read_one(buf);
    let actual: Vec<u32> = u32::from_bytes(&bytes).iter()
        .map(|&raw| MontyField::<P>::from_raw(raw).to_canonical())
        .collect();

    assert_eq!(actual, expected,
        "forward mismatch: log_n={log_n}, batch={batch}, first diff at {:?}",
        actual.iter().zip(expected.iter()).position(|(a, b)| a != b));
}

fn check_roundtrip<P: MontyParameters>(log_n: u32, batch: usize) {
    let n = 1usize << log_n;
    let device = Device::<Runtime>::acquire();
    let exec = NttExec::<P, Runtime>::new(&device);
    let client = device.client();

    let mut all_input = Vec::with_capacity(batch * n);
    for b in 0..batch {
        for i in 0..n as u32 {
            let seed = i.wrapping_mul(0x9E3779B1) ^ (b as u32).wrapping_mul(0x517CC1B7) ^ 0xDEADBEEF;
            all_input.push(MontyField::<P>::from_canonical(seed % P::PRIME).raw());
        }
    }
    let original = all_input.clone();

    let buf = client.create_from_slice(u32::as_bytes(&all_input));
    exec.forward(&buf, log_n, batch);
    exec.inverse(&buf, log_n, batch);

    let result = u32::from_bytes(&client.read_one(buf)).to_vec();
    assert_eq!(result, original,
        "roundtrip failed: log_n={log_n}, batch={batch}, first diff at {:?}",
        result.iter().zip(original.iter()).position(|(a, b)| a != b));
}

#[test]
fn forward_spot() {
    for log_n in [1, 10, 14, 20, 22] {
        check_forward::<BabyBearParameters>(log_n, 1);
    }
}

#[test]
fn forward_batch() {
    for batch in [1, 3, 101] {
        check_forward::<BabyBearParameters>(20, batch);
    }
}

#[test]
fn roundtrip() {
    check_roundtrip::<BabyBearParameters>(20, 1);
}
