//! Round-trip test for `NttExec::{forward_ext, inverse_ext}`.
//!
//! Confirms the `batch * D` arithmetic in the convenience methods is
//! wired correctly.

use cubecl::prelude::*;
use r0_cube::{Device, Runtime};
use r0_field::{ExtField, KoalaBear4, KoalaBearParameters, MontyField};
use r0_ntt::NttExec;

#[test]
fn round_trip_kb4() {
    let log_n = 8u32;
    let batch = 1usize;
    let n = 1usize << log_n;
    let total = batch * n * KoalaBear4::DEGREE as usize;

    let device = Device::<Runtime>::acquire();
    let exec = NttExec::<KoalaBearParameters, Runtime>::new(&device);
    let client = exec.client().clone();

    let original: Vec<u32> = (0..total)
        .map(|i| MontyField::<KoalaBearParameters>::from_canonical(
            (i as u32).wrapping_mul(0x9E37_79B1).wrapping_add(0xDEADBEEF),
        ).raw())
        .collect();
    let buf = client.create_from_slice(u32::as_bytes(&original));

    exec.forward_ext::<KoalaBear4>(&buf, log_n, batch);
    exec.inverse_ext::<KoalaBear4>(&buf, log_n, batch);

    let actual = u32::from_bytes(&client.read_one(buf)).to_vec();
    assert_eq!(actual, original, "round-trip mismatch for KB^4");
}
