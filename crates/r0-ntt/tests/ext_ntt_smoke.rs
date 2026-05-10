//! Round-trip test for `NttExec::{forward_ext, inverse_ext}`.
//!
//! Forward(N→R⁻¹) ∘ Inverse is the identity in our convention, and a
//! degree-D extension polynomial in transposed layout is bitwise identical
//! to D consecutive base-field polynomials. So feeding a known buffer
//! through `forward_ext` then `inverse_ext` should yield it back unchanged
//! — across any field that bound to the executor's base.
//!
//! The heavy NTT correctness lives in `tests/p3_oracle.rs`; this test
//! exists only to confirm the courtesy methods on `NttExec` get the
//! `batch * D` arithmetic right and the type bound `F::Base = P` allows
//! the right combinations.

use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;
use r0_field::{
    BabyBear4, BabyBear5, BabyBearParameters, BaseElem, Device, ExtField, KoalaBear4,
    KoalaBearParameters, MontyField, MontyParameters,
};
use r0_ntt::NttExec;

fn make_random_buf<P: MontyParameters>(total: usize, seed: u32) -> Vec<u32> {
    (0..total)
        .map(|i| {
            MontyField::<P>::from_canonical(
                (i as u32).wrapping_mul(0x9E37_79B1).wrapping_add(seed),
            )
            .raw()
        })
        .collect()
}

fn round_trip<P, F, R>(log_n: u32, batch: usize)
where
    P: MontyParameters,
    F: ExtField<Base = P>,
    R: Runtime,
    R::Device: Default,
{
    let n = 1usize << log_n;
    let total = batch * n * F::DEGREE as usize;

    let device = Device::<R>::acquire();
    let exec = NttExec::<P, R>::new(&device, 0);
    let client = exec.client().clone();

    let original = make_random_buf::<P>(total, 0xDEADBEEF);
    let buf = client.create_from_slice(u32::as_bytes(&original));

    exec.forward_ext::<F>(&buf, log_n, batch);
    exec.inverse_ext::<F>(&buf, log_n, batch);

    let bytes = client.read_one(buf);
    let actual = u32::from_bytes(&bytes).to_vec();
    assert_eq!(
        actual,
        original,
        "round-trip mismatch for {} batch={batch} log_n={log_n}",
        core::any::type_name::<F>()
    );
}

// ---------------------------------------------------------------------------
// Wgpu (works on Mac M-series locally; CUDA path is the same code).
// ---------------------------------------------------------------------------

#[test]
fn round_trip_base_bb_wgpu() {
    round_trip::<BabyBearParameters, BaseElem<BabyBearParameters>, WgpuRuntime>(8, 1);
}

#[test]
fn round_trip_bb4_wgpu() {
    round_trip::<BabyBearParameters, BabyBear4, WgpuRuntime>(8, 2);
}

#[test]
fn round_trip_bb5_wgpu() {
    round_trip::<BabyBearParameters, BabyBear5, WgpuRuntime>(8, 1);
}

#[test]
fn round_trip_kb4_wgpu() {
    round_trip::<KoalaBearParameters, KoalaBear4, WgpuRuntime>(8, 1);
}
