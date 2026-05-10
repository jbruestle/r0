//! Correctness tests against Plonky3's Radix2Dit oracle.
//!
//! Sweeps over transform sizes (log_n 1..=24), fields (BabyBear, KoalaBear),
//! and batch sizes. Tests forward NTT, inverse NTT, and roundtrip through
//! the NttExec API.

use cubecl::prelude::*;

use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_field::{PrimeField32, TwoAdicField};

use r0_field::{BabyBearParameters, Device, KoalaBearParameters, MontyField, MontyParameters};
use r0_ntt::{bit_reverse_in_place, NttExec};

// ---------------------------------------------------------------------------
// Field bridge (connects our MontyParameters to Plonky3 field types)
// ---------------------------------------------------------------------------

trait FieldBridge: MontyParameters {
    type P3: PrimeField32 + TwoAdicField + Copy;
    fn p3_from_canonical(x: u32) -> Self::P3;
}

impl FieldBridge for BabyBearParameters {
    type P3 = p3_baby_bear::BabyBear;
    fn p3_from_canonical(x: u32) -> Self::P3 {
        p3_baby_bear::BabyBear::new(x)
    }
}

impl FieldBridge for KoalaBearParameters {
    type P3 = p3_koala_bear::KoalaBear;
    fn p3_from_canonical(x: u32) -> Self::P3 {
        p3_koala_bear::KoalaBear::new(x)
    }
}

// ---------------------------------------------------------------------------
// Pseudo-random test data
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Oracle checks
// ---------------------------------------------------------------------------

/// Check forward NTT against Plonky3 for `batch` polynomials of size 2^log_n.
fn check_forward<P: FieldBridge, R: Runtime>(log_n: u32, batch: usize)
where
    R::Device: Default,
{
    let n = 1usize << log_n;
    let device = Device::<R>::acquire();
    let exec = NttExec::<P, R>::new(&device, 0);
    let client = R::client(device.inner());

    let mut all_input = Vec::with_capacity(batch * n);
    let mut expected = Vec::with_capacity(batch * n);

    for b in 0..batch {
        let canonical = pseudo_random_canonical::<P>(n, b);

        // Our input: bit-reverse the coefficients (convention: R->N).
        let mut field: Vec<MontyField<P>> = canonical
            .iter()
            .map(|&v| MontyField::<P>::from_canonical(v))
            .collect();
        bit_reverse_in_place(&mut field);
        all_input.extend(field.iter().map(|f| f.raw()));

        // Plonky3 reference.
        let p3_in: Vec<P::P3> = canonical.iter().map(|&v| P::p3_from_canonical(v)).collect();
        let dft = Radix2Dit::<P::P3>::default();
        let p3_out = dft.dft(p3_in);
        expected.extend(p3_out.iter().map(|f| f.as_canonical_u32()));
    }

    let buf = client.create_from_slice(u32::as_bytes(&all_input));
    exec.forward(&buf, log_n, batch);

    let bytes = client.read_one(buf);
    let actual: Vec<u32> = u32::from_bytes(&bytes)
        .iter()
        .map(|&raw| MontyField::<P>::from_raw(raw).to_canonical())
        .collect();

    assert_eq!(
        actual, expected,
        "forward mismatch: log_n={log_n}, batch={batch}, first diff at {:?}",
        actual.iter().zip(expected.iter()).position(|(a, b)| a != b)
    );
}

/// Check inverse NTT against Plonky3 for `batch` polynomials of size 2^log_n.
fn check_inverse<P: FieldBridge, R: Runtime>(log_n: u32, batch: usize)
where
    R::Device: Default,
{
    let n = 1usize << log_n;
    let device = Device::<R>::acquire();
    let exec = NttExec::<P, R>::new(&device, 0);
    let client = R::client(device.inner());

    let mut all_input = Vec::with_capacity(batch * n);
    let mut expected = Vec::with_capacity(batch * n);

    for b in 0..batch {
        let canonical = pseudo_random_canonical::<P>(n, b);

        // Our input: natural-order evaluations.
        let field: Vec<MontyField<P>> = canonical
            .iter()
            .map(|&v| MontyField::<P>::from_canonical(v))
            .collect();
        all_input.extend(field.iter().map(|f| f.raw()));

        // Plonky3 iDFT, then bit-reverse to match our output convention.
        let p3_in: Vec<P::P3> = canonical.iter().map(|&v| P::p3_from_canonical(v)).collect();
        let dft = Radix2Dit::<P::P3>::default();
        let p3_out = dft.idft(p3_in);
        let mut p3_canonical: Vec<u32> =
            p3_out.iter().map(|f| f.as_canonical_u32()).collect();
        bit_reverse_in_place(&mut p3_canonical);
        expected.extend(p3_canonical);
    }

    let buf = client.create_from_slice(u32::as_bytes(&all_input));
    exec.inverse(&buf, log_n, batch);

    let bytes = client.read_one(buf);
    let actual: Vec<u32> = u32::from_bytes(&bytes)
        .iter()
        .map(|&raw| MontyField::<P>::from_raw(raw).to_canonical())
        .collect();

    assert_eq!(
        actual, expected,
        "inverse mismatch: log_n={log_n}, batch={batch}, first diff at {:?}",
        actual.iter().zip(expected.iter()).position(|(a, b)| a != b)
    );
}

/// Roundtrip: forward then inverse should be identity.
fn check_roundtrip<P: MontyParameters, R: Runtime>(log_n: u32, batch: usize)
where
    R::Device: Default,
{
    let n = 1usize << log_n;
    let device = Device::<R>::acquire();
    let exec = NttExec::<P, R>::new(&device, 0);
    let client = R::client(device.inner());

    let mut all_input = Vec::with_capacity(batch * n);
    for b in 0..batch {
        for i in 0..n as u32 {
            let seed = i.wrapping_mul(0x9E3779B1)
                ^ (b as u32).wrapping_mul(0x517CC1B7)
                ^ 0xDEADBEEF;
            let val = MontyField::<P>::from_canonical(seed % P::PRIME);
            all_input.push(val.raw());
        }
    }
    let original = all_input.clone();

    let buf = client.create_from_slice(u32::as_bytes(&all_input));
    exec.forward(&buf, log_n, batch);
    exec.inverse(&buf, log_n, batch);

    let bytes = client.read_one(buf);
    let result = u32::from_bytes(&bytes).to_vec();

    assert_eq!(
        result, original,
        "roundtrip failed: log_n={log_n}, batch={batch}, first diff at {:?}",
        result.iter().zip(original.iter()).position(|(a, b)| a != b)
    );
}

// ===========================================================================
// wgpu — BabyBear log_n 1..=24, KoalaBear spot-check at 20
// ===========================================================================

#[cfg(feature = "wgpu")]
#[test]
fn wgpu_bb_forward() {
    for log_n in 1..=24u32 {
        check_forward::<BabyBearParameters, cubecl::wgpu::WgpuRuntime>(log_n, 1);
    }
}

#[cfg(feature = "wgpu")]
#[test]
fn wgpu_bb_inverse() {
    for log_n in 1..=24u32 {
        check_inverse::<BabyBearParameters, cubecl::wgpu::WgpuRuntime>(log_n, 1);
    }
}

#[cfg(feature = "wgpu")]
#[test]
fn wgpu_kb_forward() {
    check_forward::<KoalaBearParameters, cubecl::wgpu::WgpuRuntime>(20, 1);
}

#[cfg(feature = "wgpu")]
#[test]
fn wgpu_kb_inverse() {
    check_inverse::<KoalaBearParameters, cubecl::wgpu::WgpuRuntime>(20, 1);
}

// ===========================================================================
// CUDA — BabyBear, log_n 1..=24 (includes 3-pass for 21..=24)
// ===========================================================================

#[cfg(feature = "cuda")]
#[test]
fn cuda_bb_forward() {
    for log_n in 1..=24u32 {
        check_forward::<BabyBearParameters, cubecl::cuda::CudaRuntime>(log_n, 1);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_bb_inverse() {
    for log_n in 1..=24u32 {
        check_inverse::<BabyBearParameters, cubecl::cuda::CudaRuntime>(log_n, 1);
    }
}

// ===========================================================================
// CPU — BabyBear, log_n=10 (CPU compilation is very slow)
// ===========================================================================

#[cfg(feature = "cpu")]
#[test]
#[ignore]
fn cpu_bb_forward() {
    check_forward::<BabyBearParameters, cubecl::cpu::CpuRuntime>(10, 1);
}

#[cfg(feature = "cpu")]
#[test]
#[ignore]
fn cpu_bb_inverse() {
    check_inverse::<BabyBearParameters, cubecl::cpu::CpuRuntime>(10, 1);
}

// ===========================================================================
// Batch-size sweep — log_n=20
//
// With 64 MiB scratch and log_n=20 (4 MiB/poly), sub_batch=16.
// Batch sizes chosen to exercise: trivial (1), exact fit (16),
// remainder (17, 33), multiple sub-batches (32, 100).
// ===========================================================================

#[cfg(feature = "cuda")]
#[test]
fn cuda_bb_roundtrip_batch_sweep() {
    for batch in [1, 2, 3, 5, 7, 16, 17, 32, 33, 100] {
        check_roundtrip::<BabyBearParameters, cubecl::cuda::CudaRuntime>(20, batch);
    }
}

#[cfg(feature = "wgpu")]
#[test]
fn wgpu_bb_roundtrip_batch_sweep() {
    // wgpu has tighter buffer allocation limits than CUDA, so keep
    // batch sizes small enough that the user buffer fits (~128 MiB cap).
    for batch in [1, 2, 3, 5, 7, 16, 17] {
        check_roundtrip::<BabyBearParameters, cubecl::wgpu::WgpuRuntime>(20, batch);
    }
}

// Forward-only batch sweep. Roundtrip can mask sub-batch slicing bugs
// (a forward miscompute and inverse miscompute can cancel on the same row,
// while rows past sub_batch may simply be untouched). This sweep checks
// each batch row against Plonky3 directly, so it only passes if every
// sub-batch iteration writes the right slice.

#[cfg(feature = "cuda")]
#[test]
fn cuda_bb_forward_batch_sweep() {
    for batch in [1, 2, 3, 5, 7, 16, 17, 32, 33, 100] {
        check_forward::<BabyBearParameters, cubecl::cuda::CudaRuntime>(20, batch);
    }
}

#[cfg(feature = "wgpu")]
#[test]
fn wgpu_bb_forward_batch_sweep() {
    for batch in [1, 2, 3, 5, 7, 16, 17] {
        check_forward::<BabyBearParameters, cubecl::wgpu::WgpuRuntime>(20, batch);
    }
}
