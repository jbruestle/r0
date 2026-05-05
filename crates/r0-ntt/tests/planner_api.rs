//! Tests for the NttPlanner high-level API.

use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;

use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_field::PrimeField32;

use r0_field::{BabyBearParameters, MontyField, MontyParameters};
use r0_ntt::{bit_reverse_in_place, NttPlanner};

fn check_forward_planner<R: Runtime>(log_n: u32, batch: usize)
where
    R::Device: Default,
{
    type P = BabyBearParameters;
    let n = 1usize << log_n;

    // Build input: batch polynomials with pseudo-random coefficients.
    let mut all_input = Vec::with_capacity(batch * n);
    for b in 0..batch {
        let canonical: Vec<u32> = (0..n as u32)
            .map(|i| {
                let seed = i.wrapping_mul(0x9E3779B1) ^ (b as u32).wrapping_mul(0x517CC1B7) ^ 0xDEADBEEF;
                seed % P::PRIME
            })
            .collect();

        // Bit-reverse the coefficients (our convention: R→N forward).
        let mut field: Vec<MontyField<P>> = canonical
            .iter()
            .map(|&v| MontyField::<P>::from_canonical(v))
            .collect();
        bit_reverse_in_place(&mut field);
        all_input.extend(field.iter().map(|f| f.raw()));
    }

    // Compute expected via Plonky3 for each polynomial.
    let mut expected = Vec::with_capacity(batch * n);
    for b in 0..batch {
        let canonical: Vec<u32> = (0..n as u32)
            .map(|i| {
                let seed = i.wrapping_mul(0x9E3779B1) ^ (b as u32).wrapping_mul(0x517CC1B7) ^ 0xDEADBEEF;
                seed % P::PRIME
            })
            .collect();
        let p3_in: Vec<p3_baby_bear::BabyBear> = canonical
            .iter()
            .map(|&v| p3_baby_bear::BabyBear::new(v))
            .collect();
        let dft = Radix2Dit::<p3_baby_bear::BabyBear>::default();
        let p3_out = dft.dft(p3_in);
        expected.extend(p3_out.iter().map(|f| f.as_canonical_u32()));
    }

    // Run through planner.
    let device = R::Device::default();
    let planner = NttPlanner::<P, R>::new(&device, 0);
    let client = R::client(&device);
    let buf = client.create_from_slice(u32::as_bytes(&all_input));

    planner.forward(&buf, log_n, batch);

    // Read back and compare.
    let bytes = client.read_one(buf);
    let actual_raw = u32::from_bytes(&bytes);
    let actual: Vec<u32> = actual_raw
        .iter()
        .map(|&raw| MontyField::<P>::from_raw(raw).to_canonical())
        .collect();

    assert_eq!(
        actual, expected,
        "planner forward mismatch at log_n={log_n}, batch={batch}: first diff at {:?}",
        actual.iter().zip(expected.iter()).position(|(a, b)| a != b)
    );
}

fn check_roundtrip_planner<R: Runtime>(log_n: u32, batch: usize)
where
    R::Device: Default,
{
    type P = BabyBearParameters;
    let n = 1usize << log_n;

    // Build input in bit-reversed order.
    let mut all_input = Vec::with_capacity(batch * n);
    for b in 0..batch {
        for i in 0..n as u32 {
            let seed = i.wrapping_mul(0x9E3779B1) ^ (b as u32).wrapping_mul(0x517CC1B7) ^ 0xDEADBEEF;
            let val = MontyField::<P>::from_canonical(seed % P::PRIME);
            all_input.push(val.raw());
        }
    }
    let original = all_input.clone();

    let device = R::Device::default();
    let planner = NttPlanner::<P, R>::new(&device, 0);
    let client = R::client(&device);
    let buf = client.create_from_slice(u32::as_bytes(&all_input));

    // Forward then inverse should be identity.
    planner.forward(&buf, log_n, batch);
    planner.inverse(&buf, log_n, batch);

    let bytes = client.read_one(buf);
    let result = u32::from_bytes(&bytes).to_vec();

    assert_eq!(
        result, original,
        "roundtrip failed at log_n={log_n}, batch={batch}: first diff at {:?}",
        result.iter().zip(original.iter()).position(|(a, b)| a != b)
    );
}

// -- Forward tests --

#[test]
fn forward_single_pass_wgpu() {
    check_forward_planner::<WgpuRuntime>(8, 1);
    check_forward_planner::<WgpuRuntime>(10, 3);
}

#[test]
fn forward_two_pass_wgpu() {
    check_forward_planner::<WgpuRuntime>(14, 1);
    check_forward_planner::<WgpuRuntime>(14, 4);
}

#[test]
fn forward_cuda() {
    check_forward_planner::<cubecl::cuda::CudaRuntime>(20, 2);
}

// -- Roundtrip tests --

#[test]
fn roundtrip_single_pass_wgpu() {
    check_roundtrip_planner::<WgpuRuntime>(8, 1);
    check_roundtrip_planner::<WgpuRuntime>(10, 5);
}

#[test]
fn roundtrip_two_pass_wgpu() {
    check_roundtrip_planner::<WgpuRuntime>(14, 1);
    check_roundtrip_planner::<WgpuRuntime>(14, 4);
}

#[test]
fn roundtrip_cuda() {
    check_roundtrip_planner::<cubecl::cuda::CudaRuntime>(20, 2);
}
