//! Plonky3 oracle for single-pass NTT (log_n <= 10).
//!
//! Uses the unified `ntt_fwd_pass` / `ntt_inv_pass` kernels with a single
//! pass (stage_offset=0, log_pass=log_n) which is the final pass
//! (no transpose, in-place safe).

use cubecl::cpu::CpuRuntime;
use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;

use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_field::{PrimeField32, TwoAdicField};

use r0_field::{BabyBearParameters, KoalaBearParameters, MontyField, MontyParameters};
use r0_ntt::{bit_reverse_in_place, build_inv_twiddles, build_fwd_twiddles, ntt_inv_pass, n_inv, ntt_fwd_pass};

fn pick_log_wg(log_n: u32) -> u32 {
    log_n.saturating_sub(1).min(8)
}

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

fn run_ntt<P: MontyParameters, R: Runtime>(
    device: &R::Device,
    bitrev_coeffs_raw: &[u32],
    twiddles_raw: &[u32],
    log_n: u32,
) -> Vec<u32> {
    let n = 1usize << log_n;
    let log_wg = pick_log_wg(log_n);
    let client = R::client(device);
    let data_h = client.create_from_slice(u32::as_bytes(bitrev_coeffs_raw));
    let tw_h = client.create_from_slice(u32::as_bytes(twiddles_raw));

    // Single pass: stage_offset=0, log_pass=log_n -> final pass, in-place.
    unsafe {
        ntt_fwd_pass::launch_unchecked::<P, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1u32 << log_wg),
            ArrayArg::from_raw_parts::<u32>(&data_h, n, 1),
            ArrayArg::from_raw_parts::<u32>(&data_h, n, 1),
            ArrayArg::from_raw_parts::<u32>(&tw_h, n / 2, 1),
            log_n,
            log_n,
            0u32,
            log_wg,
            1u32,
        )
        .expect("ntt_fwd_pass launch failed");
    }

    let bytes = client.read_one(data_h);
    u32::from_bytes(&bytes).to_vec()
}

fn run_intt<P: MontyParameters, R: Runtime>(
    device: &R::Device,
    natural_evals_raw: &[u32],
    inv_twiddles_raw: &[u32],
    inv_n_raw: u32,
    log_n: u32,
) -> Vec<u32> {
    let n = 1usize << log_n;
    let log_wg = pick_log_wg(log_n);
    let client = R::client(device);
    let data_h = client.create_from_slice(u32::as_bytes(natural_evals_raw));
    let tw_h = client.create_from_slice(u32::as_bytes(inv_twiddles_raw));
    let inv_n_h = client.create_from_slice(u32::as_bytes(&[inv_n_raw]));

    // Single pass: stage_offset=0, log_pass=log_n -> final pass, in-place.
    unsafe {
        ntt_inv_pass::launch_unchecked::<P, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1u32 << log_wg),
            ArrayArg::from_raw_parts::<u32>(&data_h, n, 1),
            ArrayArg::from_raw_parts::<u32>(&data_h, n, 1),
            ArrayArg::from_raw_parts::<u32>(&tw_h, n / 2, 1),
            ArrayArg::from_raw_parts::<u32>(&inv_n_h, 1, 1),
            log_n,
            log_n,
            0u32,
            log_wg,
            1u32,
        )
        .expect("ntt_inv_pass launch failed");
    }

    let bytes = client.read_one(data_h);
    u32::from_bytes(&bytes).to_vec()
}

fn check_forward<P: FieldBridge, R: Runtime>(log_n: u32)
where
    R::Device: Default,
{
    let n = 1usize << log_n;
    let canonical: Vec<u32> = (0..n as u32)
        .map(|i| (i.wrapping_mul(0x9E3779B1) ^ 0xDEADBEEF) % P::PRIME)
        .collect();

    let p3_in: Vec<P::P3> = canonical.iter().map(|&v| P::p3_from_canonical(v)).collect();
    let dft = Radix2Dit::<P::P3>::default();
    let p3_out = dft.dft(p3_in);
    let expected: Vec<u32> = p3_out.iter().map(|f| f.as_canonical_u32()).collect();

    let mut our_in_field: Vec<MontyField<P>> = canonical
        .iter()
        .map(|&v| MontyField::<P>::from_canonical(v))
        .collect();
    bit_reverse_in_place(&mut our_in_field);
    let our_in_raw: Vec<u32> = our_in_field.iter().map(|f| f.raw()).collect();
    let twiddles = build_fwd_twiddles::<P>(log_n);

    let actual_raw = run_ntt::<P, R>(&Default::default(), &our_in_raw, &twiddles, log_n);
    let actual: Vec<u32> = actual_raw
        .iter()
        .map(|&raw| MontyField::<P>::from_raw(raw).to_canonical())
        .collect();

    assert_eq!(
        actual, expected,
        "forward mismatch at log_n={log_n}: first diff at {:?}",
        actual.iter().zip(expected.iter()).position(|(a, b)| a != b)
    );
}

fn check_inverse<P: FieldBridge, R: Runtime>(log_n: u32)
where
    R::Device: Default,
{
    let n = 1usize << log_n;
    let natural_evals: Vec<u32> = (0..n as u32)
        .map(|i| (i.wrapping_mul(0xC2B2AE3D) ^ 0x85EBCA77) % P::PRIME)
        .collect();

    let p3_in: Vec<P::P3> = natural_evals.iter().map(|&v| P::p3_from_canonical(v)).collect();
    let dft = Radix2Dit::<P::P3>::default();
    let p3_out = dft.idft(p3_in);
    let expected_natural: Vec<u32> = p3_out.iter().map(|f| f.as_canonical_u32()).collect();
    let mut expected_bitrev = expected_natural.clone();
    bit_reverse_in_place(&mut expected_bitrev);

    let natural_raw: Vec<u32> = natural_evals
        .iter()
        .map(|&v| MontyField::<P>::from_canonical(v).raw())
        .collect();
    let inv_twiddles = build_inv_twiddles::<P>(log_n);
    let inv_n = n_inv::<P>(log_n);

    let actual_raw = run_intt::<P, R>(&Default::default(), &natural_raw, &inv_twiddles, inv_n, log_n);
    let actual: Vec<u32> = actual_raw
        .iter()
        .map(|&raw| MontyField::<P>::from_raw(raw).to_canonical())
        .collect();

    assert_eq!(
        actual, expected_bitrev,
        "inverse mismatch at log_n={log_n}: first diff at {:?}",
        actual.iter().zip(expected_bitrev.iter()).position(|(a, b)| a != b)
    );
}

fn check_roundtrip<P: FieldBridge, R: Runtime>(log_n: u32)
where
    R::Device: Default,
{
    let n = 1usize << log_n;
    let bitrev_canonical: Vec<u32> = (0..n as u32)
        .map(|i| (i.wrapping_mul(0x9E3779B1) ^ 0xDEADBEEF) % P::PRIME)
        .collect();
    let bitrev_raw: Vec<u32> = bitrev_canonical
        .iter()
        .map(|&v| MontyField::<P>::from_canonical(v).raw())
        .collect();

    let twiddles = build_fwd_twiddles::<P>(log_n);
    let inv_twiddles = build_inv_twiddles::<P>(log_n);
    let inv_n = n_inv::<P>(log_n);

    let natural_evals_raw = run_ntt::<P, R>(&Default::default(), &bitrev_raw, &twiddles, log_n);
    let recovered_raw = run_intt::<P, R>(&Default::default(), &natural_evals_raw, &inv_twiddles, inv_n, log_n);

    let recovered: Vec<u32> = recovered_raw
        .iter()
        .map(|&raw| MontyField::<P>::from_raw(raw).to_canonical())
        .collect();

    assert_eq!(
        recovered, bitrev_canonical,
        "round-trip failed at log_n={log_n}: first diff at {:?}",
        recovered.iter().zip(bitrev_canonical.iter()).position(|(a, b)| a != b)
    );
}

fn check_all<P: FieldBridge, R: Runtime>(range: impl IntoIterator<Item = u32>)
where
    R::Device: Default,
{
    for log_n in range {
        check_forward::<P, R>(log_n);
        check_inverse::<P, R>(log_n);
        check_roundtrip::<P, R>(log_n);
    }
}

// wgpu: full single-pass range.
#[test]
fn bb_wgpu() { check_all::<BabyBearParameters,  WgpuRuntime>(1..=10); }
#[test]
fn kb_wgpu() { check_all::<KoalaBearParameters, WgpuRuntime>(1..=10); }

// cubecl-cpu: spot check.
#[test]
fn bb_cpu()  { check_all::<BabyBearParameters,  CpuRuntime>([8u32]); }
#[test]
fn kb_cpu()  { check_all::<KoalaBearParameters, CpuRuntime>([8u32]); }
