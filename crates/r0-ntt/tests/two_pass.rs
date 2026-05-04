//! Plonky3 oracle tests for forward NTT (unified `ntt_pass`) and
//! inverse NTT (`intt_pass1` + `intt_pass2`).

use cubecl::cpu::CpuRuntime;
use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;

use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_field::{PrimeField32, TwoAdicField};

use r0_field::{BabyBearParameters, KoalaBearParameters, MontyField, MontyParameters};
use r0_ntt::{
    bit_reverse_in_place, build_inv_twiddles, build_twiddles, intt_pass1, intt_pass2, n_inv,
    ntt_pass,
};

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

fn pick_log_n1(log_n: u32) -> u32 {
    log_n / 2
}

fn pick_log_wg(log_pass_size: u32) -> u32 {
    log_pass_size.saturating_sub(1).min(8)
}

// ---- Forward runner (unified ntt_pass, separate in/out buffers) ----

fn run_forward<P: MontyParameters, R: Runtime>(
    device: &R::Device,
    bitrev_coeffs_raw: &[u32],
    twiddles_raw: &[u32],
    log_n: u32,
) -> Vec<u32> {
    let n = 1usize << log_n;
    let log_n1 = pick_log_n1(log_n);
    let log_n2 = log_n - log_n1;
    let n1: usize = 1usize << log_n1;
    let n2: usize = 1usize << log_n2;

    assert_eq!(bitrev_coeffs_raw.len(), n);
    assert_eq!(twiddles_raw.len(), n / 2);

    let client = R::client(device);
    let buf_a = client.create_from_slice(u32::as_bytes(bitrev_coeffs_raw));
    let buf_b = client.create_from_slice(u32::as_bytes(&vec![0u32; n]));
    let tw_h = client.create_from_slice(u32::as_bytes(twiddles_raw));

    let log_wg1 = pick_log_wg(log_n1);
    let log_wg2 = pick_log_wg(log_n2);

    unsafe {
        // Pass 1: buf_a → buf_b (transposed, since non-final)
        ntt_pass::launch_unchecked::<P, R>(
            &client,
            CubeCount::Static(n2 as u32, 1, 1),
            CubeDim::new_1d(1u32 << log_wg1),
            ArrayArg::from_raw_parts::<u32>(&buf_a, n, 1),
            ArrayArg::from_raw_parts::<u32>(&buf_b, n, 1),
            ArrayArg::from_raw_parts::<u32>(&tw_h, n / 2, 1),
            log_n,
            log_n1,
            0u32,
            log_wg1,
            1u32,
        )
        .expect("ntt_pass (first) failed");

        // Pass 2: buf_b → buf_b (final, in-place safe)
        ntt_pass::launch_unchecked::<P, R>(
            &client,
            CubeCount::Static(n1 as u32, 1, 1),
            CubeDim::new_1d(1u32 << log_wg2),
            ArrayArg::from_raw_parts::<u32>(&buf_b, n, 1),
            ArrayArg::from_raw_parts::<u32>(&buf_b, n, 1),
            ArrayArg::from_raw_parts::<u32>(&tw_h, n / 2, 1),
            log_n,
            log_n2,
            log_n1,
            log_wg2,
            1u32,
        )
        .expect("ntt_pass (second) failed");
    }

    let bytes = client.read_one(buf_b);
    let transposed = u32::from_bytes(&bytes).to_vec();

    // Un-transpose: the output is in [N1][N2] layout.
    // natural[i_low + j * N1] = transposed[i_low * N2 + j]
    let mut natural = vec![0u32; n];
    for i_low in 0..n1 {
        for j in 0..n2 {
            natural[i_low + j * n1] = transposed[i_low * n2 + j];
        }
    }
    natural
}

// ---- Inverse runner (unchanged, uses old in-place intt_pass1/2) ----

fn run_inverse<P: MontyParameters, R: Runtime>(
    device: &R::Device,
    natural_evals_raw: &[u32],
    inv_twiddles_raw: &[u32],
    inv_n_raw: u32,
    log_n: u32,
) -> Vec<u32> {
    let n = 1usize << log_n;
    let log_n1 = pick_log_n1(log_n);
    let log_n2 = log_n - log_n1;
    let n1: usize = 1usize << log_n1;
    let n2: usize = 1usize << log_n2;

    let client = R::client(device);
    let data_h = client.create_from_slice(u32::as_bytes(natural_evals_raw));
    let tw_h = client.create_from_slice(u32::as_bytes(inv_twiddles_raw));
    let inv_n_h = client.create_from_slice(u32::as_bytes(&[inv_n_raw]));

    let log_wg1 = pick_log_wg(log_n2);
    let log_wg2 = pick_log_wg(log_n1);

    unsafe {
        intt_pass1::launch_unchecked::<P, R>(
            &client,
            CubeCount::Static(n1 as u32, 1, 1),
            CubeDim::new_1d(1u32 << log_wg1),
            ArrayArg::from_raw_parts::<u32>(&data_h, n, 1),
            ArrayArg::from_raw_parts::<u32>(&tw_h, n / 2, 1),
            ArrayArg::from_raw_parts::<u32>(&inv_n_h, 1, 1),
            log_n,
            log_n1,
            log_wg1,
        )
        .expect("intt_pass1 failed");

        intt_pass2::launch_unchecked::<P, R>(
            &client,
            CubeCount::Static(n2 as u32, 1, 1),
            CubeDim::new_1d(1u32 << log_wg2),
            ArrayArg::from_raw_parts::<u32>(&data_h, n, 1),
            ArrayArg::from_raw_parts::<u32>(&tw_h, n / 2, 1),
            log_n,
            log_n1,
            log_wg2,
        )
        .expect("intt_pass2 failed");
    }

    let bytes = client.read_one(data_h);
    u32::from_bytes(&bytes).to_vec()
}

// ---- Checks ----

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
    let twiddles = build_twiddles::<P>(log_n);

    let kernel_out_raw =
        run_forward::<P, R>(&Default::default(), &our_in_raw, &twiddles, log_n);

    let actual: Vec<u32> = kernel_out_raw
        .iter()
        .map(|&raw| MontyField::<P>::from_raw(raw).to_canonical())
        .collect();

    assert_eq!(
        actual, expected,
        "forward mismatch at log_n={log_n}: first divergence at {:?}",
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

    let kernel_out_raw = run_inverse::<P, R>(
        &Default::default(),
        &natural_raw,
        &inv_twiddles,
        inv_n,
        log_n,
    );
    let actual: Vec<u32> = kernel_out_raw
        .iter()
        .map(|&raw| MontyField::<P>::from_raw(raw).to_canonical())
        .collect();

    assert_eq!(
        actual, expected_bitrev,
        "inverse mismatch at log_n={log_n}: first divergence at {:?}",
        actual.iter().zip(expected_bitrev.iter()).position(|(a, b)| a != b)
    );
}

fn check_all<P: FieldBridge, R: Runtime>(range: impl IntoIterator<Item = u32>)
where
    R::Device: Default,
{
    for log_n in range {
        check_forward::<P, R>(log_n);
        check_inverse::<P, R>(log_n);
    }
}

// wgpu: full range.
#[test]
fn bb_wgpu()  { check_all::<BabyBearParameters,  WgpuRuntime>([11u32, 12, 14, 16, 20]); }
#[test]
fn kb_wgpu()  { check_all::<KoalaBearParameters, WgpuRuntime>([11u32, 12, 14, 16, 20]); }

// CUDA: full range.
#[test]
fn bb_cuda()  { check_all::<BabyBearParameters,  cubecl::cuda::CudaRuntime>([11u32, 12, 14, 16, 20]); }

// cubecl-cpu: spot check.
#[test]
#[ignore]
fn bb_cpu()   { check_all::<BabyBearParameters,  CpuRuntime>([14u32]); }
#[test]
#[ignore]
fn kb_cpu()   { check_all::<KoalaBearParameters, CpuRuntime>([14u32]); }
