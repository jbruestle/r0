//! Plonky3 oracle for the two-pass NTT (`log_n ∈ [11, 20]`).
//! Forward, inverse, and round-trip checks. Pattern matches
//! `tests/p3_oracle.rs` for the monolithic kernel; CPU coverage is
//! trimmed (one log_n) since the cubecl-cpu MLIR/LLVM JIT cost grows
//! with each kernel monomorphization. wgpu (Metal here) carries the
//! breadth of size coverage, including the headline `log_n = 20`.

use cubecl::cpu::CpuRuntime;
use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;

use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_field::{PrimeField32, TwoAdicField};

use r0_field::{BabyBearParameters, KoalaBearParameters, MontyField, MontyParameters};
use r0_ntt::{
    bit_reverse_in_place, build_inv_twiddles, build_twiddles, intt_pass1, intt_pass2, n_inv,
    ntt_pass1, ntt_pass2,
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

// ---- Forward two-pass runner ----

fn run_two_pass_forward<P: MontyParameters, R: Runtime>(
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
    let data_h = client.create_from_slice(u32::as_bytes(bitrev_coeffs_raw));
    let tw_h = client.create_from_slice(u32::as_bytes(twiddles_raw));

    let log_wg1 = pick_log_wg(log_n1);
    let log_wg2 = pick_log_wg(log_n2);

    unsafe {
        ntt_pass1::launch_unchecked::<P, R>(
            &client,
            CubeCount::Static(n2 as u32, 1, 1),
            CubeDim::new_1d(1u32 << log_wg1),
            ArrayArg::from_raw_parts::<u32>(&data_h, n, 1),
            ArrayArg::from_raw_parts::<u32>(&tw_h, n / 2, 1),
            log_n,
            log_n1,
            log_wg1,
            1u32,
            false,
        )
        .expect("ntt_pass1 launch failed");

        ntt_pass2::launch_unchecked::<P, R>(
            &client,
            CubeCount::Static(n1 as u32, 1, 1),
            CubeDim::new_1d(1u32 << log_wg2),
            ArrayArg::from_raw_parts::<u32>(&data_h, n, 1),
            ArrayArg::from_raw_parts::<u32>(&tw_h, n / 2, 1),
            log_n,
            log_n1,
            log_wg2,
            1u32,
            false,
        )
        .expect("ntt_pass2 launch failed");
    }

    let bytes = client.read_one(data_h);
    u32::from_bytes(&bytes).to_vec()
}

// ---- Inverse two-pass runner ----

fn run_two_pass_inverse<P: MontyParameters, R: Runtime>(
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

    assert_eq!(natural_evals_raw.len(), n);
    assert_eq!(inv_twiddles_raw.len(), n / 2);

    let client = R::client(device);
    let data_h = client.create_from_slice(u32::as_bytes(natural_evals_raw));
    let tw_h = client.create_from_slice(u32::as_bytes(inv_twiddles_raw));
    let inv_n_h = client.create_from_slice(u32::as_bytes(&[inv_n_raw]));

    let log_wg1 = pick_log_wg(log_n2);
    let log_wg2 = pick_log_wg(log_n1);

    // Inverse pipeline: pass 1 (strided slabs, high-stride stages, with
    // N^-1 pre-mult) → pass 2 (contiguous chunks, low-stride stages).
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
        .expect("intt_pass1 launch failed");

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
        .expect("intt_pass2 launch failed");
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
        run_two_pass_forward::<P, R>(&Default::default(), &our_in_raw, &twiddles, log_n);

    let actual: Vec<u32> = kernel_out_raw
        .iter()
        .map(|&raw| MontyField::<P>::from_raw(raw).to_canonical())
        .collect();

    assert_eq!(
        actual, expected,
        "two-pass forward mismatch at log_n={log_n}: first divergence at {:?}",
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

    let kernel_out_raw = run_two_pass_inverse::<P, R>(
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
        "two-pass inverse mismatch at log_n={log_n}: first divergence at {:?}",
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

    let twiddles = build_twiddles::<P>(log_n);
    let inv_twiddles = build_inv_twiddles::<P>(log_n);
    let inv_n = n_inv::<P>(log_n);

    let natural_evals_raw =
        run_two_pass_forward::<P, R>(&Default::default(), &bitrev_raw, &twiddles, log_n);
    let recovered_raw = run_two_pass_inverse::<P, R>(
        &Default::default(),
        &natural_evals_raw,
        &inv_twiddles,
        inv_n,
        log_n,
    );
    let recovered: Vec<u32> = recovered_raw
        .iter()
        .map(|&raw| MontyField::<P>::from_raw(raw).to_canonical())
        .collect();

    assert_eq!(
        recovered, bitrev_canonical,
        "two-pass round-trip failed at log_n={log_n}: first divergence at {:?}",
        recovered
            .iter()
            .zip(bitrev_canonical.iter())
            .position(|(a, b)| a != b)
    );
}

fn check_all<P: FieldBridge, R: Runtime>(range: impl IntoIterator<Item = u32> + Clone)
where
    R::Device: Default,
{
    for log_n in range {
        check_forward::<P, R>(log_n);
        check_inverse::<P, R>(log_n);
        check_roundtrip::<P, R>(log_n);
    }
}

// wgpu/Metal: full range plus the headline 1M-point target.
#[test]
fn bb_wgpu()  { check_all::<BabyBearParameters,  WgpuRuntime>([11u32, 12, 14, 16, 20]); }
#[test]
fn kb_wgpu()  { check_all::<KoalaBearParameters, WgpuRuntime>([11u32, 12, 14, 16, 20]); }

// cubecl-cpu: one mid-size spot check. Each new log_n triggers a fresh
// MLIR/LLVM JIT for both kernels; wgpu carries the breadth.
#[test]
#[ignore]
fn bb_cpu()   { check_all::<BabyBearParameters,  CpuRuntime >([14u32]); }
#[test]
#[ignore]
fn kb_cpu()   { check_all::<KoalaBearParameters, CpuRuntime >([14u32]); }

// ---- Grid-Y batched correctness ----

/// Run a forward NTT on `batch_count` independent polynomials packed
/// contiguously, using grid-Y batching (one pair of kernel launches).
fn run_batched_forward<P: MontyParameters, R: Runtime>(
    device: &R::Device,
    all_bitrev_raw: &[u32],
    twiddles_raw: &[u32],
    log_n: u32,
    batch_count: u32,
) -> Vec<u32> {
    let n = 1usize << log_n;
    let log_n1 = pick_log_n1(log_n);
    let log_n2 = log_n - log_n1;
    let n1: usize = 1usize << log_n1;
    let n2: usize = 1usize << log_n2;

    assert_eq!(all_bitrev_raw.len(), n * batch_count as usize);
    assert_eq!(twiddles_raw.len(), n / 2);

    let client = R::client(device);
    let data_h = client.create_from_slice(u32::as_bytes(all_bitrev_raw));
    let tw_h = client.create_from_slice(u32::as_bytes(twiddles_raw));

    let log_wg1 = pick_log_wg(log_n1);
    let log_wg2 = pick_log_wg(log_n2);

    unsafe {
        ntt_pass1::launch_unchecked::<P, R>(
            &client,
            CubeCount::Static(n2 as u32, batch_count, 1),
            CubeDim::new_1d(1u32 << log_wg1),
            ArrayArg::from_raw_parts::<u32>(&data_h, n * batch_count as usize, 1),
            ArrayArg::from_raw_parts::<u32>(&tw_h, n / 2, 1),
            log_n,
            log_n1,
            log_wg1,
            1u32,
            false,
        )
        .expect("ntt_pass1 launch failed");

        ntt_pass2::launch_unchecked::<P, R>(
            &client,
            CubeCount::Static(n1 as u32, batch_count, 1),
            CubeDim::new_1d(1u32 << log_wg2),
            ArrayArg::from_raw_parts::<u32>(&data_h, n * batch_count as usize, 1),
            ArrayArg::from_raw_parts::<u32>(&tw_h, n / 2, 1),
            log_n,
            log_n1,
            log_wg2,
            1u32,
            false,
        )
        .expect("ntt_pass2 launch failed");
    }

    let bytes = client.read_one(data_h);
    u32::from_bytes(&bytes).to_vec()
}

/// Verify that batched forward NTT produces the same result as running
/// each polynomial independently.
fn check_batched_forward<P: FieldBridge, R: Runtime>(log_n: u32, batch_count: u32)
where
    R::Device: Default,
{
    let n = 1usize << log_n;
    let twiddles = build_twiddles::<P>(log_n);

    // Build `batch_count` distinct polynomials and compute single-poly
    // expected outputs.
    let mut all_bitrev_raw = Vec::with_capacity(n * batch_count as usize);
    let mut expected_all = Vec::with_capacity(n * batch_count as usize);

    for b in 0..batch_count {
        let seed = 0x9E3779B1u32.wrapping_mul(b.wrapping_add(1));
        let canonical: Vec<u32> = (0..n as u32)
            .map(|i| (i.wrapping_mul(seed) ^ 0xDEADBEEF) % P::PRIME)
            .collect();

        let mut field: Vec<MontyField<P>> = canonical
            .iter()
            .map(|&v| MontyField::<P>::from_canonical(v))
            .collect();
        bit_reverse_in_place(&mut field);
        let bitrev_raw: Vec<u32> = field.iter().map(|f| f.raw()).collect();

        // Single-poly reference
        let single = run_two_pass_forward::<P, R>(&Default::default(), &bitrev_raw, &twiddles, log_n);
        expected_all.extend_from_slice(&single);
        all_bitrev_raw.extend_from_slice(&bitrev_raw);
    }

    let batched_out = run_batched_forward::<P, R>(
        &Default::default(),
        &all_bitrev_raw,
        &twiddles,
        log_n,
        batch_count,
    );

    assert_eq!(
        batched_out, expected_all,
        "batched forward mismatch at log_n={log_n}, batch={batch_count}"
    );
}

#[test]
fn bb_batched_wgpu() {
    check_batched_forward::<BabyBearParameters, WgpuRuntime>(14, 4);
    check_batched_forward::<BabyBearParameters, WgpuRuntime>(16, 3);
}

#[test]
fn kb_batched_wgpu() {
    check_batched_forward::<KoalaBearParameters, WgpuRuntime>(14, 4);
}
