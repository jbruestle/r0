//! Plonky3 oracle for the two-pass forward NTT (`log_n ∈ [11, 16]`).
//! Same pattern as `tests/p3_oracle.rs` for the monolithic kernel,
//! but launches `ntt_pass1` followed by `ntt_pass2` and uses our
//! decomposition `log_n1 + log_n2 = log_n` with `log_n1 = log_n / 2`.

use cubecl::cpu::CpuRuntime;
use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;

use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_field::{PrimeField32, TwoAdicField};

use r0_field::{BabyBearParameters, KoalaBearParameters, MontyField, MontyParameters};
use r0_ntt::{bit_reverse_in_place, build_twiddles, ntt_pass1, ntt_pass2};

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

/// Pick a balanced split. For odd `log_n`, give pass 2 the extra stage
/// (arbitrary — both kernels are symmetric in cost).
fn pick_log_n1(log_n: u32) -> u32 {
    log_n / 2
}

/// Each pass needs `log_wg ≤ log_pass_size − 1`; cap at 8 for WebGPU
/// portability (workgroup size ≤ 256). `log_pass_size_min` lets us
/// pick a single workgroup size that works for both passes when they
/// differ.
fn pick_log_wg(log_pass_size: u32) -> u32 {
    log_pass_size.saturating_sub(1).min(8)
}

fn run_two_pass<P: MontyParameters, R: Runtime>(
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
    let wg1: u32 = 1u32 << log_wg1;
    let wg2: u32 = 1u32 << log_wg2;

    // Pass 1: N2 workgroups, each processing N1 contiguous elements.
    unsafe {
        ntt_pass1::launch_unchecked::<P, R>(
            &client,
            CubeCount::Static(n2 as u32, 1, 1),
            CubeDim::new_1d(wg1),
            ArrayArg::from_raw_parts::<u32>(&data_h, n, 1),
            ArrayArg::from_raw_parts::<u32>(&tw_h, n / 2, 1),
            log_n,
            log_n1,
            log_wg1,
        )
        .expect("pass 1 launch failed");
    }

    // Pass 2: N1 workgroups, each processing N2 strided elements.
    unsafe {
        ntt_pass2::launch_unchecked::<P, R>(
            &client,
            CubeCount::Static(n1 as u32, 1, 1),
            CubeDim::new_1d(wg2),
            ArrayArg::from_raw_parts::<u32>(&data_h, n, 1),
            ArrayArg::from_raw_parts::<u32>(&tw_h, n / 2, 1),
            log_n,
            log_n1,
            log_wg2,
        )
        .expect("pass 2 launch failed");
    }

    let bytes = client.read_one(data_h);
    u32::from_bytes(&bytes).to_vec()
}

fn check_forward_two_pass<P: FieldBridge, R: Runtime>(log_n: u32)
where
    R::Device: Default,
{
    let n = 1usize << log_n;

    // Deterministic but non-trivial canonical inputs (natural-order
    // coefficients). We then bit-reverse to feed the kernel.
    let canonical: Vec<u32> = (0..n as u32)
        .map(|i| (i.wrapping_mul(0x9E3779B1) ^ 0xDEADBEEF) % P::PRIME)
        .collect();

    // Reference: Plonky3's dft (natural in → natural out).
    let p3_in: Vec<P::P3> = canonical.iter().map(|&v| P::p3_from_canonical(v)).collect();
    let dft = Radix2Dit::<P::P3>::default();
    let p3_out = dft.dft(p3_in);
    let expected: Vec<u32> = p3_out.iter().map(|f| f.as_canonical_u32()).collect();

    // Our path: bit-reverse natural canonical → Montgomery raw → kernel.
    let mut our_in_field: Vec<MontyField<P>> = canonical
        .iter()
        .map(|&v| MontyField::<P>::from_canonical(v))
        .collect();
    bit_reverse_in_place(&mut our_in_field);
    let our_in_raw: Vec<u32> = our_in_field.iter().map(|f| f.raw()).collect();

    let twiddles = build_twiddles::<P>(log_n);

    let kernel_out_raw = run_two_pass::<P, R>(&Default::default(), &our_in_raw, &twiddles, log_n);

    let actual: Vec<u32> = kernel_out_raw
        .iter()
        .map(|&raw| MontyField::<P>::from_raw(raw).to_canonical())
        .collect();

    assert_eq!(
        actual, expected,
        "two-pass NTT mismatch at log_n={log_n}: first divergence at index {:?}",
        actual.iter().zip(expected.iter()).position(|(a, b)| a != b)
    );
}

fn check_log_n_range<P: FieldBridge, R: Runtime>(range: impl IntoIterator<Item = u32>)
where
    R::Device: Default,
{
    for log_n in range {
        check_forward_two_pass::<P, R>(log_n);
    }
}

// wgpu/Metal: full range [11, 16]. Compile is fast, runtime is fast.
#[test]
fn bb_wgpu() { check_log_n_range::<BabyBearParameters,  WgpuRuntime>(11..=16); }
#[test]
fn kb_wgpu() { check_log_n_range::<KoalaBearParameters, WgpuRuntime>(11..=16); }

// cubecl-cpu: fresh MLIR/LLVM JIT per kernel-per-log_n combination, so
// keep this lean — we have wgpu coverage of the full range. Just spot
// check the boundaries.
#[test]
fn bb_cpu()  { check_log_n_range::<BabyBearParameters,  CpuRuntime >([11u32, 12]); }
#[test]
fn kb_cpu()  { check_log_n_range::<KoalaBearParameters, CpuRuntime >([11u32, 12]); }
