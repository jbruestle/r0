//! Plonky3 oracle for the monolithic forward NTT kernel.
//!
//! Plonky3's `Radix2Dit::dft` consumes natural-order coefficients and
//! produces natural-order evaluations (it bit-reverses internally before
//! running CT-DIT). Our kernel skips the bit-reversal pass and consumes
//! bit-reversed coefficients directly. Mathematically:
//!
//!     plonky3.dft(x)         = CT-DIT(bit_rev(x))      = X
//!     our_kernel(bit_rev(x)) = CT-DIT(bit_rev(x))      = X
//!
//! So both should produce the same X for the same logical x.

use cubecl::cpu::CpuRuntime;
use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;

use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_field::{PrimeField32, TwoAdicField};

use r0_field::{BabyBearParameters, KoalaBearParameters, MontyField, MontyParameters};
use r0_ntt::{
    bit_reverse_in_place, build_inv_twiddles, build_twiddles, n_inv, ntt_monolithic,
    ntt_monolithic_inverse,
};

fn run_ntt<P: MontyParameters, R: Runtime>(
    device: &R::Device,
    bitrev_coeffs_raw: &[u32],
    twiddles_raw: &[u32],
    log_n: u32,
    log_wg: u32,
) -> Vec<u32> {
    let n = 1usize << log_n;
    assert_eq!(bitrev_coeffs_raw.len(), n);
    assert_eq!(twiddles_raw.len(), n / 2);

    let client = R::client(device);
    let in_h = client.create_from_slice(u32::as_bytes(bitrev_coeffs_raw));
    let tw_h = client.create_from_slice(u32::as_bytes(twiddles_raw));
    let out_h = client.empty(n * core::mem::size_of::<u32>());

    let wg_size = 1u32 << log_wg;

    unsafe {
        ntt_monolithic::launch_unchecked::<P, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(wg_size),
            ArrayArg::from_raw_parts::<u32>(&in_h, n, 1),
            ArrayArg::from_raw_parts::<u32>(&tw_h, n / 2, 1),
            ArrayArg::from_raw_parts::<u32>(&out_h, n, 1),
            log_n,
            log_wg,
        )
        .expect("kernel launch failed");
    }

    let bytes = client.read_one(out_h);
    u32::from_bytes(&bytes).to_vec()
}

/// Pick a workgroup-size exponent suitable for `log_n`. Each thread
/// must do at least one butterfly per stage, so `log_wg <= log_n - 1`.
/// We also cap at 8 (256 threads) for WebGPU portability.
fn pick_log_wg(log_n: u32) -> u32 {
    let max_useful = log_n.saturating_sub(1);
    max_useful.min(8)
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

fn check_forward<P: FieldBridge, R: Runtime>(log_n: u32)
where
    R::Device: Default,
{
    let n = 1usize << log_n;
    let log_wg = pick_log_wg(log_n);

    // Deterministic but non-trivial canonical inputs.
    let canonical: Vec<u32> = (0..n as u32)
        .map(|i| (i.wrapping_mul(0x9E3779B1) ^ 0xDEADBEEF) % P::PRIME)
        .collect();

    // ---- Reference path: Plonky3 dft (natural in → natural out) ----
    let p3_in: Vec<P::P3> = canonical.iter().map(|&v| P::p3_from_canonical(v)).collect();
    let dft = Radix2Dit::<P::P3>::default();
    let p3_out = dft.dft(p3_in);
    let expected: Vec<u32> = p3_out.iter().map(|f| f.as_canonical_u32()).collect();

    // ---- Our path: bit-reverse coeffs → kernel → canonical ----
    let mut our_in_field: Vec<MontyField<P>> = canonical
        .iter()
        .map(|&v| MontyField::<P>::from_canonical(v))
        .collect();
    bit_reverse_in_place(&mut our_in_field);
    let our_in_raw: Vec<u32> = our_in_field.iter().map(|f| f.raw()).collect();

    let twiddles = build_twiddles::<P>(log_n);

    let kernel_out_raw = run_ntt::<P, R>(&Default::default(), &our_in_raw, &twiddles, log_n, log_wg);

    let actual: Vec<u32> = kernel_out_raw
        .iter()
        .map(|&raw| MontyField::<P>::from_raw(raw).to_canonical())
        .collect();

    assert_eq!(
        actual, expected,
        "NTT mismatch at log_n={log_n}: first divergence at index {:?}",
        actual.iter().zip(expected.iter()).position(|(a, b)| a != b)
    );
}

// ---- Inverse kernel runner + oracle/round-trip checks ----

fn run_intt<P: MontyParameters, R: Runtime>(
    device: &R::Device,
    natural_evals_raw: &[u32],
    inv_twiddles_raw: &[u32],
    inv_n_raw: u32,
    log_n: u32,
    log_wg: u32,
) -> Vec<u32> {
    let n = 1usize << log_n;
    assert_eq!(natural_evals_raw.len(), n);
    assert_eq!(inv_twiddles_raw.len(), n / 2);

    let client = R::client(device);
    let in_h = client.create_from_slice(u32::as_bytes(natural_evals_raw));
    let tw_h = client.create_from_slice(u32::as_bytes(inv_twiddles_raw));
    let inv_n_h = client.create_from_slice(u32::as_bytes(&[inv_n_raw]));
    let out_h = client.empty(n * core::mem::size_of::<u32>());

    let wg_size = 1u32 << log_wg;

    unsafe {
        ntt_monolithic_inverse::launch_unchecked::<P, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(wg_size),
            ArrayArg::from_raw_parts::<u32>(&in_h, n, 1),
            ArrayArg::from_raw_parts::<u32>(&tw_h, n / 2, 1),
            ArrayArg::from_raw_parts::<u32>(&inv_n_h, 1, 1),
            ArrayArg::from_raw_parts::<u32>(&out_h, n, 1),
            log_n,
            log_wg,
        )
        .expect("inverse kernel launch failed");
    }

    let bytes = client.read_one(out_h);
    u32::from_bytes(&bytes).to_vec()
}

/// Round-trip: bit-reversed canonical x → forward NTT → inverse NTT →
/// bit-reversed canonical x. Both kernels run on the same backend.
fn check_roundtrip<P: FieldBridge, R: Runtime>(log_n: u32)
where
    R::Device: Default,
{
    let n = 1usize << log_n;
    let log_wg = pick_log_wg(log_n);

    // Random canonical input. Treated as bit-reversed coefficients
    // (our convention).
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

    // Forward: bit-rev coeffs → natural evals.
    let natural_evals_raw =
        run_ntt::<P, R>(&Default::default(), &bitrev_raw, &twiddles, log_n, log_wg);

    // Inverse: natural evals → bit-rev coeffs (should be the original).
    let recovered_raw = run_intt::<P, R>(
        &Default::default(),
        &natural_evals_raw,
        &inv_twiddles,
        inv_n,
        log_n,
        log_wg,
    );

    let recovered: Vec<u32> = recovered_raw
        .iter()
        .map(|&raw| MontyField::<P>::from_raw(raw).to_canonical())
        .collect();

    assert_eq!(
        recovered, bitrev_canonical,
        "round-trip failed at log_n={log_n}: first divergence at index {:?}",
        recovered
            .iter()
            .zip(bitrev_canonical.iter())
            .position(|(a, b)| a != b)
    );
}

/// Plonky3 idft oracle: their `idft` consumes natural evals and produces
/// natural coefficients. Our kernel produces bit-reversed coefficients,
/// so we bit-reverse on the test side to compare.
fn check_inverse<P: FieldBridge, R: Runtime>(log_n: u32)
where
    R::Device: Default,
{
    let n = 1usize << log_n;
    let log_wg = pick_log_wg(log_n);

    // Random natural-order evaluations.
    let natural_evals: Vec<u32> = (0..n as u32)
        .map(|i| (i.wrapping_mul(0xC2B2AE3D) ^ 0x85EBCA77) % P::PRIME)
        .collect();

    // ---- Reference: Plonky3 idft (natural in → natural out) ----
    let p3_in: Vec<P::P3> = natural_evals.iter().map(|&v| P::p3_from_canonical(v)).collect();
    let dft = Radix2Dit::<P::P3>::default();
    let p3_out = dft.idft(p3_in);
    let expected_natural_coeffs: Vec<u32> = p3_out.iter().map(|f| f.as_canonical_u32()).collect();
    // Apply bit-reversal on the host side to align with our convention.
    let mut expected_bitrev = expected_natural_coeffs.clone();
    bit_reverse_in_place(&mut expected_bitrev);

    // ---- Our path: kernel → bit-rev coeffs ----
    let natural_evals_raw: Vec<u32> = natural_evals
        .iter()
        .map(|&v| MontyField::<P>::from_canonical(v).raw())
        .collect();
    let inv_twiddles = build_inv_twiddles::<P>(log_n);
    let inv_n = n_inv::<P>(log_n);

    let actual_raw = run_intt::<P, R>(
        &Default::default(),
        &natural_evals_raw,
        &inv_twiddles,
        inv_n,
        log_n,
        log_wg,
    );
    let actual: Vec<u32> = actual_raw
        .iter()
        .map(|&raw| MontyField::<P>::from_raw(raw).to_canonical())
        .collect();

    assert_eq!(
        actual, expected_bitrev,
        "iNTT mismatch at log_n={log_n}: first divergence at index {:?}",
        actual
            .iter()
            .zip(expected_bitrev.iter())
            .position(|(a, b)| a != b)
    );
}

/// Run all of forward-oracle, inverse-oracle, and round-trip across
/// `log_n ∈ [1, 10]` on the given (field, runtime) pair.
fn check_all<P: FieldBridge, R: Runtime>()
where
    R::Device: Default,
{
    for log_n in 1..=10u32 {
        check_forward::<P, R>(log_n);
        check_inverse::<P, R>(log_n);
        check_roundtrip::<P, R>(log_n);
    }
}

#[test]
fn bb_cpu()  { check_all::<BabyBearParameters,  CpuRuntime >(); }
#[test]
fn bb_wgpu() { check_all::<BabyBearParameters,  WgpuRuntime>(); }
#[test]
fn kb_cpu()  { check_all::<KoalaBearParameters, CpuRuntime >(); }
#[test]
fn kb_wgpu() { check_all::<KoalaBearParameters, WgpuRuntime>(); }
