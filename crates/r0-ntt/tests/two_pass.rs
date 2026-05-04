//! Plonky3 oracle tests for forward NTT (`ntt_fwd_pass`) and inverse NTT
//! (`ntt_inv_pass`).

use cubecl::cpu::CpuRuntime;
use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;

use p3_dft::{Radix2Dit, TwoAdicSubgroupDft};
use p3_field::{PrimeField32, TwoAdicField};

use r0_field::{BabyBearParameters, KoalaBearParameters, MontyField, MontyParameters};
use r0_ntt::{
    bit_reverse_in_place, build_partial_fwd_twiddles, build_partial_inv_twiddles,
    ntt_inv_pass, n_inv, ntt_fwd_pass, PARTIAL_TWIDDLE_LEN,
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

// ---- Forward runner (unified ntt_fwd_pass, separate in/out buffers) ----

fn run_forward<P: MontyParameters, R: Runtime>(
    device: &R::Device,
    bitrev_coeffs_raw: &[u32],
    partial_twiddles_raw: &[u32],
    log_n: u32,
) -> Vec<u32> {
    let n = 1usize << log_n;
    let log_n1 = pick_log_n1(log_n);
    let log_n2 = log_n - log_n1;
    let n1: usize = 1usize << log_n1;
    let n2: usize = 1usize << log_n2;

    assert_eq!(bitrev_coeffs_raw.len(), n);
    assert_eq!(partial_twiddles_raw.len(), PARTIAL_TWIDDLE_LEN);

    let client = R::client(device);
    let buf_a = client.create_from_slice(u32::as_bytes(bitrev_coeffs_raw));
    let buf_b = client.create_from_slice(u32::as_bytes(&vec![0u32; n]));
    let tw_h = client.create_from_slice(u32::as_bytes(partial_twiddles_raw));

    let log_wg1 = pick_log_wg(log_n1);
    let log_wg2 = pick_log_wg(log_n2);

    unsafe {
        // Pass 1: buf_a -> buf_b (transposed, since non-final)
        ntt_fwd_pass::launch_unchecked::<P, R>(
            &client,
            CubeCount::Static(n2 as u32, 1, 1),
            CubeDim::new_1d(1u32 << log_wg1),
            ArrayArg::from_raw_parts::<u32>(&buf_a, n, 1),
            ArrayArg::from_raw_parts::<u32>(&buf_b, n, 1),
            ArrayArg::from_raw_parts::<u32>(&tw_h, PARTIAL_TWIDDLE_LEN, 1),
            log_n,
            log_n1,
            0u32,
            log_wg1,
            1u32,
        )
        .expect("ntt_fwd_pass (first) failed");

        // Pass 2: buf_b -> buf_b (final, in-place safe)
        ntt_fwd_pass::launch_unchecked::<P, R>(
            &client,
            CubeCount::Static(n1 as u32, 1, 1),
            CubeDim::new_1d(1u32 << log_wg2),
            ArrayArg::from_raw_parts::<u32>(&buf_b, n, 1),
            ArrayArg::from_raw_parts::<u32>(&buf_b, n, 1),
            ArrayArg::from_raw_parts::<u32>(&tw_h, PARTIAL_TWIDDLE_LEN, 1),
            log_n,
            log_n2,
            log_n1,
            log_wg2,
            1u32,
        )
        .expect("ntt_fwd_pass (second) failed");
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

// ---- Inverse runner (unified ntt_inv_pass, separate in/out buffers) ----

fn run_inverse<P: MontyParameters, R: Runtime>(
    device: &R::Device,
    natural_evals_raw: &[u32],
    partial_inv_twiddles_raw: &[u32],
    inv_n_raw: u32,
    log_n: u32,
) -> Vec<u32> {
    let n = 1usize << log_n;
    let log_n1 = pick_log_n1(log_n);
    let log_n2 = log_n - log_n1;
    let n1: usize = 1usize << log_n1;
    let n2: usize = 1usize << log_n2;

    let client = R::client(device);

    // Pre-transpose input from natural [N2][N1] to [N1][N2] so the
    // first inverse pass (N1 workgroups, each reading N2 contiguous
    // elements) picks up the correct stride-N1 slabs.
    let mut transposed_input = vec![0u32; n];
    for i in 0..n1 {
        for j in 0..n2 {
            transposed_input[i * n2 + j] = natural_evals_raw[i + j * n1];
        }
    }

    let buf_a = client.create_from_slice(u32::as_bytes(&transposed_input));
    let buf_b = client.create_from_slice(u32::as_bytes(&vec![0u32; n]));
    let tw_h = client.create_from_slice(u32::as_bytes(partial_inv_twiddles_raw));
    let inv_n_h = client.create_from_slice(u32::as_bytes(&[inv_n_raw]));

    let log_wg1 = pick_log_wg(log_n2);
    let log_wg2 = pick_log_wg(log_n1);

    // Inverse pass 1: high-stride stages (descending), N^{-1} pre-mult.
    // buf_a -> buf_b (transposed for pass 2).
    unsafe {
        ntt_inv_pass::launch_unchecked::<P, R>(
            &client,
            CubeCount::Static(n1 as u32, 1, 1),
            CubeDim::new_1d(1u32 << log_wg1),
            ArrayArg::from_raw_parts::<u32>(&buf_a, n, 1),
            ArrayArg::from_raw_parts::<u32>(&buf_b, n, 1),
            ArrayArg::from_raw_parts::<u32>(&tw_h, PARTIAL_TWIDDLE_LEN, 1),
            ArrayArg::from_raw_parts::<u32>(&inv_n_h, 1, 1),
            log_n,
            log_n2,
            0u32,
            log_wg1,
            1u32,
        )
        .expect("ntt_inv_pass (first) failed");

        // Inverse pass 2: low-stride stages (descending).
        // buf_b -> buf_b (final, in-place safe).
        ntt_inv_pass::launch_unchecked::<P, R>(
            &client,
            CubeCount::Static(n2 as u32, 1, 1),
            CubeDim::new_1d(1u32 << log_wg2),
            ArrayArg::from_raw_parts::<u32>(&buf_b, n, 1),
            ArrayArg::from_raw_parts::<u32>(&buf_b, n, 1),
            ArrayArg::from_raw_parts::<u32>(&tw_h, PARTIAL_TWIDDLE_LEN, 1),
            ArrayArg::from_raw_parts::<u32>(&inv_n_h, 1, 1),
            log_n,
            log_n1,
            log_n2,
            log_wg2,
            1u32,
        )
        .expect("ntt_inv_pass (second) failed");
    }

    let bytes = client.read_one(buf_b);
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
    let partial_twiddles = build_partial_fwd_twiddles::<P>(log_n);

    let kernel_out_raw =
        run_forward::<P, R>(&Default::default(), &our_in_raw, &partial_twiddles, log_n);

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
    let partial_inv_twiddles = build_partial_inv_twiddles::<P>(log_n);
    let inv_n = n_inv::<P>(log_n);

    let kernel_out_raw = run_inverse::<P, R>(
        &Default::default(),
        &natural_raw,
        &partial_inv_twiddles,
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

// ---- 3-pass forward test (log_n = 21) ----

/// 3-pass forward NTT. Decomposition: (7, 7, 7) for log_n=21.
/// Pass 1: stages 0..6,  pass 2: stages 7..13, pass 3: stages 14..20.
/// Ping-pong: buf_a -> buf_b -> buf_a -> result in buf_a (3 passes = odd,
/// but pass 2 is non-final so it transposes too, and pass 3 is final).
/// Actually: pass1 non-final (a->b), pass2 non-final (b->a), pass3 final (a->a in-place).
fn run_forward_3pass<P: MontyParameters, R: Runtime>(
    device: &R::Device,
    bitrev_coeffs_raw: &[u32],
    partial_twiddles_raw: &[u32],
    log_n: u32,
) -> Vec<u32> {
    let n = 1usize << log_n;
    // Balanced 3-way split
    let step = log_n / 3;
    let rem = log_n % 3;
    let log_a = step + if rem > 0 { 1 } else { 0 };
    let log_b = step + if rem > 1 { 1 } else { 0 };
    let log_c = step;
    assert_eq!(log_a + log_b + log_c, log_n);

    let n_bc = 1usize << (log_b + log_c); // number of chunks for pass 1
    let n_ac = 1usize << (log_a + log_c); // number of chunks for pass 2
    let n_ab = 1usize << (log_a + log_b); // number of chunks for pass 3

    let client = R::client(device);
    let buf_a = client.create_from_slice(u32::as_bytes(bitrev_coeffs_raw));
    let buf_b = client.create_from_slice(u32::as_bytes(&vec![0u32; n]));
    let tw_h = client.create_from_slice(u32::as_bytes(partial_twiddles_raw));

    let log_wg_a = pick_log_wg(log_a);
    let log_wg_b = pick_log_wg(log_b);
    let log_wg_c = pick_log_wg(log_c);

    unsafe {
        // Pass 1: stages 0..log_a, buf_a -> buf_b (non-final, transposes)
        ntt_fwd_pass::launch_unchecked::<P, R>(
            &client,
            CubeCount::Static(n_bc as u32, 1, 1),
            CubeDim::new_1d(1u32 << log_wg_a),
            ArrayArg::from_raw_parts::<u32>(&buf_a, n, 1),
            ArrayArg::from_raw_parts::<u32>(&buf_b, n, 1),
            ArrayArg::from_raw_parts::<u32>(&tw_h, PARTIAL_TWIDDLE_LEN, 1),
            log_n,
            log_a,
            0u32,
            log_wg_a,
            1u32,
        ).expect("3-pass: pass 1 failed");

        // Pass 2: stages log_a..log_a+log_b, buf_b -> buf_a (non-final, transposes)
        ntt_fwd_pass::launch_unchecked::<P, R>(
            &client,
            CubeCount::Static(n_ac as u32, 1, 1),
            CubeDim::new_1d(1u32 << log_wg_b),
            ArrayArg::from_raw_parts::<u32>(&buf_b, n, 1),
            ArrayArg::from_raw_parts::<u32>(&buf_a, n, 1),
            ArrayArg::from_raw_parts::<u32>(&tw_h, PARTIAL_TWIDDLE_LEN, 1),
            log_n,
            log_b,
            log_a,
            log_wg_b,
            1u32,
        ).expect("3-pass: pass 2 failed");

        // Pass 3: stages log_a+log_b..log_n, buf_a -> buf_a (final, in-place)
        ntt_fwd_pass::launch_unchecked::<P, R>(
            &client,
            CubeCount::Static(n_ab as u32, 1, 1),
            CubeDim::new_1d(1u32 << log_wg_c),
            ArrayArg::from_raw_parts::<u32>(&buf_a, n, 1),
            ArrayArg::from_raw_parts::<u32>(&buf_a, n, 1),
            ArrayArg::from_raw_parts::<u32>(&tw_h, PARTIAL_TWIDDLE_LEN, 1),
            log_n,
            log_c,
            log_a + log_b,
            log_wg_c,
            1u32,
        ).expect("3-pass: pass 3 failed");
    }

    let raw = u32::from_bytes(&client.read_one(buf_a)).to_vec();

    // The 3-pass output layout after two transposes + final contiguous store
    // is [N_b][N_a][N_c]. Un-transpose to natural order.
    let n_a = 1usize << log_a;
    let n_b = 1usize << log_b;
    let n_c = 1usize << log_c;
    let mut natural = vec![0u32; n];
    for m in 0..n {
        let i_b = m / (n_a * n_c);
        let i_a = (m / n_c) % n_a;
        let i_c = m % n_c;
        let p = i_a + i_b * n_a + i_c * n_a * n_b;
        natural[p] = raw[m];
    }
    natural
}

#[test]
fn three_pass_forward_cuda_21() {
    type P = BabyBearParameters;
    type R = cubecl::cuda::CudaRuntime;
    let log_n = 21u32;
    let n = 1usize << log_n;

    let canonical: Vec<u32> = (0..n as u32)
        .map(|i| (i.wrapping_mul(0x9E3779B1) ^ 0xDEADBEEF) % P::PRIME)
        .collect();

    // Plonky3 reference
    let p3_in: Vec<p3_baby_bear::BabyBear> = canonical.iter()
        .map(|&v| p3_baby_bear::BabyBear::new(v)).collect();
    let dft = Radix2Dit::<p3_baby_bear::BabyBear>::default();
    let p3_out = dft.dft(p3_in);
    let expected: Vec<u32> = p3_out.iter().map(|f| f.as_canonical_u32()).collect();

    // Our path
    let mut our_in_field: Vec<MontyField<P>> = canonical.iter()
        .map(|&v| MontyField::<P>::from_canonical(v)).collect();
    bit_reverse_in_place(&mut our_in_field);
    let our_in_raw: Vec<u32> = our_in_field.iter().map(|f| f.raw()).collect();
    let partial_twiddles = build_partial_fwd_twiddles::<P>(log_n);

    let result = run_forward_3pass::<P, R>(
        &Default::default(), &our_in_raw, &partial_twiddles, log_n);

    let actual: Vec<u32> = result.iter()
        .map(|&raw| MontyField::<P>::from_raw(raw).to_canonical()).collect();

    assert_eq!(
        actual, expected,
        "3-pass forward mismatch at log_n={log_n}: first divergence at {:?}",
        actual.iter().zip(expected.iter()).position(|(a, b)| a != b)
    );
}
