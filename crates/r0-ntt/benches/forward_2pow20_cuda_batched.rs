//! Batched forward NTT throughput, BabyBear, log_n=20, batch=100, CUDA.
//!
//! Tests grid-Y batching: 100 independent 1M-point NTTs in a single
//! pair of kernel launches. Measures total throughput (100M points per
//! iteration).

use criterion::{criterion_group, criterion_main, Criterion};
use cubecl::cuda::CudaRuntime;
use cubecl::prelude::*;

use r0_field::{BabyBearParameters, MontyField, MontyParameters};
use r0_ntt::{bit_reverse_in_place, build_twiddles, ntt_pass1, ntt_pass2};

fn forward_ntt_batched(c: &mut Criterion) {
    type P = BabyBearParameters;
    type R = CudaRuntime;

    const LOG_N: u32 = 20;
    const LOG_N1: u32 = 10;
    const LOG_N2: u32 = 10;
    const LOG_WG: u32 = 8;
    const BATCH: usize = 100;

    let n: usize = 1usize << LOG_N;
    let n1: usize = 1usize << LOG_N1;
    let n2: usize = 1usize << LOG_N2;

    // ---- Build one polynomial's worth of input, replicate BATCH times ----
    let canonical: Vec<u32> = (0..n as u32)
        .map(|i| (i.wrapping_mul(0x9E3779B1) ^ 0xDEADBEEF) % P::PRIME)
        .collect();
    let mut input_field: Vec<MontyField<P>> = canonical
        .iter()
        .map(|&v| MontyField::<P>::from_canonical(v))
        .collect();
    bit_reverse_in_place(&mut input_field);
    let one_poly_raw: Vec<u32> = input_field.iter().map(|f| f.raw()).collect();

    // Pack BATCH polynomials contiguously.
    let mut all_raw: Vec<u32> = Vec::with_capacity(BATCH * n);
    for _ in 0..BATCH {
        all_raw.extend_from_slice(&one_poly_raw);
    }

    let twiddles = build_twiddles::<P>(LOG_N);

    // ---- Persistent device buffers ----
    let device = <R as Runtime>::Device::default();
    let client = R::client(&device);

    let data_h = client.create_from_slice(u32::as_bytes(&all_raw));
    let tw_h = client.create_from_slice(u32::as_bytes(&twiddles));

    let total_elements = (BATCH * n) as u64;

    let mut group = c.benchmark_group("forward_ntt_bb_log_n20_cuda_batch100");
    group.throughput(criterion::Throughput::Elements(total_elements));
    group.bench_function("forward_batched", |b| {
        b.iter(|| {
            unsafe {
                ntt_pass1::launch_unchecked::<P, R>(
                    &client,
                    CubeCount::Static(n2 as u32, BATCH as u32, 1),
                    CubeDim::new_1d(1u32 << LOG_WG),
                    ArrayArg::from_raw_parts::<u32>(&data_h, BATCH * n, 1),
                    ArrayArg::from_raw_parts::<u32>(&tw_h, n / 2, 1),
                    LOG_N,
                    LOG_N1,
                    LOG_WG,
                )
                .expect("ntt_pass1 launch failed");

                ntt_pass2::launch_unchecked::<P, R>(
                    &client,
                    CubeCount::Static(n1 as u32, BATCH as u32, 1),
                    CubeDim::new_1d(1u32 << LOG_WG),
                    ArrayArg::from_raw_parts::<u32>(&data_h, BATCH * n, 1),
                    ArrayArg::from_raw_parts::<u32>(&tw_h, n / 2, 1),
                    LOG_N,
                    LOG_N1,
                    LOG_WG,
                )
                .expect("ntt_pass2 launch failed");
            }

            cubecl_common::reader::read_sync(client.sync()).expect("sync failed");
        });
    });
    group.finish();
}

criterion_group!(benches, forward_ntt_batched);
criterion_main!(benches);
