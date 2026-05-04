//! Per-pass forward NTT timing, BabyBear, log_n=20, CUDA.
//!
//! Benchmarks pass1 and pass2 independently so we can see where time
//! is spent. Uses z_count=8 to match the combined benchmark.

use criterion::{criterion_group, criterion_main, Criterion};
use cubecl::cuda::CudaRuntime;
use cubecl::prelude::*;

use r0_field::{BabyBearParameters, MontyField, MontyParameters};
use r0_ntt::{bit_reverse_in_place, build_twiddles, ntt_pass1, ntt_pass2};

fn forward_ntt_split(c: &mut Criterion) {
    type P = BabyBearParameters;
    type R = CudaRuntime;

    const LOG_N: u32 = 20;
    const LOG_N1: u32 = 10;
    const LOG_N2: u32 = 10;
    const LOG_WG: u32 = 8;
    const Z: u32 = 8;

    let n: usize = 1usize << LOG_N;
    let n1: usize = 1usize << LOG_N1;
    let n2: usize = 1usize << LOG_N2;

    let canonical: Vec<u32> = (0..n as u32)
        .map(|i| (i.wrapping_mul(0x9E3779B1) ^ 0xDEADBEEF) % P::PRIME)
        .collect();
    let mut input_field: Vec<MontyField<P>> = canonical
        .iter()
        .map(|&v| MontyField::<P>::from_canonical(v))
        .collect();
    bit_reverse_in_place(&mut input_field);
    let input_raw: Vec<u32> = input_field.iter().map(|f| f.raw()).collect();
    let twiddles = build_twiddles::<P>(LOG_N);

    let device = <R as Runtime>::Device::default();
    let client = R::client(&device);

    let data_h = client.create_from_slice(u32::as_bytes(&input_raw));
    let tw_h = client.create_from_slice(u32::as_bytes(&twiddles));

    let mut group = c.benchmark_group("forward_ntt_bb_log_n20_cuda_split");
    group.throughput(criterion::Throughput::Elements(n as u64));

    // ---- Pass 1 only: contiguous chunks ----
    group.bench_function("pass1_only", |b| {
        b.iter(|| {
            unsafe {
                ntt_pass1::launch_unchecked::<P, R>(
                    &client,
                    CubeCount::Static((n2 / Z as usize) as u32, 1, 1),
                    CubeDim::new_1d(1u32 << LOG_WG),
                    ArrayArg::from_raw_parts::<u32>(&data_h, n, 1),
                    ArrayArg::from_raw_parts::<u32>(&tw_h, n / 2, 1),
                    LOG_N,
                    LOG_N1,
                    LOG_WG,
                    Z,
                    true,
                )
                .expect("ntt_pass1 launch failed");
            }
            cubecl_common::reader::read_sync(client.sync()).expect("sync failed");
        });
    });

    // ---- Pass 2 only: strided slabs ----
    group.bench_function("pass2_only", |b| {
        b.iter(|| {
            unsafe {
                ntt_pass2::launch_unchecked::<P, R>(
                    &client,
                    CubeCount::Static((n1 / Z as usize) as u32, 1, 1),
                    CubeDim::new_1d(1u32 << LOG_WG),
                    ArrayArg::from_raw_parts::<u32>(&data_h, n, 1),
                    ArrayArg::from_raw_parts::<u32>(&tw_h, n / 2, 1),
                    LOG_N,
                    LOG_N1,
                    LOG_WG,
                    Z,
                    true,
                )
                .expect("ntt_pass2 launch failed");
            }
            cubecl_common::reader::read_sync(client.sync()).expect("sync failed");
        });
    });

    group.finish();
}

criterion_group!(benches, forward_ntt_split);
criterion_main!(benches);
