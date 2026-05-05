//! Forward NTT throughput, BabyBear, log_n=20 (1M points), wgpu.

use criterion::{criterion_group, criterion_main, Criterion};
use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;

use r0_field::{BabyBearParameters, MontyField, MontyParameters};
use r0_ntt::{bit_reverse_in_place, build_partial_fwd_twiddles, ntt_fwd_pass, PARTIAL_TWIDDLE_LEN};

fn forward_ntt(c: &mut Criterion) {
    type P = BabyBearParameters;
    type R = WgpuRuntime;

    const LOG_N: u32 = 20;
    const LOG_N1: u32 = 10;
    const LOG_N2: u32 = 10;
    const LOG_WG: u32 = 8;

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
    let partial_twiddles = build_partial_fwd_twiddles::<P>(LOG_N);

    let device = <R as Runtime>::Device::default();
    let client = R::client(&device);

    let buf_a = client.create_from_slice(u32::as_bytes(&input_raw));
    let buf_b = client.create_from_slice(u32::as_bytes(&vec![0u32; n]));
    let tw_h = client.create_from_slice(u32::as_bytes(&partial_twiddles));

    let mut group = c.benchmark_group("forward_ntt_bb_log_n20_wgpu");
    group.throughput(criterion::Throughput::Elements(n as u64));
    group.bench_function("forward", |b| {
        b.iter(|| {
            unsafe {
                ntt_fwd_pass::launch_unchecked::<P, R>(
                    &client,
                    CubeCount::Static(n2 as u32, 1, 1),
                    CubeDim::new_1d(1u32 << LOG_WG),
                    ArrayArg::from_raw_parts::<u32>(&buf_a, n, 1),
                    ArrayArg::from_raw_parts::<u32>(&buf_b, n, 1),
                    ArrayArg::from_raw_parts::<u32>(&tw_h, PARTIAL_TWIDDLE_LEN, 1),
                    LOG_N,
                    LOG_N1,
                    0u32,
                    LOG_WG,
                    1u32,
                )
                .expect("ntt_fwd_pass (first) failed");

                ntt_fwd_pass::launch_unchecked::<P, R>(
                    &client,
                    CubeCount::Static(n1 as u32, 1, 1),
                    CubeDim::new_1d(1u32 << LOG_WG),
                    ArrayArg::from_raw_parts::<u32>(&buf_b, n, 1),
                    ArrayArg::from_raw_parts::<u32>(&buf_b, n, 1),
                    ArrayArg::from_raw_parts::<u32>(&tw_h, PARTIAL_TWIDDLE_LEN, 1),
                    LOG_N,
                    LOG_N2,
                    LOG_N1,
                    LOG_WG,
                    1u32,
                )
                .expect("ntt_fwd_pass (second) failed");
            }

            cubecl_common::reader::read_sync(client.sync()).expect("sync failed");
        });
    });
    group.finish();
}

criterion_group!(benches, forward_ntt);
criterion_main!(benches);
