//! NTT benchmarks through the NttExec API.

use criterion::{criterion_group, criterion_main, Criterion};
use cubecl::prelude::*;

use r0_field::{MontyField, MontyParameters};
use r0_ntt::NttExec;

#[derive(Clone, Copy)]
enum Direction {
    Forward,
    Inverse,
}

fn bench_ntt<P: MontyParameters, R: Runtime>(
    c: &mut Criterion,
    name: &str,
    log_n: u32,
    batch: usize,
    dir: Direction,
) where
    R::Device: Default,
{
    let n = 1usize << log_n;
    let total = batch * n;

    let device = R::Device::default();
    let exec = NttExec::<P, R>::new(&device, 0);
    let client = R::client(&device);

    let input: Vec<u32> = (0..total as u32)
        .map(|i| {
            MontyField::<P>::from_canonical(i.wrapping_mul(0x9E3779B1) % P::PRIME).raw()
        })
        .collect();
    let buf = client.create_from_slice(u32::as_bytes(&input));

    let mut group = c.benchmark_group(name);
    group.throughput(criterion::Throughput::Elements((batch * n) as u64));
    group.bench_function("run", |b| {
        b.iter(|| {
            match dir {
                Direction::Forward => exec.forward_auto(&buf, log_n, batch),
                Direction::Inverse => exec.inverse_auto(&buf, log_n, batch),
            }
            cubecl_common::reader::read_sync(client.sync()).expect("sync failed");
        });
    });
    group.finish();
}

// -- CUDA, BabyBear, forward --

fn cuda_bb_fwd_20_b1(c: &mut Criterion) {
    bench_ntt::<r0_field::BabyBearParameters, cubecl::cuda::CudaRuntime>(
        c, "cuda_bb_fwd_20_b1", 20, 1, Direction::Forward,
    );
}

fn cuda_bb_fwd_20_b32(c: &mut Criterion) {
    bench_ntt::<r0_field::BabyBearParameters, cubecl::cuda::CudaRuntime>(
        c, "cuda_bb_fwd_20_b32", 20, 32, Direction::Forward,
    );
}

fn cuda_bb_fwd_22_b1(c: &mut Criterion) {
    bench_ntt::<r0_field::BabyBearParameters, cubecl::cuda::CudaRuntime>(
        c, "cuda_bb_fwd_22_b1", 22, 1, Direction::Forward,
    );
}

fn cuda_bb_fwd_22_b32(c: &mut Criterion) {
    bench_ntt::<r0_field::BabyBearParameters, cubecl::cuda::CudaRuntime>(
        c, "cuda_bb_fwd_22_b32", 22, 32, Direction::Forward,
    );
}

// -- CUDA, BabyBear, inverse --

fn cuda_bb_inv_20_b32(c: &mut Criterion) {
    bench_ntt::<r0_field::BabyBearParameters, cubecl::cuda::CudaRuntime>(
        c, "cuda_bb_inv_20_b32", 20, 32, Direction::Inverse,
    );
}

// -- CUDA, KoalaBear, forward --

fn cuda_kb_fwd_20_b32(c: &mut Criterion) {
    bench_ntt::<r0_field::KoalaBearParameters, cubecl::cuda::CudaRuntime>(
        c, "cuda_kb_fwd_20_b32", 20, 32, Direction::Forward,
    );
}

// -- wgpu, BabyBear, forward --

fn wgpu_bb_fwd_20_b1(c: &mut Criterion) {
    bench_ntt::<r0_field::BabyBearParameters, cubecl::wgpu::WgpuRuntime>(
        c, "wgpu_bb_fwd_20_b1", 20, 1, Direction::Forward,
    );
}

fn wgpu_bb_fwd_20_b8(c: &mut Criterion) {
    bench_ntt::<r0_field::BabyBearParameters, cubecl::wgpu::WgpuRuntime>(
        c, "wgpu_bb_fwd_20_b8", 20, 8, Direction::Forward,
    );
}

fn wgpu_bb_fwd_22_b1(c: &mut Criterion) {
    bench_ntt::<r0_field::BabyBearParameters, cubecl::wgpu::WgpuRuntime>(
        c, "wgpu_bb_fwd_22_b1", 22, 1, Direction::Forward,
    );
}

fn wgpu_bb_fwd_22_b8(c: &mut Criterion) {
    bench_ntt::<r0_field::BabyBearParameters, cubecl::wgpu::WgpuRuntime>(
        c, "wgpu_bb_fwd_22_b8", 22, 8, Direction::Forward,
    );
}

// -- wgpu, BabyBear, inverse --

fn wgpu_bb_inv_20_b8(c: &mut Criterion) {
    bench_ntt::<r0_field::BabyBearParameters, cubecl::wgpu::WgpuRuntime>(
        c, "wgpu_bb_inv_20_b8", 20, 8, Direction::Inverse,
    );
}

// -- wgpu, KoalaBear, forward --

fn wgpu_kb_fwd_20_b8(c: &mut Criterion) {
    bench_ntt::<r0_field::KoalaBearParameters, cubecl::wgpu::WgpuRuntime>(
        c, "wgpu_kb_fwd_20_b8", 20, 8, Direction::Forward,
    );
}

criterion_group!(
    benches,
    cuda_bb_fwd_20_b1,
    cuda_bb_fwd_20_b32,
    cuda_bb_fwd_22_b1,
    cuda_bb_fwd_22_b32,
    cuda_bb_inv_20_b32,
    cuda_kb_fwd_20_b32,
    wgpu_bb_fwd_20_b1,
    wgpu_bb_fwd_20_b8,
    wgpu_bb_fwd_22_b1,
    wgpu_bb_fwd_22_b8,
    wgpu_bb_inv_20_b8,
    wgpu_kb_fwd_20_b8,
);
criterion_main!(benches);
