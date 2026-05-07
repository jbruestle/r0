//! NTT benchmarks through the NttExec API.

use criterion::{criterion_group, criterion_main, Criterion};
use cubecl::prelude::*;

use r0_field::MontyParameters;
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
            r0_field::MontyField::<P>::from_canonical(i.wrapping_mul(0x9E3779B1) % P::PRIME).raw()
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

// -- CUDA benchmarks --

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;

    pub fn bb_fwd_20_b1(c: &mut Criterion) {
        bench_ntt::<r0_field::BabyBearParameters, cubecl::cuda::CudaRuntime>(
            c, "cuda_bb_fwd_20_b1", 20, 1, Direction::Forward,
        );
    }

    pub fn bb_fwd_20_b32(c: &mut Criterion) {
        bench_ntt::<r0_field::BabyBearParameters, cubecl::cuda::CudaRuntime>(
            c, "cuda_bb_fwd_20_b32", 20, 32, Direction::Forward,
        );
    }

    pub fn bb_fwd_22_b1(c: &mut Criterion) {
        bench_ntt::<r0_field::BabyBearParameters, cubecl::cuda::CudaRuntime>(
            c, "cuda_bb_fwd_22_b1", 22, 1, Direction::Forward,
        );
    }

    pub fn bb_fwd_22_b32(c: &mut Criterion) {
        bench_ntt::<r0_field::BabyBearParameters, cubecl::cuda::CudaRuntime>(
            c, "cuda_bb_fwd_22_b32", 22, 32, Direction::Forward,
        );
    }

    pub fn bb_inv_20_b32(c: &mut Criterion) {
        bench_ntt::<r0_field::BabyBearParameters, cubecl::cuda::CudaRuntime>(
            c, "cuda_bb_inv_20_b32", 20, 32, Direction::Inverse,
        );
    }

    pub fn kb_fwd_20_b32(c: &mut Criterion) {
        bench_ntt::<r0_field::KoalaBearParameters, cubecl::cuda::CudaRuntime>(
            c, "cuda_kb_fwd_20_b32", 20, 32, Direction::Forward,
        );
    }

    criterion_group!(
        benches,
        bb_fwd_20_b1,
        bb_fwd_20_b32,
        bb_fwd_22_b1,
        bb_fwd_22_b32,
        bb_inv_20_b32,
        kb_fwd_20_b32,
    );
}

// -- wgpu benchmarks --

#[cfg(feature = "wgpu")]
mod wgpu_benches {
    use super::*;

    pub fn bb_fwd_20_b1(c: &mut Criterion) {
        bench_ntt::<r0_field::BabyBearParameters, cubecl::wgpu::WgpuRuntime>(
            c, "wgpu_bb_fwd_20_b1", 20, 1, Direction::Forward,
        );
    }

    pub fn bb_fwd_20_b8(c: &mut Criterion) {
        bench_ntt::<r0_field::BabyBearParameters, cubecl::wgpu::WgpuRuntime>(
            c, "wgpu_bb_fwd_20_b8", 20, 8, Direction::Forward,
        );
    }

    pub fn bb_fwd_22_b1(c: &mut Criterion) {
        bench_ntt::<r0_field::BabyBearParameters, cubecl::wgpu::WgpuRuntime>(
            c, "wgpu_bb_fwd_22_b1", 22, 1, Direction::Forward,
        );
    }

    pub fn bb_fwd_22_b8(c: &mut Criterion) {
        bench_ntt::<r0_field::BabyBearParameters, cubecl::wgpu::WgpuRuntime>(
            c, "wgpu_bb_fwd_22_b8", 22, 8, Direction::Forward,
        );
    }

    pub fn bb_inv_20_b8(c: &mut Criterion) {
        bench_ntt::<r0_field::BabyBearParameters, cubecl::wgpu::WgpuRuntime>(
            c, "wgpu_bb_inv_20_b8", 20, 8, Direction::Inverse,
        );
    }

    pub fn kb_fwd_20_b8(c: &mut Criterion) {
        bench_ntt::<r0_field::KoalaBearParameters, cubecl::wgpu::WgpuRuntime>(
            c, "wgpu_kb_fwd_20_b8", 20, 8, Direction::Forward,
        );
    }

    criterion_group!(
        benches,
        bb_fwd_20_b1,
        bb_fwd_20_b8,
        bb_fwd_22_b1,
        bb_fwd_22_b8,
        bb_inv_20_b8,
        kb_fwd_20_b8,
    );
}

// -- Main: include whichever groups are enabled --

#[cfg(all(feature = "cuda", feature = "wgpu"))]
criterion_main!(cuda::benches, wgpu_benches::benches);

#[cfg(all(feature = "cuda", not(feature = "wgpu")))]
criterion_main!(cuda::benches);

#[cfg(all(not(feature = "cuda"), feature = "wgpu"))]
criterion_main!(wgpu_benches::benches);

#[cfg(all(not(feature = "cuda"), not(feature = "wgpu")))]
fn main() {
    eprintln!("No GPU features enabled — nothing to benchmark.");
}
