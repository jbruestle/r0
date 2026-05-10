//! `PolyDivExec::div_by_x_minus_z` benchmarks.
//!
//! Headline target: `BB^4`, `log_n = 20`, `batch = 1` — the
//! coefficients-of-a-single-polynomial-of-2^20 case that gates whether
//! the substrate is worth taking further. wgpu numbers serve as a
//! correctness-floor reference; CUDA numbers are the target metric per
//! the design doc (`< 1 ms` for `BB^4` `log_n=20` `batch=32`, with
//! `batch=1` proportionally cheaper).

use criterion::{criterion_group, criterion_main, Criterion};
use cubecl::prelude::*;

use r0_cube::Device;
use r0_polynomial::{PairScanLayout, PolyDivExec};

/// Run one benchmark configuration. Allocates buffers once and times
/// only the kernel dispatch + sync; the `div_by_x_minus_z` is in-place
/// so the input doesn't need re-uploading between iterations.
fn bench_div<F, R>(c: &mut Criterion, name: &str, log_n: u32, batch: usize)
where
    F: PairScanLayout,
    R: Runtime,
    R::Device: Default,
{
    let n = 1usize << log_n;
    let degree = F::DEGREE as usize;
    let total_buf_u32 = batch * n * degree;
    let total_zs_u32 = batch * degree;

    let device = Device::<R>::acquire();
    let exec = PolyDivExec::<F, R>::new(&device, log_n, batch);
    let client = exec.client().clone();

    // Inputs are arbitrary — kernel time is data-independent for a scan.
    let buf_data: Vec<u32> = (0..total_buf_u32 as u32)
        .map(|i| i.wrapping_mul(0x9E3779B1))
        .collect();
    let zs_data: Vec<u32> = (0..total_zs_u32 as u32)
        .map(|i| i.wrapping_mul(0x517CC1B7).wrapping_add(1))
        .collect();
    let buf = client.create_from_slice(u32::as_bytes(&buf_data));
    let zs = client.create_from_slice(u32::as_bytes(&zs_data));

    let mut group = c.benchmark_group(name);
    group.throughput(criterion::Throughput::Elements((batch * n) as u64));
    group.bench_function("run", |b| {
        b.iter(|| {
            exec.div_by_x_minus_z(&buf, &zs, log_n, batch);
            cubecl_common::reader::read_sync(client.sync()).expect("sync failed");
        });
    });
    group.finish();
}

// -- CUDA benchmarks --

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;

    pub fn bb4_div_20_b1(c: &mut Criterion) {
        bench_div::<r0_field::Ext4<r0_field::BabyBear4Parameters>, cubecl::cuda::CudaRuntime>(
            c,
            "cuda_bb4_div_20_b1",
            20,
            1,
        );
    }

    pub fn bb4_div_20_b32(c: &mut Criterion) {
        bench_div::<r0_field::Ext4<r0_field::BabyBear4Parameters>, cubecl::cuda::CudaRuntime>(
            c,
            "cuda_bb4_div_20_b32",
            20,
            32,
        );
    }

    pub fn bb_div_20_b1(c: &mut Criterion) {
        bench_div::<r0_field::BaseElem<r0_field::BabyBearParameters>, cubecl::cuda::CudaRuntime>(
            c,
            "cuda_bb_div_20_b1",
            20,
            1,
        );
    }

    criterion_group!(benches, bb4_div_20_b1, bb4_div_20_b32, bb_div_20_b1);
}

// -- wgpu benchmarks --

#[cfg(feature = "wgpu")]
mod wgpu_benches {
    use super::*;

    pub fn bb4_div_20_b1(c: &mut Criterion) {
        bench_div::<r0_field::Ext4<r0_field::BabyBear4Parameters>, cubecl::wgpu::WgpuRuntime>(
            c,
            "wgpu_bb4_div_20_b1",
            20,
            1,
        );
    }

    pub fn bb4_div_20_b8(c: &mut Criterion) {
        bench_div::<r0_field::Ext4<r0_field::BabyBear4Parameters>, cubecl::wgpu::WgpuRuntime>(
            c,
            "wgpu_bb4_div_20_b8",
            20,
            8,
        );
    }

    pub fn bb_div_20_b1(c: &mut Criterion) {
        bench_div::<r0_field::BaseElem<r0_field::BabyBearParameters>, cubecl::wgpu::WgpuRuntime>(
            c,
            "wgpu_bb_div_20_b1",
            20,
            1,
        );
    }

    criterion_group!(benches, bb4_div_20_b1, bb4_div_20_b8, bb_div_20_b1);
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
