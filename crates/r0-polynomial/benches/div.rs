//! `PolyDivExec::div_by_x_minus_z` benchmarks.

use criterion::{criterion_group, criterion_main, Criterion};
use cubecl::prelude::*;

use r0_cube::{Device, Runtime};
use r0_polynomial::{PairScanLayout, PolyDivExec};

fn bench_div<F: PairScanLayout>(c: &mut Criterion, name: &str, log_n: u32, batch: usize) {
    let n = 1usize << log_n;
    let degree = F::DEGREE as usize;

    let device = Device::<Runtime>::acquire();
    let exec = PolyDivExec::<F, Runtime>::new(&device, log_n, batch);
    let client = exec.client().clone();

    let buf_data: Vec<u32> = (0..batch * n * degree)
        .map(|i| (i as u32).wrapping_mul(0x9E3779B1))
        .collect();
    let zs_data: Vec<u32> = (0..batch * degree)
        .map(|i| (i as u32).wrapping_mul(0x517CC1B7).wrapping_add(1))
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

fn kb4_div_20_b1(c: &mut Criterion) {
    bench_div::<r0_field::Ext4<r0_field::KoalaBear4Parameters>>(c, "kb4_div_20_b1", 20, 1);
}
fn kb4_div_20_b32(c: &mut Criterion) {
    bench_div::<r0_field::Ext4<r0_field::KoalaBear4Parameters>>(c, "kb4_div_20_b32", 20, 32);
}

criterion_group!(benches, kb4_div_20_b1, kb4_div_20_b32);
criterion_main!(benches);
