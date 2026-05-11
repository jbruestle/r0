//! NTT benchmarks through the NttExec API.

use criterion::{criterion_group, criterion_main, Criterion};
use cubecl::prelude::*;

use r0_cube::{Device, Runtime};
use r0_field::MontyParameters;
use r0_ntt::NttExec;

#[derive(Clone, Copy)]
enum Direction { Forward, Inverse }

fn bench_ntt<P: MontyParameters>(
    c: &mut Criterion, name: &str, log_n: u32, batch: usize, dir: Direction,
) {
    let n = 1usize << log_n;
    let total = batch * n;

    let device = Device::<Runtime>::acquire();
    let exec = NttExec::<P, Runtime>::new(&device);
    let client = Runtime::client(device.inner());

    let input: Vec<u32> = (0..total as u32)
        .map(|i| r0_field::MontyField::<P>::from_canonical(i.wrapping_mul(0x9E3779B1) % P::PRIME).raw())
        .collect();
    let buf = client.create_from_slice(u32::as_bytes(&input));

    let mut group = c.benchmark_group(name);
    group.throughput(criterion::Throughput::Elements((batch * n) as u64));
    group.bench_function("run", |b| {
        b.iter(|| {
            match dir {
                Direction::Forward => exec.forward(&buf, log_n, batch),
                Direction::Inverse => exec.inverse(&buf, log_n, batch),
            }
            cubecl_common::reader::read_sync(client.sync()).expect("sync failed");
        });
    });
    group.finish();
}

fn bb_fwd_20_b1(c: &mut Criterion) {
    bench_ntt::<r0_field::BabyBearParameters>(c, "bb_fwd_20_b1", 20, 1, Direction::Forward);
}
fn bb_fwd_20_b32(c: &mut Criterion) {
    bench_ntt::<r0_field::BabyBearParameters>(c, "bb_fwd_20_b32", 20, 32, Direction::Forward);
}
fn bb_fwd_22_b1(c: &mut Criterion) {
    bench_ntt::<r0_field::BabyBearParameters>(c, "bb_fwd_22_b1", 22, 1, Direction::Forward);
}
fn bb_inv_20_b32(c: &mut Criterion) {
    bench_ntt::<r0_field::BabyBearParameters>(c, "bb_inv_20_b32", 20, 32, Direction::Inverse);
}

criterion_group!(benches, bb_fwd_20_b1, bb_fwd_20_b32, bb_fwd_22_b1, bb_inv_20_b32);
criterion_main!(benches);
