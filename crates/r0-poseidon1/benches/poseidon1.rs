//! r0-poseidon1 throughput benchmarks.
//!
//! Three modes — pure permutation, permutation + witness write,
//! per-row constraint contribution — all at 2^18 perms (witness buffer
//! ~150 MiB). Throughput reported as permutations / second; sync via
//! `client.sync()` so each iteration waits for kernel completion.

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use cubecl::prelude::*;

use r0_cube::{Device, Runtime};
use r0_field::{ext4_from_raws, KoalaBear4Parameters};
use r0_poseidon1::{
    poseidon1_kb16_constraint, poseidon1_kb16_permute, poseidon1_kb16_permute_with_witness,
    ConstraintAccumulator, N_WITNESS_SBOXES,
};

const WG: u32 = 64;

// ---------------------------------------------------------------------------
// Launchable wrappers — one thread = one Poseidon, contiguous-per-thread I/O.
// ---------------------------------------------------------------------------

#[cube(launch_unchecked)]
fn k_permute(input: &Array<u32>, output: &mut Array<u32>) {
    let tid = ABSOLUTE_POS;
    let mut state = Array::<u32>::new(comptime!(16usize));
    #[unroll]
    for i in 0u32..16u32 {
        state[comptime!(i as usize)] = input[tid * 16usize + comptime!(i as usize)];
    }
    poseidon1_kb16_permute(&mut state);
    #[unroll]
    for i in 0u32..16u32 {
        output[tid * 16usize + comptime!(i as usize)] = state[comptime!(i as usize)];
    }
}

#[cube(launch_unchecked)]
fn k_witgen(
    input: &Array<u32>,
    output: &mut Array<u32>,
    witness: &mut Array<u32>,
    stride: u32,
) {
    let tid = ABSOLUTE_POS;
    let mut state = Array::<u32>::new(comptime!(16usize));
    #[unroll]
    for i in 0u32..16u32 {
        state[comptime!(i as usize)] = input[tid * 16usize + comptime!(i as usize)];
    }
    poseidon1_kb16_permute_with_witness(&mut state, witness, 0u32, tid as u32, stride);
    #[unroll]
    for i in 0u32..16u32 {
        output[tid * 16usize + comptime!(i as usize)] = state[comptime!(i as usize)];
    }
}

#[cube(launch_unchecked)]
fn k_constraint(
    input_states: &Array<u32>,
    witness: &Array<u32>,
    output: &mut Array<u32>,
    alpha_c0: u32,
    alpha_c1: u32,
    alpha_c2: u32,
    alpha_c3: u32,
    stride: u32,
) {
    let tid = ABSOLUTE_POS;
    let mut input_state = Array::<u32>::new(comptime!(16usize));
    #[unroll]
    for i in 0u32..16u32 {
        input_state[comptime!(i as usize)] = input_states[tid * 16usize + comptime!(i as usize)];
    }

    let alpha = ext4_from_raws::<KoalaBear4Parameters>(alpha_c0, alpha_c1, alpha_c2, alpha_c3);
    let zero = ext4_from_raws::<KoalaBear4Parameters>(0u32, 0u32, 0u32, 0u32);
    let one = ext4_from_raws::<KoalaBear4Parameters>(1u32, 0u32, 0u32, 0u32);
    let cs = ConstraintAccumulator { alpha, acc: zero, alpha_pow: one };

    let cs = poseidon1_kb16_constraint(&input_state, witness, 0u32, tid as u32, stride, cs);

    output[tid * 8usize + 0usize] = cs.acc.c0;
    output[tid * 8usize + 1usize] = cs.acc.c1;
    output[tid * 8usize + 2usize] = cs.acc.c2;
    output[tid * 8usize + 3usize] = cs.acc.c3;
    output[tid * 8usize + 4usize] = cs.alpha_pow.c0;
    output[tid * 8usize + 5usize] = cs.alpha_pow.c1;
    output[tid * 8usize + 6usize] = cs.alpha_pow.c2;
    output[tid * 8usize + 7usize] = cs.alpha_pow.c3;
}

// ---------------------------------------------------------------------------
// Benches
// ---------------------------------------------------------------------------

fn input_buffer(n_perms: u32) -> Vec<u32> {
    (0..(n_perms * 16))
        .map(|i| i.wrapping_mul(0x9E37_79B1).wrapping_add(1) % 0x7f00_0000)
        .collect()
}

fn bench_permute(c: &mut Criterion, n_perms: u32) {
    let device = Device::<Runtime>::acquire();
    let client = device.client().clone();

    let in_data = input_buffer(n_perms);
    let in_h = client.create_from_slice(u32::as_bytes(&in_data));
    let out_h = client.empty(in_data.len() * core::mem::size_of::<u32>());

    let n_blocks = n_perms.div_ceil(WG);

    let mut group = c.benchmark_group(format!("permute_2^{}", n_perms.trailing_zeros()));
    group.throughput(Throughput::Elements(n_perms as u64));
    group.bench_function("run", |b| {
        b.iter(|| {
            unsafe {
                k_permute::launch_unchecked::<Runtime>(
                    &client,
                    CubeCount::Static(n_blocks, 1, 1),
                    CubeDim::new_1d(WG),
                    ArrayArg::from_raw_parts::<u32>(&in_h, in_data.len(), 1),
                    ArrayArg::from_raw_parts::<u32>(&out_h, in_data.len(), 1),
                )
                .expect("k_permute launch");
            }
            cubecl_common::reader::read_sync(client.sync()).expect("sync failed");
        });
    });
    group.finish();
}

fn bench_witgen(c: &mut Criterion, n_perms: u32) {
    let device = Device::<Runtime>::acquire();
    let client = device.client().clone();

    let in_data = input_buffer(n_perms);
    let in_h = client.create_from_slice(u32::as_bytes(&in_data));
    let out_h = client.empty(in_data.len() * core::mem::size_of::<u32>());
    let stride = n_perms;
    let wit_len = N_WITNESS_SBOXES * stride as usize;
    let wit_h = client.empty(wit_len * core::mem::size_of::<u32>());

    let n_blocks = n_perms.div_ceil(WG);

    let mut group = c.benchmark_group(format!("witgen_2^{}", n_perms.trailing_zeros()));
    group.throughput(Throughput::Elements(n_perms as u64));
    group.bench_function("run", |b| {
        b.iter(|| {
            unsafe {
                k_witgen::launch_unchecked::<Runtime>(
                    &client,
                    CubeCount::Static(n_blocks, 1, 1),
                    CubeDim::new_1d(WG),
                    ArrayArg::from_raw_parts::<u32>(&in_h, in_data.len(), 1),
                    ArrayArg::from_raw_parts::<u32>(&out_h, in_data.len(), 1),
                    ArrayArg::from_raw_parts::<u32>(&wit_h, wit_len, 1),
                    ScalarArg::new(stride),
                )
                .expect("k_witgen launch");
            }
            cubecl_common::reader::read_sync(client.sync()).expect("sync failed");
        });
    });
    group.finish();
}

fn bench_constraint(c: &mut Criterion, n_perms: u32) {
    let device = Device::<Runtime>::acquire();
    let client = device.client().clone();

    let in_data = input_buffer(n_perms);
    let in_h = client.create_from_slice(u32::as_bytes(&in_data));
    let stride = n_perms;
    let wit_len = N_WITNESS_SBOXES * stride as usize;
    // Witness contents don't affect timing — fill with zeros.
    let wit_h = client.empty(wit_len * core::mem::size_of::<u32>());
    let out_len = (n_perms as usize) * 8;
    let out_h = client.empty(out_len * core::mem::size_of::<u32>());

    let n_blocks = n_perms.div_ceil(WG);

    let mut group = c.benchmark_group(format!("constraint_2^{}", n_perms.trailing_zeros()));
    group.throughput(Throughput::Elements(n_perms as u64));
    group.bench_function("run", |b| {
        b.iter(|| {
            unsafe {
                k_constraint::launch_unchecked::<Runtime>(
                    &client,
                    CubeCount::Static(n_blocks, 1, 1),
                    CubeDim::new_1d(WG),
                    ArrayArg::from_raw_parts::<u32>(&in_h, in_data.len(), 1),
                    ArrayArg::from_raw_parts::<u32>(&wit_h, wit_len, 1),
                    ArrayArg::from_raw_parts::<u32>(&out_h, out_len, 1),
                    ScalarArg::new(0x12345678u32),
                    ScalarArg::new(0x9ABCDEF0u32),
                    ScalarArg::new(0x0FEDCBA9u32),
                    ScalarArg::new(0x87654321u32),
                    ScalarArg::new(stride),
                )
                .expect("k_constraint launch");
            }
            cubecl_common::reader::read_sync(client.sync()).expect("sync failed");
        });
    });
    group.finish();
}

const N_PERMS: u32 = 1 << 18;

fn permute(c: &mut Criterion) { bench_permute(c, N_PERMS); }
fn witgen(c: &mut Criterion) { bench_witgen(c, N_PERMS); }
fn constraint(c: &mut Criterion) { bench_constraint(c, N_PERMS); }

criterion_group!(benches, permute, witgen, constraint);
criterion_main!(benches);
