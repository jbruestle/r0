//! Cube `poseidon1_kb16_permute` end-to-end against the host serial
//! reference. Two checks:
//!   1. The Plonky3 [0..15] oracle vector.
//!   2. Random inputs across many threads.

use cubecl::prelude::*;

use r0_cube::{Device, Runtime};
use r0_field::KoalaBear;
use r0_poseidon1::{host_permute, poseidon1_kb16_permute};

/// Thin launchable wrapper: each thread loads 16 raw u32s from `input`,
/// runs the permutation in place on a local Array<u32>, writes 16 raw
/// u32s back to `output`. Layout is contiguous-per-thread:
/// thread `t` reads/writes `[t*16, t*16 + 16)`.
#[cube(launch_unchecked)]
fn perm_kernel(input: &Array<u32>, output: &mut Array<u32>, #[comptime] n_threads: u32) {
    let tid = ABSOLUTE_POS;
    if tid < comptime!(n_threads as usize) {
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
}

fn run_perm(client: &cubecl::prelude::ComputeClient<Runtime>, raws: &[u32]) -> Vec<u32> {
    assert_eq!(raws.len() % 16, 0, "input must be 16-aligned");
    let n_threads = (raws.len() / 16) as u32;

    let in_h = client.create_from_slice(u32::as_bytes(raws));
    let out_h = client.empty(raws.len() * core::mem::size_of::<u32>());

    unsafe {
        perm_kernel::launch_unchecked::<Runtime>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(n_threads),
            ArrayArg::from_raw_parts::<u32>(&in_h, raws.len(), 1),
            ArrayArg::from_raw_parts::<u32>(&out_h, raws.len(), 1),
            n_threads,
        )
        .expect("perm_kernel launch failed");
    }

    let bytes = client.read_one(out_h);
    u32::from_bytes(&bytes).to_vec()
}

fn pack_raw(state: &[KoalaBear; 16]) -> [u32; 16] {
    core::array::from_fn(|i| state[i].raw())
}

fn unpack_raw(raws: &[u32]) -> [KoalaBear; 16] {
    core::array::from_fn(|i| KoalaBear::from_raw(raws[i]))
}

fn assert_state_eq(actual: &[KoalaBear; 16], expected: &[KoalaBear; 16], label: &str) {
    let actual_canon: [u32; 16] = core::array::from_fn(|i| actual[i].to_canonical());
    let expected_canon: [u32; 16] = core::array::from_fn(|i| expected[i].to_canonical());
    assert_eq!(actual_canon, expected_canon, "{label}");
}

#[test]
fn cube_permute_zero_to_fifteen() {
    let device = Device::<Runtime>::acquire();
    let client = device.client();

    let in_state: [KoalaBear; 16] =
        core::array::from_fn(|i| KoalaBear::from_canonical(i as u32));
    let in_raws = pack_raw(&in_state);

    let out_raws = run_perm(client, &in_raws);
    assert_eq!(out_raws.len(), 16);
    let actual = unpack_raw(&out_raws[..16]);

    let mut expected = in_state;
    host_permute(&mut expected);

    assert_state_eq(
        &actual,
        &expected,
        "cube poseidon1_kb16_permute([0..15]) disagrees with host_permute",
    );
}

#[test]
fn cube_permute_random_batch() {
    let device = Device::<Runtime>::acquire();
    let client = device.client();

    const N: usize = 32;
    // Build N independent inputs.
    let mut all_in_raws: Vec<u32> = Vec::with_capacity(N * 16);
    let mut all_in_states: Vec<[KoalaBear; 16]> = Vec::with_capacity(N);
    for t in 0..N as u32 {
        let s: [KoalaBear; 16] = core::array::from_fn(|i| {
            let v = t.wrapping_mul(0x9E37_79B1).wrapping_add((i as u32).wrapping_mul(0xC2B2_AE3D));
            KoalaBear::from_canonical(v)
        });
        all_in_raws.extend_from_slice(&pack_raw(&s));
        all_in_states.push(s);
    }

    let out_raws = run_perm(client, &all_in_raws);
    assert_eq!(out_raws.len(), N * 16);

    for t in 0..N {
        let actual = unpack_raw(&out_raws[t * 16..(t + 1) * 16]);
        let mut expected = all_in_states[t];
        host_permute(&mut expected);
        assert_state_eq(
            &actual,
            &expected,
            &format!("cube vs host disagreement at thread {t}"),
        );
    }
}
