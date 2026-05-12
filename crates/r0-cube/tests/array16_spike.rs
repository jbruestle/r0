//! Spike: which cubecl-0.9 primitive holds a per-thread 16-u32 working set?
//!
//! Poseidon1 width-16 wants the state to live in registers across 28 rounds.
//! With FFT-based MDS (~50 monty_muls per round vs 256 naive), the win
//! depends on the indexed access pattern unrolling cleanly. This file tries
//! the two plausible representations and runs them through a small workload
//! that touches every slot with comptime indices:
//!
//!   1. `Array::<u32>::new(16)`   — local array, lowers to `var a: array<u32, 16>`
//!                                  in WGSL (function address space).
//!   2. `Line::<u32>::empty(16)`  — cubecl's lane vector. We use it elsewhere
//!                                  for cross-lane shuffles in PairScan; here we
//!                                  ask whether it works as a pure per-thread
//!                                  register array.
//!
//! Workload: read 16 u32s, reverse them, then compute prefix sums (left to
//! right). Each step depends on the previous, so the optimiser can't elide
//! either storage. Output is checked against the host computation.

use cubecl::prelude::*;
use r0_cube::{Device, Runtime};

const W: u32 = 16;
const NUM_THREADS: u32 = 4;

// ---------------------------------------------------------------------------
// Approach A: local Array<u32>
// ---------------------------------------------------------------------------

#[cube(launch_unchecked)]
fn spike_local_array(input: &Array<u32>, output: &mut Array<u32>) {
    let tid = ABSOLUTE_POS;
    let mut state = Array::<u32>::new(comptime!(16usize));

    // Load.
    #[unroll]
    for i in 0usize..16usize {
        state[i] = input[tid * 16usize + i];
    }

    // Reverse in place. Comptime symmetric pair indexing.
    #[unroll]
    for i in 0usize..8usize {
        let lo = state[i];
        let hi = state[15usize - i];
        state[i] = hi;
        state[15usize - i] = lo;
    }

    // Inclusive prefix sum (forces serial dependency between slots).
    #[unroll]
    for i in 1usize..16usize {
        state[i] = state[i] + state[i - 1usize];
    }

    // Store.
    #[unroll]
    for i in 0usize..16usize {
        output[tid * 16usize + i] = state[i];
    }
}

// ---------------------------------------------------------------------------
// Approach B: per-thread Line<u32> of size 16
// ---------------------------------------------------------------------------

#[cube(launch_unchecked)]
fn spike_local_line(input: &Array<u32>, output: &mut Array<u32>) {
    let tid = ABSOLUTE_POS;
    let mut state = Line::<u32>::empty(comptime!(16usize));

    #[unroll]
    for i in 0..16usize {
        state[i] = input[tid * 16usize + i];
    }

    #[unroll]
    for i in 0..8usize {
        let lo = state[i];
        let hi = state[15usize - i];
        state[i] = hi;
        state[15usize - i] = lo;
    }

    #[unroll]
    for i in 1usize..16usize {
        state[i] = state[i] + state[i - 1usize];
    }

    #[unroll]
    for i in 0..16usize {
        output[tid * 16usize + i] = state[i];
    }
}

// ---------------------------------------------------------------------------
// Host reference
// ---------------------------------------------------------------------------

fn host_workload(input: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(input.len());
    for chunk in input.chunks_exact(W as usize) {
        let mut s: [u32; 16] = chunk.try_into().unwrap();
        s.reverse();
        for i in 1..16 {
            s[i] = s[i].wrapping_add(s[i - 1]);
        }
        out.extend_from_slice(&s);
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

fn make_input() -> Vec<u32> {
    (0..(NUM_THREADS * W))
        .map(|i| i.wrapping_mul(0x9E37_79B1).wrapping_add(1))
        .collect()
}

#[test]
fn array16_local_array() {
    let device = Device::<Runtime>::acquire();
    let client = device.client();

    let input = make_input();
    let n = input.len();
    let expected = host_workload(&input);

    let in_h = client.create_from_slice(u32::as_bytes(&input));
    let out_h = client.empty(n * core::mem::size_of::<u32>());

    unsafe {
        spike_local_array::launch_unchecked::<Runtime>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(NUM_THREADS),
            ArrayArg::from_raw_parts::<u32>(&in_h, n, 1),
            ArrayArg::from_raw_parts::<u32>(&out_h, n, 1),
        )
        .expect("local Array<u32> launch");
    }

    let bytes = client.read_one(out_h);
    let actual: Vec<u32> = u32::from_bytes(&bytes).to_vec();
    assert_eq!(actual, expected, "local Array<u32> mismatch");
}

#[test]
fn array16_local_line() {
    let device = Device::<Runtime>::acquire();
    let client = device.client();

    let input = make_input();
    let n = input.len();
    let expected = host_workload(&input);

    let in_h = client.create_from_slice(u32::as_bytes(&input));
    let out_h = client.empty(n * core::mem::size_of::<u32>());

    unsafe {
        spike_local_line::launch_unchecked::<Runtime>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(NUM_THREADS),
            ArrayArg::from_raw_parts::<u32>(&in_h, n, 1),
            ArrayArg::from_raw_parts::<u32>(&out_h, n, 1),
        )
        .expect("local Line<u32> launch");
    }

    let bytes = client.read_one(out_h);
    let actual: Vec<u32> = u32::from_bytes(&bytes).to_vec();
    assert_eq!(actual, expected, "local Line<u32> mismatch");
}
