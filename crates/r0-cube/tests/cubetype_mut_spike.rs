//! Spike: does a `#[cube] fn` accept `&mut <CubeType-struct>` and
//! propagate field mutations back to the caller?
//!
//! Needed for the Poseidon constraint subroutine, where we want
//! `ConstraintAccumulator { acc, alpha_pow }` threaded through a chain
//! of constraint subroutines via `&mut`.
//!
//! We test two shapes:
//!   1. Helper `#[cube] fn` mutates `&mut Pair` directly.
//!   2. Helper `#[cube] fn` mutates `&mut Pair` and is called from
//!      another helper, which is called from the kernel — to verify the
//!      mutation propagates through nested `&mut` borrowing.

use cubecl::prelude::*;
use r0_cube::{Device, Runtime};

#[derive(CubeType, Copy, Clone)]
pub struct Pair {
    pub x: u32,
    pub y: u32,
}

#[cube]
fn add_to_pair(p: &mut Pair, dx: u32, dy: u32) {
    p.x = p.x + dx;
    p.y = p.y + dy;
}

#[cube]
fn double_then_add(p: &mut Pair, dx: u32, dy: u32) {
    p.x = p.x * 2u32;
    p.y = p.y * 2u32;
    add_to_pair(p, dx, dy);
}

#[cube(launch_unchecked)]
fn spike_kernel(input: &Array<u32>, output: &mut Array<u32>) {
    let i = ABSOLUTE_POS;
    let mut p = Pair {
        x: input[i],
        y: input[i] + 1u32,
    };
    // After: p = (input[i] * 2 + 100, (input[i] + 1) * 2 + 200)
    double_then_add(&mut p, 100u32, 200u32);
    output[2usize * i + 0usize] = p.x;
    output[2usize * i + 1usize] = p.y;
}

#[test]
fn cubetype_mut_propagates() {
    let device = Device::<Runtime>::acquire();
    let client = device.client();

    let n = 4usize;
    let input: Vec<u32> = vec![10, 20, 30, 40];
    let in_h = client.create_from_slice(u32::as_bytes(&input));
    let out_h = client.empty(2 * n * core::mem::size_of::<u32>());

    unsafe {
        spike_kernel::launch_unchecked::<Runtime>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(n as u32),
            ArrayArg::from_raw_parts::<u32>(&in_h, n, 1),
            ArrayArg::from_raw_parts::<u32>(&out_h, 2 * n, 1),
        )
        .expect("kernel launch");
    }

    let bytes = client.read_one(out_h);
    let actual: Vec<u32> = u32::from_bytes(&bytes).to_vec();

    let mut expected = vec![];
    for &x in &input {
        expected.push(x * 2 + 100);
        expected.push((x + 1) * 2 + 200);
    }
    assert_eq!(actual, expected, "&mut Pair mutation did not propagate");
}
