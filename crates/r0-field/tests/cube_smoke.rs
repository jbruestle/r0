//! End-to-end exercise of the `#[cube]` IR path. Launches a generic
//! element-wise `monty_mul` kernel and asserts its output matches the
//! host `monty_mul` byte-for-byte.

use cubecl::prelude::*;

use r0_cube::{Device, Runtime};
use r0_field::{monty_mul, BabyBearParameters, KoalaBearParameters, MontyField, MontyParameters};

#[cube(launch_unchecked)]
fn vec_monty_mul<P: MontyParameters>(a: &Array<u32>, b: &Array<u32>, out: &mut Array<u32>) {
    if ABSOLUTE_POS < a.len() {
        out[ABSOLUTE_POS] = monty_mul::<P>(a[ABSOLUTE_POS], b[ABSOLUTE_POS]);
    }
}

fn check_vec_mul<P: MontyParameters>(n: usize) {
    let a: Vec<u32> = (0..n)
        .map(|i| MontyField::<P>::from_canonical((i as u32).wrapping_mul(0x9E3779B1) ^ 0xDEADBEEF).raw())
        .collect();
    let b: Vec<u32> = (0..n)
        .map(|i| MontyField::<P>::from_canonical((i as u32).wrapping_mul(0x85EBCA77) ^ 0xC2B2AE3D).raw())
        .collect();
    let expected: Vec<u32> = a.iter().zip(b.iter()).map(|(&x, &y)| monty_mul::<P>(x, y)).collect();

    let device = Device::<Runtime>::acquire();
    let client = device.client();

    let a_handle = client.create_from_slice(u32::as_bytes(&a));
    let b_handle = client.create_from_slice(u32::as_bytes(&b));
    let out_handle = client.empty(n * core::mem::size_of::<u32>());

    let block_size: u32 = 64;
    let num_blocks: u32 = n.div_ceil(block_size as usize) as u32;

    unsafe {
        vec_monty_mul::launch_unchecked::<P, Runtime>(
            &client, CubeCount::Static(num_blocks, 1, 1), CubeDim::new_1d(block_size),
            ArrayArg::from_raw_parts::<u32>(&a_handle, n, 1),
            ArrayArg::from_raw_parts::<u32>(&b_handle, n, 1),
            ArrayArg::from_raw_parts::<u32>(&out_handle, n, 1),
        ).expect("kernel launch failed");
    }

    let bytes = client.read_one(out_handle);
    let actual: Vec<u32> = u32::from_bytes(&bytes).to_vec();
    assert_eq!(actual, expected);
}

#[test]
fn babybear_vec_mul() { check_vec_mul::<BabyBearParameters>(1024); }

#[test]
fn koalabear_vec_mul() { check_vec_mul::<KoalaBearParameters>(1024); }
