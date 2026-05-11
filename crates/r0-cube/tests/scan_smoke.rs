//! End-to-end smoke for the generic scan primitives.
//!
//! Defines a private `(u32, +)` monoid (`SumU32`), launches a single
//! workgroup that runs `block_inclusive_scan` and `block_inclusive_reduce`
//! over an input array, and checks the outputs against the host wrapping
//! prefix sum / total.

use core::marker::PhantomData;

use cubecl::prelude::*;

use r0_cube::{block_inclusive_reduce, block_inclusive_scan, Device, Monoid, Runtime};

#[derive(CubeType, Copy, Clone, Debug, Default)]
pub struct SumU32 {
    pub v: u32,
    #[cube(comptime)]
    _p: PhantomData<()>,
}

#[cube]
impl Monoid for SumU32 {
    type Repr = u32;
    const REPR_LANES: u32 = 1;

    fn identity() -> Self {
        SumU32 { v: 0u32, _p: PhantomData }
    }
    fn combine(left: Self, right: Self) -> Self {
        SumU32 { v: left.v + right.v, _p: PhantomData }
    }
    fn to_repr(value: Self) -> u32 { value.v }
    fn from_repr(repr: u32) -> Self {
        SumU32 { v: repr, _p: PhantomData }
    }
    fn alloc_scratch(#[comptime] count: u32) -> SharedMemory<u32> {
        SharedMemory::<u32>::new(comptime!(count as usize))
    }
}

#[cube(launch_unchecked)]
fn smoke_block_scan(
    input: &Array<u32>,
    output: &mut Array<u32>,
    #[comptime] log_warp: u32,
    #[comptime] log_wg: u32,
    #[comptime] num_warps: u32,
) {
    let mut scratch = SharedMemory::<u32>::new(comptime!(num_warps as usize));
    let i = ABSOLUTE_POS;
    let v = SumU32 { v: input[i], _p: PhantomData };
    let scanned = block_inclusive_scan::<SumU32>(v, &mut scratch, log_warp, log_wg);
    output[i] = scanned.v;
}

#[cube(launch_unchecked)]
fn smoke_block_reduce(
    input: &Array<u32>,
    output: &mut Array<u32>,
    #[comptime] log_warp: u32,
    #[comptime] log_wg: u32,
    #[comptime] num_warps: u32,
) {
    let mut scratch = SharedMemory::<u32>::new(comptime!(num_warps as usize));
    let i = ABSOLUTE_POS;
    let v = SumU32 { v: input[i], _p: PhantomData };
    let total = block_inclusive_reduce::<SumU32>(v, &mut scratch, log_warp, log_wg);
    output[i] = total.v;
}

fn host_prefix_sum(input: &[u32]) -> Vec<u32> {
    let mut acc = 0u32;
    input.iter().map(|&x| { acc = acc.wrapping_add(x); acc }).collect()
}

fn run_block_scan(device: &Device<Runtime>, log_warp: u32, log_wg: u32) {
    let wg_size = 1u32 << log_wg;
    let num_warps = 1u32 << (log_wg - log_warp);
    let n = wg_size as usize;

    let input: Vec<u32> = (1..=n as u32).collect();
    let expected = host_prefix_sum(&input);

    let client = device.client();
    let in_h = client.create_from_slice(u32::as_bytes(&input));
    let out_h = client.empty(n * core::mem::size_of::<u32>());

    unsafe {
        smoke_block_scan::launch_unchecked::<Runtime>(
            client, CubeCount::Static(1, 1, 1), CubeDim::new_1d(wg_size),
            ArrayArg::from_raw_parts::<u32>(&in_h, n, 1),
            ArrayArg::from_raw_parts::<u32>(&out_h, n, 1),
            log_warp, log_wg, num_warps,
        ).expect("kernel launch failed");
    }

    let bytes = client.read_one(out_h);
    let actual: Vec<u32> = u32::from_bytes(&bytes).to_vec();
    assert_eq!(actual, expected,
        "block_inclusive_scan mismatch (log_warp={log_warp}, log_wg={log_wg})");
}

fn run_block_reduce(device: &Device<Runtime>, log_warp: u32, log_wg: u32) {
    let wg_size = 1u32 << log_wg;
    let num_warps = 1u32 << (log_wg - log_warp);
    let n = wg_size as usize;

    let input: Vec<u32> = (1..=n as u32).collect();
    let total: u32 = input.iter().copied().fold(0u32, u32::wrapping_add);
    let expected = vec![total; n];

    let client = device.client();
    let in_h = client.create_from_slice(u32::as_bytes(&input));
    let out_h = client.empty(n * core::mem::size_of::<u32>());

    unsafe {
        smoke_block_reduce::launch_unchecked::<Runtime>(
            client, CubeCount::Static(1, 1, 1), CubeDim::new_1d(wg_size),
            ArrayArg::from_raw_parts::<u32>(&in_h, n, 1),
            ArrayArg::from_raw_parts::<u32>(&out_h, n, 1),
            log_warp, log_wg, num_warps,
        ).expect("kernel launch failed");
    }

    let bytes = client.read_one(out_h);
    let actual: Vec<u32> = u32::from_bytes(&bytes).to_vec();
    assert_eq!(actual, expected,
        "block_inclusive_reduce mismatch (log_warp={log_warp}, log_wg={log_wg})");
}

#[test]
fn smoke() {
    let device = Device::<Runtime>::acquire();
    let client = device.client();

    let plane_size = client.properties().hardware.plane_size_max;
    assert!(plane_size.is_power_of_two() && plane_size >= 4);
    let log_warp = plane_size.trailing_zeros();
    let max_wg = client.properties().hardware.max_units_per_cube;
    let log_wg_max = max_wg.trailing_zeros();

    // Single-warp baseline.
    run_block_scan(&device, log_warp, log_warp);
    run_block_reduce(&device, log_warp, log_warp);

    // Largest the device allows.
    let log_wg_max_safe = log_wg_max.min(2 * log_warp);
    if log_wg_max_safe > log_warp {
        run_block_scan(&device, log_warp, log_wg_max_safe);
        run_block_reduce(&device, log_warp, log_wg_max_safe);
    }
}
