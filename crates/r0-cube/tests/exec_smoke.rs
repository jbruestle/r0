//! End-to-end smoke for `ScanExec` + `ScanRecipe`.
//!
//! Exercises the single-block fast path, the L=0 multi-block pipeline,
//! and the L=1 recursive spine.

use core::marker::PhantomData;

use cubecl::prelude::*;

use r0_cube::{Device, Monoid, Runtime, ScanExec, ScanRecipe};

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
    fn identity() -> Self { SumU32 { v: 0u32, _p: PhantomData } }
    fn combine(left: Self, right: Self) -> Self {
        SumU32 { v: left.v + right.v, _p: PhantomData }
    }
    fn to_repr(value: Self) -> u32 { value.v }
    fn from_repr(repr: u32) -> Self { SumU32 { v: repr, _p: PhantomData } }
    fn alloc_scratch(#[comptime] count: u32) -> SharedMemory<u32> {
        SharedMemory::<u32>::new(comptime!(count as usize))
    }
}

pub struct SumU32Recipe;

#[cube]
impl ScanRecipe for SumU32Recipe {
    type Monoid = SumU32;

    fn load(
        _contexts: &Array<u32>, input: &Array<u32>,
        batch: u32, scan_pos: u32, n: u32, _batch_count: u32,
    ) -> SumU32 {
        SumU32 { v: input[(batch * n + scan_pos) as usize], _p: PhantomData }
    }

    fn store(
        _contexts: &Array<u32>, output: &mut Array<u32>,
        batch: u32, scan_pos: u32, n: u32, _batch_count: u32, value: SumU32,
    ) {
        output[(batch * n + scan_pos) as usize] = value.v;
    }
}

fn host_prefix_sum(row: &[u32]) -> Vec<u32> {
    let mut acc = 0u32;
    row.iter().map(|&x| { acc = acc.wrapping_add(x); acc }).collect()
}

fn run_case(exec: &ScanExec<Runtime, SumU32Recipe>, log_n: u32, batch: usize, label: &str) {
    let n = 1usize << log_n;
    let total = batch * n;

    let input: Vec<u32> = (0..total as u32)
        .map(|i| i.wrapping_mul(2654435761).wrapping_add(1))
        .collect();
    let mut expected = vec![0u32; total];
    for b in 0..batch {
        let scanned = host_prefix_sum(&input[b * n..(b + 1) * n]);
        expected[b * n..(b + 1) * n].copy_from_slice(&scanned);
    }

    let client = exec.client();
    let in_h = client.create_from_slice(u32::as_bytes(&input));
    let out_h = client.empty(total * core::mem::size_of::<u32>());
    let ctx_h = client.create_from_slice(u32::as_bytes(&[0u32]));

    exec.run(&ctx_h, &in_h, &out_h, log_n, batch);

    let bytes = client.read_one(out_h);
    let actual: Vec<u32> = u32::from_bytes(&bytes).to_vec();
    assert_eq!(actual, expected, "scan mismatch in `{label}` (log_n={log_n}, batch={batch})");
}

#[test]
fn smoke_sum_recipe() {
    let device = Device::<Runtime>::acquire();
    let client = device.client();
    assert!(client.properties().hardware.plane_size_max.is_power_of_two());
    let log_wg = client.properties().hardware.max_units_per_cube.trailing_zeros();

    let log_n_max = 2 * log_wg + 2;
    let exec = ScanExec::<Runtime, SumU32Recipe>::new(&device, log_n_max, 4);

    run_case(&exec, log_wg, 2, "single-block");
    run_case(&exec, log_wg + 2, 2, "L=0 multi-block");
    run_case(&exec, 2 * log_wg + 1, 2, "L=1 multi-block");
}
