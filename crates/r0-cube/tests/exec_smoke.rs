//! End-to-end smoke for `ScanExec` + `ScanRecipe`.
//!
//! Defines a private `SumU32Recipe` (`Monoid = SumU32` over `u32`,
//! identity row-major layout, no per-batch context), runs the executor
//! over a few `(log_n, batch)` configurations on wgpu, and checks the
//! output against the host wrapping prefix sum row-by-row.
//!
//! Exercises both the single-block fast path (`n <= wg_size`) and the
//! multi-block 3-kernel pipeline (`wg_size < n <= wg_size²`), including
//! the spine scan that bridges them.
//!
//! CPU runtime is skipped — same reason as `scan_smoke.rs`.

#![cfg(feature = "wgpu")]

use core::marker::PhantomData;

use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;

use r0_cube::{Device, Monoid, ScanExec, ScanRecipe};

// ---------------------------------------------------------------------------
// SumU32 monoid (private to this test, same shape as scan_smoke.rs).
// ---------------------------------------------------------------------------

#[derive(CubeType, Copy, Clone, Debug, Default)]
pub struct SumU32 {
    pub v: u32,
    #[cube(comptime)]
    _p: PhantomData<()>,
}

#[cube]
impl Monoid for SumU32 {
    type Repr = u32;
    fn identity() -> Self {
        SumU32 {
            v: 0u32,
            _p: PhantomData,
        }
    }
    fn combine(left: Self, right: Self) -> Self {
        SumU32 {
            v: left.v + right.v,
            _p: PhantomData,
        }
    }
    fn to_repr(value: Self) -> u32 {
        value.v
    }
    fn from_repr(repr: u32) -> Self {
        SumU32 {
            v: repr,
            _p: PhantomData,
        }
    }
}

// ---------------------------------------------------------------------------
// SumU32Recipe: identity row-major layout, no per-batch context.
// ---------------------------------------------------------------------------

pub struct SumU32Recipe;

#[cube]
impl ScanRecipe for SumU32Recipe {
    type Monoid = SumU32;

    fn load(
        _contexts: &Array<u32>,
        input: &Array<u32>,
        batch: u32,
        scan_pos: u32,
        n: u32,
        _batch_count: u32,
    ) -> SumU32 {
        SumU32 {
            v: input[(batch * n + scan_pos) as usize],
            _p: PhantomData,
        }
    }

    fn store(
        _contexts: &Array<u32>,
        output: &mut Array<u32>,
        batch: u32,
        scan_pos: u32,
        n: u32,
        _batch_count: u32,
        value: SumU32,
    ) {
        output[(batch * n + scan_pos) as usize] = value.v;
    }
}

// ---------------------------------------------------------------------------
// Driver + assertion.
// ---------------------------------------------------------------------------

fn host_prefix_sum(row: &[u32]) -> Vec<u32> {
    let mut acc = 0u32;
    row.iter()
        .map(|&x| {
            acc = acc.wrapping_add(x);
            acc
        })
        .collect()
}

fn run_case<R: Runtime>(
    exec: &ScanExec<R, SumU32Recipe>,
    log_n: u32,
    batch: usize,
    label: &str,
) {
    let n = 1usize << log_n;
    let total = batch * n;

    // Fixed pseudo-random input per row — deterministic across runs.
    let input: Vec<u32> = (0..total as u32)
        .map(|i| i.wrapping_mul(2654435761).wrapping_add(1))
        .collect();
    let mut expected = vec![0u32; total];
    for b in 0..batch {
        let row = &input[b * n..(b + 1) * n];
        let scanned = host_prefix_sum(row);
        expected[b * n..(b + 1) * n].copy_from_slice(&scanned);
    }

    let client = exec.client();
    let in_h = client.create_from_slice(u32::as_bytes(&input));
    let out_h = client.empty(total * core::mem::size_of::<u32>());
    // Recipe ignores contexts — pass a 1-u32 dummy buffer.
    let ctx_h = client.create_from_slice(u32::as_bytes(&[0u32]));

    exec.run(&ctx_h, &in_h, &out_h, log_n, batch);

    let bytes = client.read_one(out_h);
    let actual: Vec<u32> = u32::from_bytes(&bytes).to_vec();

    assert_eq!(
        actual, expected,
        "scan mismatch in case `{label}` (log_n={log_n}, batch={batch})"
    );
}

#[test]
fn smoke_sum_recipe_wgpu() {
    let device = Device::<WgpuRuntime>::acquire();
    let client = device.client();

    let plane_size = client.properties().hardware.plane_size_max;
    let max_threads = client.properties().hardware.max_units_per_cube;
    assert!(plane_size.is_power_of_two() && plane_size >= 4);
    assert!(max_threads.is_power_of_two());
    let log_wg = max_threads.trailing_zeros();

    // Construct the executor for two levels past the L=0 boundary so
    // L=1 paths actually get exercised on devices with large workgroup
    // sizes (Mac wgpu reports log_wg=10 → L=0 covers up through 2^20).
    let log_n_max = 2 * log_wg + 2;
    let max_batch = 4;
    let exec = ScanExec::<WgpuRuntime, SumU32Recipe>::new(&device, log_n_max, max_batch);

    // L=0 single-block: n == wg_size, exercises the fast path.
    run_case(&exec, log_wg, 2, "single-block, n=wg_size");

    // L=0 tiny single-block (n < wg_size): kernel uses log_n as wg-size exponent.
    if log_wg >= 4 {
        run_case(&exec, 4, 3, "single-block, n=16");
    }

    // L=0 multi-block, modest size.
    run_case(&exec, log_wg + 2, 2, "L=0 multi-block, n=4*wg_size");

    // L=0 multi-block, max size at this depth (n = wg_size²).
    run_case(&exec, 2 * log_wg, 2, "L=0 multi-block, n=wg_size^2");

    // L=1 multi-block, just over the L=0 boundary (num_blocks at level 1 = 2).
    run_case(&exec, 2 * log_wg + 1, 2, "L=1 multi-block, n=2*wg_size^2");

    // L=1 multi-block, num_blocks at level 1 = 4.
    run_case(&exec, 2 * log_wg + 2, 2, "L=1 multi-block, n=4*wg_size^2");
}
