//! `ScanExec`: device-resident driver for [`ScanRecipe`]-based scans.
//!
//! Runs the standard map → scan → unmap pipeline against a recipe. Two
//! cases:
//!
//! - **Single block** (`n <= wg_size`): one workgroup per polynomial.
//!   `Recipe::load` → [`block_inclusive_scan`] → `Recipe::store`.
//!
//! - **Multi-block** (`n > wg_size`, `n <= wg_size²`): three kernels.
//!   1. `k0_reduce` — each block reduces, lane 0 writes its block total
//!      to the spine.
//!   2. `spine_scan` — one workgroup per polynomial scans the
//!      per-block totals (constraint: `num_blocks <= wg_size`, i.e.
//!      `n <= wg_size²`).
//!   3. `k0_apply` — each block re-loads, re-scans, combines its spine
//!      carry, projects, stores.
//!
//! Recursive spine for `n > wg_size²` is the next milestone; not
//! implemented yet. With wgpu's typical 256-thread cap that's
//! `n <= 65536`; with CUDA's 1024-thread cap it's `n <= 1M` (the
//! polynomial-division headline target).

use core::marker::PhantomData;

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::device::Device;
use crate::monoid::Monoid;
use crate::recipe::ScanRecipe;
use crate::scan::{block_inclusive_reduce, block_inclusive_scan, combine_via};

const U32_BYTES: u64 = 4;

/// Device-resident scan executor for one recipe.
///
/// Construct one per `(device, recipe, log_n_max, max_batch)` and reuse
/// it. The spine buffer slices into `device.scratch()` — no per-executor
/// allocation. Pre-bakes `log_warp` and `log_wg` from the device's
/// hardware properties so each `run` call only needs to compute the
/// per-call `num_blocks` and dispatch.
pub struct ScanExec<R: Runtime, Recipe: ScanRecipe> {
    client: ComputeClient<R>,
    spine: Handle,
    log_warp: u32,
    log_wg: u32,
    num_warps: u32,
    log_n_max: u32,
    max_batch: u32,
    _r: PhantomData<Recipe>,
}

impl<R: Runtime, Recipe: ScanRecipe> ScanExec<R, Recipe> {
    /// Construct an executor for `(device, Recipe)`, capable of scans
    /// with `log_n` up to `log_n_max` and batches up to `max_batch`.
    ///
    /// Constraints (panics if violated):
    /// - `log_n_max <= 2 * log2(max_threads_per_wg)` — required for the
    ///   spine to fit one workgroup at the worst case.
    /// - `device.scratch_bytes() >= max_batch * num_blocks_max *
    ///   sizeof::<M::Repr>()` — spine has to fit in the device's shared
    ///   scratch buffer.
    pub fn new(device: &Device<R>, log_n_max: u32, max_batch: usize) -> Self {
        assert!(log_n_max >= 1, "log_n_max must be >= 1");

        let client = device.client().clone();
        let plane_size = client.properties().hardware.plane_size_max;
        assert!(
            plane_size.is_power_of_two() && plane_size >= 2,
            "expected a power-of-two plane size >= 2, got {plane_size}"
        );
        let log_warp = plane_size.trailing_zeros();

        let max_threads = client.properties().hardware.max_units_per_cube;
        assert!(
            max_threads.is_power_of_two(),
            "expected a power-of-two max_units_per_cube, got {max_threads}"
        );
        let log_wg = max_threads.trailing_zeros();
        assert!(
            log_n_max <= 2 * log_wg,
            "log_n_max={log_n_max} requires log_wg >= {} but device offers log_wg <= {log_wg}",
            log_n_max.div_ceil(2)
        );

        let num_warps = 1u32 << (log_wg - log_warp);
        let max_n = 1usize << log_n_max;
        let wg_size = 1usize << log_wg;
        let max_num_blocks = max_n.div_ceil(wg_size).max(1);
        let max_spine_elems = max_batch * max_num_blocks;
        let repr_bytes =
            <<Recipe::Monoid as Monoid>::Repr as CubePrimitive>::type_size();
        let spine_bytes = (max_spine_elems * repr_bytes) as u64;
        assert!(
            device.scratch_bytes() as u64 >= spine_bytes,
            "ScanExec spine needs {spine_bytes} bytes; device scratch is {} bytes",
            device.scratch_bytes()
        );
        let spine = device.scratch().clone();

        Self {
            client,
            spine,
            log_warp,
            log_wg,
            num_warps,
            log_n_max,
            max_batch: max_batch as u32,
            _r: PhantomData,
        }
    }

    /// Underlying cubecl client. Use to allocate input/output buffers
    /// and read results back.
    pub fn client(&self) -> &ComputeClient<R> {
        &self.client
    }

    /// Run the scan over `batch` polynomials of length `2^log_n`.
    ///
    /// Buffer expectations (caller's responsibility — recipes own
    /// layout, so this driver only sizes element counts via byte length):
    /// - `input`, `output`: each holds `batch * 2^log_n * R::WORDS` u32s
    ///   in whatever layout the recipe expects.
    /// - `contexts`: holds whatever per-batch-row data the recipe reads
    ///   (e.g. `batch * F::DEGREE` u32s for `DivByXMinusZ<F>`); pass a
    ///   minimal dummy buffer for context-free recipes.
    ///
    /// `output` may alias `input` (in-place).
    pub fn run(
        &self,
        contexts: &Handle,
        input: &Handle,
        output: &Handle,
        log_n: u32,
        batch: usize,
    ) {
        assert!(
            log_n >= 1 && log_n <= self.log_n_max,
            "log_n={log_n} out of [1, {}]",
            self.log_n_max
        );
        assert!(
            (batch as u32) <= self.max_batch,
            "batch={batch} > max_batch={}",
            self.max_batch
        );

        let n = 1u32 << log_n;
        let wg_size = 1u32 << self.log_wg;
        let batch_count = batch as u32;

        if n <= wg_size {
            self.launch_single_block(contexts, input, output, log_n, batch_count);
        } else {
            let num_blocks = n / wg_size;
            assert!(
                num_blocks <= wg_size,
                "num_blocks={num_blocks} > wg_size={wg_size}; recursive spine not yet implemented"
            );
            self.launch_k0_reduce(contexts, input, log_n, batch_count, num_blocks);
            self.launch_spine_scan(batch_count, num_blocks);
            self.launch_k0_apply(contexts, input, output, log_n, batch_count, num_blocks);
        }
    }

    fn launch_single_block(
        &self,
        contexts: &Handle,
        input: &Handle,
        output: &Handle,
        log_n: u32,
        batch_count: u32,
    ) {
        let n = 1u32 << log_n;
        let wg_size = 1u32 << self.log_wg;
        // For n < wg_size we'd waste threads — kernel uses log_n as the
        // workgroup-size exponent in that case so threads == n.
        let kernel_log_wg = log_n.min(self.log_wg);
        let kernel_wg_size = 1u32 << kernel_log_wg;
        let _ = wg_size; // (informational, may differ from kernel_wg_size for small n)

        let in_count = (input.size() / U32_BYTES) as usize;
        let out_count = (output.size() / U32_BYTES) as usize;
        let ctx_count = (contexts.size() / U32_BYTES) as usize;
        let num_warps = 1u32 << (kernel_log_wg.saturating_sub(self.log_warp));

        unsafe {
            k_single_block::launch_unchecked::<Recipe, R>(
                &self.client,
                CubeCount::Static(1, batch_count, 1),
                CubeDim::new_1d(kernel_wg_size),
                ArrayArg::from_raw_parts::<u32>(contexts, ctx_count, 1),
                ArrayArg::from_raw_parts::<u32>(input, in_count, 1),
                ArrayArg::from_raw_parts::<u32>(output, out_count, 1),
                log_n,
                self.log_warp,
                kernel_log_wg,
                num_warps,
                batch_count,
                n,
            )
            .expect("k_single_block launch failed");
        }
    }

    fn launch_k0_reduce(
        &self,
        contexts: &Handle,
        input: &Handle,
        log_n: u32,
        batch_count: u32,
        num_blocks: u32,
    ) {
        let in_count = (input.size() / U32_BYTES) as usize;
        let ctx_count = (contexts.size() / U32_BYTES) as usize;
        let spine_elems = self.spine_elem_count();
        let n = 1u32 << log_n;

        unsafe {
            k0_reduce::launch_unchecked::<Recipe, R>(
                &self.client,
                CubeCount::Static(num_blocks, batch_count, 1),
                CubeDim::new_1d(1u32 << self.log_wg),
                ArrayArg::from_raw_parts::<u32>(contexts, ctx_count, 1),
                ArrayArg::from_raw_parts::<u32>(input, in_count, 1),
                ArrayArg::from_raw_parts::<<Recipe::Monoid as Monoid>::Repr>(
                    &self.spine,
                    spine_elems,
                    1,
                ),
                log_n,
                self.log_warp,
                self.log_wg,
                self.num_warps,
                num_blocks,
                batch_count,
                n,
            )
            .expect("k0_reduce launch failed");
        }
    }

    fn launch_spine_scan(&self, batch_count: u32, num_blocks: u32) {
        let spine_elems = self.spine_elem_count();
        unsafe {
            spine_scan::launch_unchecked::<Recipe::Monoid, R>(
                &self.client,
                CubeCount::Static(1, batch_count, 1),
                CubeDim::new_1d(1u32 << self.log_wg),
                ArrayArg::from_raw_parts::<<Recipe::Monoid as Monoid>::Repr>(
                    &self.spine,
                    spine_elems,
                    1,
                ),
                self.log_warp,
                self.log_wg,
                self.num_warps,
                num_blocks,
            )
            .expect("spine_scan launch failed");
        }
    }

    fn launch_k0_apply(
        &self,
        contexts: &Handle,
        input: &Handle,
        output: &Handle,
        log_n: u32,
        batch_count: u32,
        num_blocks: u32,
    ) {
        let in_count = (input.size() / U32_BYTES) as usize;
        let out_count = (output.size() / U32_BYTES) as usize;
        let ctx_count = (contexts.size() / U32_BYTES) as usize;
        let spine_elems = self.spine_elem_count();
        let n = 1u32 << log_n;

        unsafe {
            k0_apply::launch_unchecked::<Recipe, R>(
                &self.client,
                CubeCount::Static(num_blocks, batch_count, 1),
                CubeDim::new_1d(1u32 << self.log_wg),
                ArrayArg::from_raw_parts::<u32>(contexts, ctx_count, 1),
                ArrayArg::from_raw_parts::<u32>(input, in_count, 1),
                ArrayArg::from_raw_parts::<<Recipe::Monoid as Monoid>::Repr>(
                    &self.spine,
                    spine_elems,
                    1,
                ),
                ArrayArg::from_raw_parts::<u32>(output, out_count, 1),
                log_n,
                self.log_warp,
                self.log_wg,
                self.num_warps,
                num_blocks,
                batch_count,
                n,
            )
            .expect("k0_apply launch failed");
        }
    }

    fn spine_elem_count(&self) -> usize {
        let repr_bytes =
            <<Recipe::Monoid as Monoid>::Repr as CubePrimitive>::type_size();
        (self.spine.size() / repr_bytes as u64) as usize
    }
}

// -- Kernels ------------------------------------------------------------

/// Single-block fast path: `n == wg_size`. One workgroup per polynomial
/// does the whole scan with no spine.
#[cube(launch_unchecked)]
fn k_single_block<Recipe: ScanRecipe>(
    contexts: &Array<u32>,
    input: &Array<u32>,
    output: &mut Array<u32>,
    #[comptime] log_n: u32,
    #[comptime] log_warp: u32,
    #[comptime] log_wg: u32,
    #[comptime] num_warps: u32,
    #[comptime] batch_count: u32,
    #[comptime] n: u32,
) {
    let _ = log_n;
    let batch = CUBE_POS_Y;
    let scan_pos = UNIT_POS;

    let mut scratch = SharedMemory::<<Recipe::Monoid as Monoid>::Repr>::new(comptime!(
        num_warps as usize
    ));

    let v = Recipe::load(contexts, input, batch, scan_pos, n, batch_count);
    let scanned =
        block_inclusive_scan::<Recipe::Monoid>(v, &mut scratch, log_warp, log_wg);
    Recipe::store(contexts, output, batch, scan_pos, n, batch_count, scanned);
}

/// Multi-block stage 1: each block reduces, lane 0 writes its block
/// total to `spine[batch * num_blocks + block_id]`.
#[cube(launch_unchecked)]
fn k0_reduce<Recipe: ScanRecipe>(
    contexts: &Array<u32>,
    input: &Array<u32>,
    spine: &mut Array<<Recipe::Monoid as Monoid>::Repr>,
    #[comptime] log_n: u32,
    #[comptime] log_warp: u32,
    #[comptime] log_wg: u32,
    #[comptime] num_warps: u32,
    #[comptime] num_blocks: u32,
    #[comptime] batch_count: u32,
    #[comptime] n: u32,
) {
    let _ = log_n;
    let batch = CUBE_POS_Y;
    let block_id = CUBE_POS_X;
    let tid = UNIT_POS;
    let scan_pos = block_id * comptime!(1u32 << log_wg) + tid;

    let mut scratch = SharedMemory::<<Recipe::Monoid as Monoid>::Repr>::new(comptime!(
        num_warps as usize
    ));

    let v = Recipe::load(contexts, input, batch, scan_pos, n, batch_count);
    let total =
        block_inclusive_reduce::<Recipe::Monoid>(v, &mut scratch, log_warp, log_wg);

    if tid == 0u32 {
        spine[(batch * num_blocks + block_id) as usize] =
            <Recipe::Monoid as Monoid>::to_repr(total);
    }
}

/// Multi-block stage 2: per batch row, scan the per-block totals.
/// Generic over the monoid (no recipe dependency at this stage).
#[cube(launch_unchecked)]
fn spine_scan<M: Monoid>(
    spine: &mut Array<M::Repr>,
    #[comptime] log_warp: u32,
    #[comptime] log_wg: u32,
    #[comptime] num_warps: u32,
    #[comptime] num_blocks: u32,
) {
    let batch = CUBE_POS_Y;
    let tid = UNIT_POS;

    let mut scratch =
        SharedMemory::<M::Repr>::new(comptime!(num_warps as usize));

    let safe_idx = if tid < num_blocks {
        tid
    } else {
        0u32.into()
    };
    let loaded_repr = spine[(batch * num_blocks + safe_idx) as usize];
    let v_repr = if tid < num_blocks {
        loaded_repr
    } else {
        M::to_repr(M::identity())
    };
    let scanned =
        block_inclusive_scan::<M>(M::from_repr(v_repr), &mut scratch, log_warp, log_wg);
    if tid < num_blocks {
        spine[(batch * num_blocks + tid) as usize] = M::to_repr(scanned);
    }
}

/// Multi-block stage 3: per block, re-load and re-scan, combine the
/// spine carry, project to output via `Recipe::store`.
#[cube(launch_unchecked)]
fn k0_apply<Recipe: ScanRecipe>(
    contexts: &Array<u32>,
    input: &Array<u32>,
    spine: &Array<<Recipe::Monoid as Monoid>::Repr>,
    output: &mut Array<u32>,
    #[comptime] log_n: u32,
    #[comptime] log_warp: u32,
    #[comptime] log_wg: u32,
    #[comptime] num_warps: u32,
    #[comptime] num_blocks: u32,
    #[comptime] batch_count: u32,
    #[comptime] n: u32,
) {
    let _ = log_n;
    let batch = CUBE_POS_Y;
    let block_id = CUBE_POS_X;
    let tid = UNIT_POS;
    let scan_pos = block_id * comptime!(1u32 << log_wg) + tid;

    let mut scratch = SharedMemory::<<Recipe::Monoid as Monoid>::Repr>::new(comptime!(
        num_warps as usize
    ));

    let v = Recipe::load(contexts, input, batch, scan_pos, n, batch_count);
    let scanned =
        block_inclusive_scan::<Recipe::Monoid>(v, &mut scratch, log_warp, log_wg);
    let scanned_repr = <Recipe::Monoid as Monoid>::to_repr(scanned);

    let carry_idx = if block_id > 0 {
        block_id - 1
    } else {
        0u32.into()
    };
    let carry_repr = spine[(batch * num_blocks + carry_idx) as usize];
    let final_repr = if block_id > 0 {
        combine_via::<Recipe::Monoid>(carry_repr, scanned_repr)
    } else {
        scanned_repr
    };
    let final_v = <Recipe::Monoid as Monoid>::from_repr(final_repr);
    Recipe::store(contexts, output, batch, scan_pos, n, batch_count, final_v);
}
