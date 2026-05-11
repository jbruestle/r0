//! `ScanExec`: device-resident driver for [`ScanRecipe`]-based scans.
//!
//! Runs the standard map → scan → unmap pipeline against a recipe, with
//! a recursive spine to support transforms larger than the device's
//! `wg_size²` single-spine cap.
//!
//! # Pipeline
//!
//! For `n <= wg_size`: a single `k_single_block` kernel does the whole
//! scan in one workgroup per polynomial.
//!
//! For `n > wg_size`: choose `L = ceil(log_n / log_wg) - 2` spine
//! recursion levels (`L = 0` if `wg_size < n <= wg_size²`, `L = 1` if
//! `wg_size² < n <= wg_size³`, etc.) and dispatch `2(L+1) + 1` kernels:
//!
//! 1. `k0_reduce<Recipe>` — recipe-aware. Each block reduces, lane 0
//!    writes its total to `spine[0][block]`.
//! 2. `k_reduce_spine<M>` × L — each upper-spine cell is the reduction
//!    of `wg_size` lower-spine cells.
//! 3. `spine_top_scan<M>` — one workgroup per polynomial does the
//!    top-level inclusive scan in place.
//! 4. `k_apply_spine<M>` × L (in reverse) — each group re-scans
//!    `wg_size` lower-spine cells, combines with the upper-spine carry,
//!    writes back. After this stage `spine[0][k]` holds the inclusive
//!    prefix of the original input through block `k`.
//! 5. `k0_apply<Recipe>` — recipe-aware. Re-loads input, re-scans
//!    within block, combines with `spine[0][block-1]` as carry, projects
//!    via `Recipe::store`.
//!
//! Recipe-aware kernels are monomorphized per `Recipe`; spine kernels
//! are monomorphized per `Recipe::Monoid` (and shared across all spine
//! levels — only the comptime sizes differ at launch time).
//!
//! # Spine layout
//!
//! Each spine level slices into `device.scratch()` at a fixed byte
//! offset computed at construction time, sized for
//! `max_batch * max_num_blocks_at_level * sizeof(M::Repr)`. Smaller
//! per-call `(log_n, batch)` use only the leftmost portion of each
//! slice — addressing is `spine[batch * num_blocks_l + block_id]`,
//! always within the construction-time bound.
//!
//! # Limits and constraints
//!
//! - `wg_size` must be a power of two. Workgroup size is fixed at the
//!   device's `max_units_per_cube` (largest the device allows).
//! - `log_n_max` is bounded only by scratch budget and grid-dim limits;
//!   a future revision will sub-batch when `num_blocks_0` exceeds the
//!   device's `max_cube_count.0`.

use core::marker::PhantomData;

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::device::Device;
use crate::monoid::Monoid;
use crate::recipe::ScanRecipe;
use crate::scan::{block_inclusive_reduce, block_inclusive_scan, combine_via};

const U32_BYTES: u64 = 4;

/// Spine recursion depth needed to scan a polynomial of size `2^log_n`
/// on a device with workgroup-size exponent `log_wg`. `0` means a
/// single-pass spine (level-0 only); `k` means `k` recursive reductions
/// before the top-level scan fits in one workgroup.
fn spine_levels_needed(log_n: u32, log_wg: u32) -> u32 {
    log_n.div_ceil(log_wg).saturating_sub(2)
}

/// Number of blocks at spine level `level`: `n / wg_size^(level+1)`,
/// floored at 1.
fn num_blocks_at_level(log_n: u32, log_wg: u32, level: u32) -> u32 {
    let log = log_n.saturating_sub(log_wg * (level + 1));
    if log == 0 {
        1
    } else {
        1u32 << log
    }
}

/// Device-resident scan executor for one recipe.
pub struct ScanExec<R: Runtime, Recipe: ScanRecipe> {
    client: ComputeClient<R>,
    /// Per-level spine slices into `device.scratch()`. `spines[k]` holds
    /// up to `max_batch * num_blocks_at_level(log_n_max, log_wg, k)`
    /// `Repr` values.
    spines: Vec<Handle>,
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
    /// Pre-slices the device's shared scratch into per-spine-level
    /// sub-handles. Panics if the scratch is too small for the worst-case
    /// (`log_n_max`, `max_batch`) spine bytes.
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

        let num_warps = 1u32 << (log_wg - log_warp);
        let max_levels = spine_levels_needed(log_n_max, log_wg);

        // `Repr::type_size()` returns bytes per *lane* (4 for u32 / Line<u32>);
        // multi-lane Reprs need it scaled by REPR_LANES.
        let lane_bytes = <<Recipe::Monoid as Monoid>::Repr as CubePrimitive>::type_size() as u64;
        let lanes = <Recipe::Monoid as Monoid>::REPR_LANES as u64;
        let repr_bytes = lane_bytes * lanes;
        let mut spines = Vec::with_capacity(max_levels as usize + 1);
        let mut byte_offset = 0u64;
        for level in 0..=max_levels {
            let nb = num_blocks_at_level(log_n_max, log_wg, level);
            let level_bytes = max_batch as u64 * nb as u64 * repr_bytes;
            spines.push(device.scratch().clone().offset_start(byte_offset));
            byte_offset += level_bytes;
        }
        assert!(
            byte_offset <= device.scratch_bytes() as u64,
            "ScanExec spine needs {byte_offset} bytes; device scratch is {} bytes",
            device.scratch_bytes()
        );

        Self {
            client,
            spines,
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
    /// Buffer expectations (recipe-owned layout, this driver only sizes
    /// element counts via byte length):
    /// - `input`, `output`: each holds the recipe's u32 layout for
    ///   `batch × 2^log_n` logical elements.
    /// - `contexts`: per-batch-row data the recipe reads (e.g.
    ///   `batch × F::DEGREE` u32s for `DivByXMinusZ<F>`); pass a 1-u32
    ///   dummy buffer for context-free recipes.
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
            return;
        }

        let levels = spine_levels_needed(log_n, self.log_wg);
        debug_assert!(levels < self.spines.len() as u32);

        // Phase 1: reduce up.
        let nb_0 = num_blocks_at_level(log_n, self.log_wg, 0);
        self.launch_k0_reduce(contexts, input, log_n, batch_count, nb_0);
        for level in 1..=levels {
            let nb_lower = num_blocks_at_level(log_n, self.log_wg, level - 1);
            let nb_upper = num_blocks_at_level(log_n, self.log_wg, level);
            self.launch_k_reduce_spine(level, batch_count, nb_lower, nb_upper);
        }

        // Phase 2: top-level scan.
        let nb_top = num_blocks_at_level(log_n, self.log_wg, levels);
        self.launch_spine_top_scan(levels, batch_count, nb_top);

        // Phase 3: apply down.
        for level in (1..=levels).rev() {
            let nb_lower = num_blocks_at_level(log_n, self.log_wg, level - 1);
            let nb_upper = num_blocks_at_level(log_n, self.log_wg, level);
            self.launch_k_apply_spine(level, batch_count, nb_lower, nb_upper);
        }
        self.launch_k0_apply(contexts, input, output, log_n, batch_count, nb_0);
    }

    fn spine_elem_count(&self, level: u32) -> usize {
        let lane_bytes =
            <<Recipe::Monoid as Monoid>::Repr as CubePrimitive>::type_size() as u64;
        let lanes = <Recipe::Monoid as Monoid>::REPR_LANES as u64;
        (self.spines[level as usize].size() / (lane_bytes * lanes)) as usize
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
        let kernel_log_wg = log_n.min(self.log_wg);
        let kernel_wg_size = 1u32 << kernel_log_wg;
        // Cap log_warp to the actual workgroup size — if n < warp_size,
        // we launch fewer threads than a full warp and the plane scan
        // must not shuffle beyond the live lanes.
        let kernel_log_warp = self.log_warp.min(kernel_log_wg);
        let num_warps = 1u32 << (kernel_log_wg.saturating_sub(kernel_log_warp));

        let in_count = (input.size() / U32_BYTES) as usize;
        let out_count = (output.size() / U32_BYTES) as usize;
        let ctx_count = (contexts.size() / U32_BYTES) as usize;

        unsafe {
            k_single_block::launch_unchecked::<Recipe, R>(
                &self.client,
                CubeCount::Static(1, batch_count, 1),
                CubeDim::new_1d(kernel_wg_size),
                ArrayArg::from_raw_parts::<u32>(contexts, ctx_count, 1),
                ArrayArg::from_raw_parts::<u32>(input, in_count, 1),
                ArrayArg::from_raw_parts::<u32>(output, out_count, 1),
                kernel_log_warp,
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
        nb_0: u32,
    ) {
        let in_count = (input.size() / U32_BYTES) as usize;
        let ctx_count = (contexts.size() / U32_BYTES) as usize;
        let spine_elems = self.spine_elem_count(0);
        let lanes = <Recipe::Monoid as Monoid>::REPR_LANES as usize;
        let n = 1u32 << log_n;

        unsafe {
            k0_reduce::launch_unchecked::<Recipe, R>(
                &self.client,
                CubeCount::Static(nb_0, batch_count, 1),
                CubeDim::new_1d(1u32 << self.log_wg),
                ArrayArg::from_raw_parts::<u32>(contexts, ctx_count, 1),
                ArrayArg::from_raw_parts::<u32>(input, in_count, 1),
                ArrayArg::from_raw_parts::<<Recipe::Monoid as Monoid>::Repr>(
                    &self.spines[0],
                    spine_elems,
                    lanes,
                ),
                self.log_warp,
                self.log_wg,
                self.num_warps,
                nb_0,
                batch_count,
                n,
            )
            .expect("k0_reduce launch failed");
        }
    }

    fn launch_k_reduce_spine(
        &self,
        level: u32,
        batch_count: u32,
        nb_lower: u32,
        nb_upper: u32,
    ) {
        let spine_lower = &self.spines[(level - 1) as usize];
        let spine_upper = &self.spines[level as usize];
        let lower_count = self.spine_elem_count(level - 1);
        let upper_count = self.spine_elem_count(level);
        let lanes = <Recipe::Monoid as Monoid>::REPR_LANES as usize;

        unsafe {
            k_reduce_spine::launch_unchecked::<Recipe::Monoid, R>(
                &self.client,
                CubeCount::Static(nb_upper, batch_count, 1),
                CubeDim::new_1d(1u32 << self.log_wg),
                ArrayArg::from_raw_parts::<<Recipe::Monoid as Monoid>::Repr>(
                    spine_lower,
                    lower_count,
                    lanes,
                ),
                ArrayArg::from_raw_parts::<<Recipe::Monoid as Monoid>::Repr>(
                    spine_upper,
                    upper_count,
                    lanes,
                ),
                self.log_warp,
                self.log_wg,
                self.num_warps,
                nb_lower,
                nb_upper,
            )
            .expect("k_reduce_spine launch failed");
        }
    }

    fn launch_spine_top_scan(&self, level: u32, batch_count: u32, nb_top: u32) {
        let spine_top = &self.spines[level as usize];
        let count = self.spine_elem_count(level);
        let lanes = <Recipe::Monoid as Monoid>::REPR_LANES as usize;

        unsafe {
            spine_top_scan::launch_unchecked::<Recipe::Monoid, R>(
                &self.client,
                CubeCount::Static(1, batch_count, 1),
                CubeDim::new_1d(1u32 << self.log_wg),
                ArrayArg::from_raw_parts::<<Recipe::Monoid as Monoid>::Repr>(
                    spine_top, count, lanes,
                ),
                self.log_warp,
                self.log_wg,
                self.num_warps,
                nb_top,
            )
            .expect("spine_top_scan launch failed");
        }
    }

    fn launch_k_apply_spine(
        &self,
        level: u32,
        batch_count: u32,
        nb_lower: u32,
        nb_upper: u32,
    ) {
        let spine_lower = &self.spines[(level - 1) as usize];
        let spine_upper = &self.spines[level as usize];
        let lower_count = self.spine_elem_count(level - 1);
        let upper_count = self.spine_elem_count(level);
        let lanes = <Recipe::Monoid as Monoid>::REPR_LANES as usize;

        // One workgroup per upper-level slot (each carries `wg_size`
        // lower-level slots).
        unsafe {
            k_apply_spine::launch_unchecked::<Recipe::Monoid, R>(
                &self.client,
                CubeCount::Static(nb_upper, batch_count, 1),
                CubeDim::new_1d(1u32 << self.log_wg),
                ArrayArg::from_raw_parts::<<Recipe::Monoid as Monoid>::Repr>(
                    spine_lower,
                    lower_count,
                    lanes,
                ),
                ArrayArg::from_raw_parts::<<Recipe::Monoid as Monoid>::Repr>(
                    spine_upper,
                    upper_count,
                    lanes,
                ),
                self.log_warp,
                self.log_wg,
                self.num_warps,
                nb_lower,
                nb_upper,
            )
            .expect("k_apply_spine launch failed");
        }
    }

    fn launch_k0_apply(
        &self,
        contexts: &Handle,
        input: &Handle,
        output: &Handle,
        log_n: u32,
        batch_count: u32,
        nb_0: u32,
    ) {
        let in_count = (input.size() / U32_BYTES) as usize;
        let out_count = (output.size() / U32_BYTES) as usize;
        let ctx_count = (contexts.size() / U32_BYTES) as usize;
        let spine_elems = self.spine_elem_count(0);
        let lanes = <Recipe::Monoid as Monoid>::REPR_LANES as usize;
        let n = 1u32 << log_n;

        unsafe {
            k0_apply::launch_unchecked::<Recipe, R>(
                &self.client,
                CubeCount::Static(nb_0, batch_count, 1),
                CubeDim::new_1d(1u32 << self.log_wg),
                ArrayArg::from_raw_parts::<u32>(contexts, ctx_count, 1),
                ArrayArg::from_raw_parts::<u32>(input, in_count, 1),
                ArrayArg::from_raw_parts::<<Recipe::Monoid as Monoid>::Repr>(
                    &self.spines[0],
                    spine_elems,
                    lanes,
                ),
                ArrayArg::from_raw_parts::<u32>(output, out_count, 1),
                self.log_warp,
                self.log_wg,
                self.num_warps,
                nb_0,
                batch_count,
                n,
            )
            .expect("k0_apply launch failed");
        }
    }
}

// -- Kernels ------------------------------------------------------------

/// Single-block fast path: `n <= wg_size`. One workgroup per polynomial
/// does the whole scan with no spine.
#[cube(launch_unchecked)]
fn k_single_block<Recipe: ScanRecipe>(
    contexts: &Array<u32>,
    input: &Array<u32>,
    output: &mut Array<u32>,
    #[comptime] log_warp: u32,
    #[comptime] log_wg: u32,
    #[comptime] num_warps: u32,
    #[comptime] batch_count: u32,
    #[comptime] n: u32,
) {
    let batch = CUBE_POS_Y;
    let scan_pos = UNIT_POS;

    let mut scratch = <Recipe::Monoid as Monoid>::alloc_scratch(num_warps);

    let v = Recipe::load(contexts, input, batch, scan_pos, n, batch_count);
    let scanned = block_inclusive_scan::<Recipe::Monoid>(v, &mut scratch, log_warp, log_wg);
    Recipe::store(contexts, output, batch, scan_pos, n, batch_count, scanned);
}

/// Multi-block stage 1 (recipe-aware): each level-0 block reduces, lane
/// 0 writes its block total to `spine[batch * num_blocks + block_id]`.
#[cube(launch_unchecked)]
fn k0_reduce<Recipe: ScanRecipe>(
    contexts: &Array<u32>,
    input: &Array<u32>,
    spine: &mut Array<<Recipe::Monoid as Monoid>::Repr>,
    #[comptime] log_warp: u32,
    #[comptime] log_wg: u32,
    #[comptime] num_warps: u32,
    #[comptime] num_blocks: u32,
    #[comptime] batch_count: u32,
    #[comptime] n: u32,
) {
    let batch = CUBE_POS_Y;
    let block_id = CUBE_POS_X;
    let tid = UNIT_POS;
    let scan_pos = block_id * comptime!(1u32 << log_wg) + tid;

    let mut scratch = <Recipe::Monoid as Monoid>::alloc_scratch(num_warps);

    let v = Recipe::load(contexts, input, batch, scan_pos, n, batch_count);
    let total = block_inclusive_reduce::<Recipe::Monoid>(v, &mut scratch, log_warp, log_wg);

    if tid == 0u32 {
        spine[(batch * num_blocks + block_id) as usize] =
            <Recipe::Monoid as Monoid>::to_repr(total);
    }
}

/// Spine reduce: each upper-level cell becomes the reduction of
/// `wg_size` consecutive lower-level cells.
#[cube(launch_unchecked)]
fn k_reduce_spine<M: Monoid>(
    spine_lower: &Array<M::Repr>,
    spine_upper: &mut Array<M::Repr>,
    #[comptime] log_warp: u32,
    #[comptime] log_wg: u32,
    #[comptime] num_warps: u32,
    #[comptime] nb_lower: u32,
    #[comptime] nb_upper: u32,
) {
    let batch = CUBE_POS_Y;
    let block_upper = CUBE_POS_X;
    let tid = UNIT_POS;
    let pos_lower = block_upper * comptime!(1u32 << log_wg) + tid;

    let mut scratch = M::alloc_scratch(num_warps);

    let safe_idx = if pos_lower < nb_lower {
        pos_lower
    } else {
        0u32.into()
    };
    let loaded_repr = spine_lower[(batch * nb_lower + safe_idx) as usize];
    let v_repr = if pos_lower < nb_lower {
        loaded_repr
    } else {
        M::to_repr(M::identity())
    };

    let total = block_inclusive_reduce::<M>(M::from_repr(v_repr), &mut scratch, log_warp, log_wg);

    if tid == 0u32 {
        spine_upper[(batch * nb_upper + block_upper) as usize] = M::to_repr(total);
    }
}

/// Top-level spine scan: in-place inclusive scan within one workgroup
/// per batch row. Constraint at the call site: `num_blocks <= wg_size`.
#[cube(launch_unchecked)]
fn spine_top_scan<M: Monoid>(
    spine: &mut Array<M::Repr>,
    #[comptime] log_warp: u32,
    #[comptime] log_wg: u32,
    #[comptime] num_warps: u32,
    #[comptime] num_blocks: u32,
) {
    let batch = CUBE_POS_Y;
    let tid = UNIT_POS;
    let mut scratch = M::alloc_scratch(num_warps);

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

/// Spine apply: re-scans `wg_size` lower-level cells, combines with the
/// upper-level carry, writes back. After this the lower spine holds the
/// inclusive prefix through each lower-level cell.
#[cube(launch_unchecked)]
fn k_apply_spine<M: Monoid>(
    spine_lower: &mut Array<M::Repr>,
    spine_upper: &Array<M::Repr>,
    #[comptime] log_warp: u32,
    #[comptime] log_wg: u32,
    #[comptime] num_warps: u32,
    #[comptime] nb_lower: u32,
    #[comptime] nb_upper: u32,
) {
    let batch = CUBE_POS_Y;
    let group = CUBE_POS_X; // one workgroup per upper-level slot
    let tid = UNIT_POS;
    let pos_lower = group * comptime!(1u32 << log_wg) + tid;

    let mut scratch = M::alloc_scratch(num_warps);

    let safe_idx = if pos_lower < nb_lower {
        pos_lower
    } else {
        0u32.into()
    };
    let loaded_repr = spine_lower[(batch * nb_lower + safe_idx) as usize];
    let v_repr = if pos_lower < nb_lower {
        loaded_repr
    } else {
        M::to_repr(M::identity())
    };

    let scanned =
        block_inclusive_scan::<M>(M::from_repr(v_repr), &mut scratch, log_warp, log_wg);
    let scanned_repr = M::to_repr(scanned);

    let carry_idx = if group > 0 {
        group - 1
    } else {
        0u32.into()
    };
    let carry_repr = spine_upper[(batch * nb_upper + carry_idx) as usize];
    let final_repr = if group > 0 {
        combine_via::<M>(carry_repr, scanned_repr)
    } else {
        scanned_repr
    };

    if pos_lower < nb_lower {
        spine_lower[(batch * nb_lower + pos_lower) as usize] = final_repr;
    }
}

/// Multi-block stage 5 (recipe-aware): each level-0 block re-loads
/// input via `Recipe::load`, re-scans, combines with the spine carry,
/// projects via `Recipe::store`.
#[cube(launch_unchecked)]
fn k0_apply<Recipe: ScanRecipe>(
    contexts: &Array<u32>,
    input: &Array<u32>,
    spine: &Array<<Recipe::Monoid as Monoid>::Repr>,
    output: &mut Array<u32>,
    #[comptime] log_warp: u32,
    #[comptime] log_wg: u32,
    #[comptime] num_warps: u32,
    #[comptime] num_blocks: u32,
    #[comptime] batch_count: u32,
    #[comptime] n: u32,
) {
    let batch = CUBE_POS_Y;
    let block_id = CUBE_POS_X;
    let tid = UNIT_POS;
    let scan_pos = block_id * comptime!(1u32 << log_wg) + tid;

    let mut scratch = <Recipe::Monoid as Monoid>::alloc_scratch(num_warps);

    let v = Recipe::load(contexts, input, batch, scan_pos, n, batch_count);
    let scanned = block_inclusive_scan::<Recipe::Monoid>(v, &mut scratch, log_warp, log_wg);
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
