//! Plane- and block-level inclusive prefix scans / reduces, generic over
//! [`Monoid`].
//!
//! All `#[cube]`. Same source compiles to host, CUDA, WGSL, and the
//! cubecl CPU emulator.
//!
//! Internally these scans operate entirely on [`Monoid::Repr`] (a
//! `CubePrimitive`) and only cross back to `Self` for the algebraic
//! `combine` and at the function boundary. That sidesteps cubecl 0.9's
//! restrictions on mutating, if-else'ing, and shared-memory-indexing
//! arbitrary `CubeType` values — see the `monoid` module docs.

use cubecl::prelude::*;

use crate::monoid::Monoid;

/// Combine two `Repr` values via `M::combine`, with one round-trip
/// through `Self`. Crate-internal helper used wherever a generic
/// scan-style if-else would otherwise return a `M`; staying in `Repr`
/// keeps cubecl 0.9 happy.
#[cube]
pub(crate) fn combine_via<M: Monoid>(left: M::Repr, right: M::Repr) -> M::Repr {
    M::to_repr(M::combine(M::from_repr(left), M::from_repr(right)))
}

/// Inclusive prefix scan within a single plane (warp).
///
/// After the call, every lane holds the `combine` of all values from
/// lane 0 through itself. `log_warp` is the base-2 log of the plane
/// size — pass
/// `client.properties().hardware.plane_size_max.trailing_zeros()`.
#[cube]
pub fn plane_inclusive_scan<M: Monoid>(value: M, #[comptime] log_warp: u32) -> M {
    let lane = UNIT_POS_PLANE;
    let mut v = M::to_repr(value);
    #[unroll]
    for s in 0..log_warp {
        let off = comptime!(1u32 << s);
        let other = plane_shuffle_up(v, off);
        v = if lane >= off {
            combine_via::<M>(other, v)
        } else {
            v
        };
    }
    M::from_repr(v)
}

/// Inclusive prefix scan across an entire workgroup.
///
/// Each thread returns its own cumulative scan result. `scratch` must
/// hold at least `1 << (log_wg - log_warp)` `M::Repr` values.
///
/// Constraint: `wg_size <= warp_size²` (so warp 0 can scan the per-warp
/// totals in a single plane scan). For warp 32 that's `wg_size <= 1024`,
/// which covers every device we target.
#[cube]
pub fn block_inclusive_scan<M: Monoid>(
    value: M,
    scratch: &mut SharedMemory<M::Repr>,
    #[comptime] log_warp: u32,
    #[comptime] log_wg: u32,
) -> M {
    let warp_size = comptime!(1u32 << log_warp);
    let num_warps = comptime!(1u32 << (log_wg - log_warp));
    let lane = UNIT_POS_PLANE;
    let warp_id = UNIT_POS / warp_size;

    // Stage 1: scan within each warp; carry the result as `Repr` so the
    // remaining stages stay in CubePrimitive-space.
    let scanned_repr = M::to_repr(plane_inclusive_scan::<M>(value, log_warp));

    let result_repr = if comptime!(num_warps == 1) {
        scanned_repr
    } else {
        // Stage 2: last lane of each warp publishes its warp's total.
        if lane == warp_size - 1 {
            scratch[warp_id as usize] = scanned_repr;
        }
        sync_cube();

        // Stage 3: warp 0 scans the per-warp totals. Inactive lanes
        // (>= num_warps) feed identity. Indexing scratch[lane] is safe
        // for lane < warp_size since scratch holds num_warps <=
        // warp_size entries; we still clamp the read to avoid OOB on
        // lanes we'll discard anyway.
        if warp_id == 0 {
            let safe_idx = if lane < num_warps { lane } else { 0u32.into() };
            let loaded = scratch[safe_idx as usize];
            let v_repr = if lane < num_warps {
                loaded
            } else {
                M::to_repr(M::identity())
            };
            let scanned_warp_sums = plane_inclusive_scan::<M>(M::from_repr(v_repr), log_warp);
            if lane < num_warps {
                scratch[lane as usize] = M::to_repr(scanned_warp_sums);
            }
        }
        sync_cube();

        // Stage 4: each warp k>0 reads scratch[k-1] as its carry.
        let carry_idx = if warp_id > 0 { warp_id - 1 } else { 0u32.into() };
        let carry_repr = scratch[carry_idx as usize];
        if warp_id > 0 {
            combine_via::<M>(carry_repr, scanned_repr)
        } else {
            scanned_repr
        }
    };
    M::from_repr(result_repr)
}

/// Inclusive reduce across an entire workgroup. Returns the same total
/// to every lane.
///
/// Cheaper than [`block_inclusive_scan`] — no per-lane carry phase.
/// The block total ends up in slot 0 of `scratch` and is read by every
/// lane.
#[cube]
pub fn block_inclusive_reduce<M: Monoid>(
    value: M,
    scratch: &mut SharedMemory<M::Repr>,
    #[comptime] log_warp: u32,
    #[comptime] log_wg: u32,
) -> M {
    let warp_size = comptime!(1u32 << log_warp);
    let num_warps = comptime!(1u32 << (log_wg - log_warp));
    let lane = UNIT_POS_PLANE;
    let warp_id = UNIT_POS / warp_size;

    let scanned_repr = M::to_repr(plane_inclusive_scan::<M>(value, log_warp));

    if lane == warp_size - 1 {
        scratch[warp_id as usize] = scanned_repr;
    }
    sync_cube();

    if comptime!(num_warps > 1) {
        if warp_id == 0 {
            let safe_idx = if lane < num_warps { lane } else { 0u32.into() };
            let loaded = scratch[safe_idx as usize];
            let v_repr = if lane < num_warps {
                loaded
            } else {
                M::to_repr(M::identity())
            };
            let scanned_warp = plane_inclusive_scan::<M>(M::from_repr(v_repr), log_warp);
            if lane == num_warps - 1 {
                scratch[0] = M::to_repr(scanned_warp);
            }
        }
        sync_cube();
    }
    // Single-warp case: slot 0 already holds the warp's total
    // (lane warp_size-1 wrote it above).

    M::from_repr(scratch[0])
}
