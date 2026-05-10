//! `ScanRecipe`: the per-application interface to [`crate::exec::ScanExec`].
//!
//! A recipe says how to read an input element, lift it into a [`Monoid`]
//! value, and project a scanned monoid back to an output element. It
//! owns the index transformation (lets `DivByXMinusZ` read in
//! descending degree order so the Horner-style scan goes the right way)
//! and the per-batch-row context interpretation (lets `DivByXMinusZ`
//! pull a per-row `z` field element from the contexts buffer).
//!
//! `ScanExec` then runs the standard map → scan → unmap pipeline against
//! the recipe — a per-block reduce that writes each block's total to a
//! spine, a single-pass spine scan, and a per-block apply that re-loads,
//! re-scans, combines the spine carry, and stores. That structure is
//! recipe-agnostic; the only recipe-aware kernels are the two outer
//! ones.
//!
//! Two design notes:
//!
//! 1. **Context as `&Array<u32>`, recipe-interpreted layout.** An earlier
//!    sketch had `type Context: CubeType` baked into the trait, but the
//!    cubecl 0.9 issues with multi-word `CubeType` plumbing (the same
//!    ones [`Monoid`] works around with `Repr`) would have surfaced
//!    again here. Passing `contexts: &Array<u32>` and letting the
//!    recipe pull what it needs (typically via `ExtField::load` for
//!    field-shaped contexts) sidesteps that. For recipes that need no
//!    per-batch context (sum, product), the buffer is a one-u32 dummy
//!    that the recipe's `load` ignores.
//!
//! 2. **No `lift` / `project` separation from `load` / `store`.** The
//!    recipe's `load` lifts as part of the same call (read u32s,
//!    construct monoid value); the recipe's `store` projects (extract
//!    the relevant fields, write u32s). Fusing them means the recipe
//!    owns layout end-to-end and `ScanExec` never sees raw element
//!    types.

use cubecl::prelude::*;

use crate::monoid::Monoid;

/// The per-application interface that drives [`crate::exec::ScanExec`].
///
/// The recipe is the only thing in the pipeline that touches the input
/// and output element types and their layout — `ScanExec` itself only
/// knows about [`Self::Monoid`].
#[cube]
pub trait ScanRecipe: 'static + Send + Sync {
    /// The monoid the scan operates on. `ScanExec` instantiates the
    /// generic [`crate::block_inclusive_scan`] / [`crate::block_inclusive_reduce`]
    /// against this type.
    type Monoid: Monoid;

    /// Load element at `scan_pos` of batch row `batch` (out of
    /// `batch_count`) from a polynomial of length `n`, lift into a
    /// monoid value. `contexts` carries any per-batch-row data the
    /// recipe needs (e.g. the `z` for division by `x - z`); recipes
    /// with no context ignore the argument.
    ///
    /// Recipes are free to apply any index transformation here — for
    /// example `DivByXMinusZ` reads `arr[n - 1 - scan_pos]` so the
    /// inclusive prefix scan walks the polynomial in descending degree
    /// order, matching the Horner-style recurrence.
    fn load(
        contexts: &Array<u32>,
        input: &Array<u32>,
        batch: u32,
        scan_pos: u32,
        n: u32,
        batch_count: u32,
    ) -> Self::Monoid;

    /// Project a scanned monoid value back to the output buffer at
    /// scan position `scan_pos` of batch row `batch`. Inverse of
    /// [`load`](Self::load) up to the algebra; `DivByXMinusZ` writes to
    /// `arr[n - 1 - scan_pos]` to undo the input flip.
    fn store(
        contexts: &Array<u32>,
        output: &mut Array<u32>,
        batch: u32,
        scan_pos: u32,
        n: u32,
        batch_count: u32,
        value: Self::Monoid,
    );
}
