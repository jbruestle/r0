//! `DivByXMinusZ<F>` — `ScanRecipe` for synthetic division by `(x − z)`.

use core::marker::PhantomData;

use cubecl::prelude::*;

use r0_cube::ScanRecipe;
use r0_field::ExtField;

use crate::pair_scan::{pair, PairScan, PairScanLayout};

/// Recipe carrying no runtime state — only a phantom for `F`. Each
/// `PolyDivExec<F>` instantiates `ScanExec<R, DivByXMinusZ<F>>` once.
pub struct DivByXMinusZ<F: ExtField> {
    _f: PhantomData<F>,
}

#[cube]
impl<F: PairScanLayout> ScanRecipe for DivByXMinusZ<F> {
    type Monoid = PairScan<F>;

    /// Read coefficient `a_{n-1-scan_pos}` (descending degree order),
    /// pull per-batch `z` from `contexts`, lift to monoid `(z, a)`.
    fn load(
        contexts: &Array<u32>,
        input: &Array<u32>,
        batch: u32,
        scan_pos: u32,
        n: u32,
        batch_count: u32,
    ) -> PairScan<F> {
        // Per-polynomial slice in `input`. Each polynomial is `n × DEGREE`
        // u32s in transposed layout (handled by `F::load`).
        let base = batch * n * F::DEGREE;
        let coeff = F::load(input, base, n - 1u32 - scan_pos, n);

        // Per-batch `z`: `batch_count` extension elements in transposed
        // layout, component `c` of row `b` at offset `c · batch_count + b`.
        let z = F::load(contexts, 0u32, batch, batch_count);

        pair::<F>(z, coeff)
    }

    /// Project the scanned `(p, a)` back: write `a` to the same flipped
    /// position so the natural lowest-degree-first output convention
    /// holds (quotient at [0..n−1], remainder at [n−1]).
    fn store(
        _contexts: &Array<u32>,
        output: &mut Array<u32>,
        batch: u32,
        scan_pos: u32,
        n: u32,
        _batch_count: u32,
        value: PairScan<F>,
    ) {
        let base = batch * n * F::DEGREE;
        F::store(output, base, n - 1u32 - scan_pos, n, value.a);
    }
}
