//! `PairScan<F>` — the monoid for division-by-`(x − z)` style scans.
//!
//! Synthetic division `Q ← z·Q + c` is the running product of the
//! degree-1 affine matrices `M_c = [[z, c]; [0, 1]]`. Encoded as a pair
//! `(p, a)` representing `[[p, a]; [0, 1]]` the combine is
//!
//! ```text
//! (p_L, a_L) ⊕ (p_R, a_R) = (p_L · p_R,  p_R · a_L + a_R)
//! ```
//!
//! associative, non-commutative, with identity `(1, 0)`.
//!
//! # Why a layout trait
//!
//! cubecl 0.9's `Line<P>` does not encode its lane count in the Rust
//! type — the lane count is attached at IR-construction time. So
//! `PairScan<F>::Repr` is `Line<u32>` for every `F`, and the per-field
//! shape (2 lanes for base, 8 for degree-4, 16 for padded degree-5)
//! has to come from somewhere. We could macro-expand a separate
//! `Monoid` impl per concrete `F`, but the `combine` body is
//! identical across them — duplicating it is ugly. Instead the
//! per-field knowledge lives in [`PairScanLayout`] (one small impl
//! per concrete `F` in [`r0-field`]: pack / unpack / alloc_scratch),
//! while the `Monoid` impl is a single generic blanket that delegates
//! the layout-specific bits to `F` and keeps `combine` generic.
//!
//! Padding: `BB^5`'s `PairScan` packs 10 real `u32`s into a
//! `Line<u32>` of 16 lanes (`LANES` is power-of-two only on every
//! GPU backend). The 6 padding lanes hold zero and never affect
//! `combine`. Spine cost stays trivial.

use cubecl::prelude::*;

use r0_cube::Monoid;
use r0_field::{
    base_elem_from_raw, ext4_from_raws, ext5_from_raws, BabyBear4Parameters, BabyBear5Parameters,
    BabyBearParameters, BaseElem, Ext4, Ext5, ExtField, KoalaBear4Parameters, KoalaBearParameters,
};

/// The pair-scan monoid value `(p, a)` representing the affine matrix
/// `[[p, a]; [0, 1]]`. Lifting `c` against scalar `z` is `(z, c)`.
#[derive(CubeType, Copy, Clone)]
pub struct PairScan<F: ExtField> {
    pub p: F,
    pub a: F,
}

/// Construction helper. The cubecl 0.9 macro can't infer `F` for the
/// expanded `PairScan { p, a }` literal inside `#[cube] impl` blocks,
/// and turbofish through nested generics (`PairScan::<Ext4<P>> { … }`)
/// trips the Rust parser. A free generic `#[cube] fn` sidesteps both.
#[cube]
pub fn pair<F: ExtField>(p: F, a: F) -> PairScan<F> {
    PairScan::<F> { p, a }
}

/// Per-field layout for [`PairScan`]: lane count, pack / unpack between
/// `(p, a)` and a fixed-size `Line<u32>`, and shared-memory allocation
/// honoring the lane count.
///
/// One small impl per concrete `F`. The generic `Monoid` blanket below
/// delegates to these methods so the algebraic [`combine`](Monoid::combine)
/// body is written once.
#[cube]
pub trait PairScanLayout: ExtField {
    /// Number of `u32` lanes in the `Line<u32>` `Repr`. Always a power
    /// of two; equals `2 · DEGREE` rounded up (16 for `BB^5` so 10
    /// real words pad to power-of-two-friendly width).
    const LANES: u32;

    /// Pack `(p, a)` into a single `Line<u32>` with `LANES` lanes.
    /// Padding lanes (`BB^5` only) are written as zero by `Line::empty`.
    fn pack(value: PairScan<Self>) -> Line<u32>;

    /// Inverse of [`pack`](Self::pack).
    fn unpack(repr: Line<u32>) -> PairScan<Self>;

    /// Allocate `count` `Line<u32>` slots in shared memory with the
    /// right line size — `Line<u32>` doesn't statically encode its
    /// lane count, so this can't be done generically.
    fn alloc_scratch(#[comptime] count: u32) -> SharedMemory<Line<u32>>;
}

// ---------------------------------------------------------------------------
// Generic Monoid blanket: combine + identity stay in one place.
// ---------------------------------------------------------------------------

#[cube]
impl<F: PairScanLayout> Monoid for PairScan<F> {
    type Repr = Line<u32>;
    const REPR_LANES: u32 = F::LANES;

    fn identity() -> Self {
        pair::<F>(F::one(), F::zero())
    }

    fn combine(left: Self, right: Self) -> Self {
        // (p_L · p_R,  p_R · a_L + a_R)
        pair::<F>(
            F::mul(left.p, right.p),
            F::add(F::mul(right.p, left.a), right.a),
        )
    }

    fn to_repr(value: Self) -> Line<u32> {
        F::pack(value)
    }

    fn from_repr(repr: Line<u32>) -> Self {
        F::unpack(repr)
    }

    fn alloc_scratch(#[comptime] count: u32) -> SharedMemory<Line<u32>> {
        F::alloc_scratch(count)
    }
}

// ---------------------------------------------------------------------------
// Per-field layouts. Each is a tight pack/unpack over the concrete field.
// ---------------------------------------------------------------------------

#[cube]
impl PairScanLayout for BaseElem<BabyBearParameters> {
    const LANES: u32 = 2;

    fn pack(value: PairScan<Self>) -> Line<u32> {
        let mut line = Line::<u32>::empty(comptime!(2usize));
        line[0] = value.p.raw;
        line[1] = value.a.raw;
        line
    }

    fn unpack(repr: Line<u32>) -> PairScan<Self> {
        pair::<BaseElem<BabyBearParameters>>(
            base_elem_from_raw::<BabyBearParameters>(repr[0]),
            base_elem_from_raw::<BabyBearParameters>(repr[1]),
        )
    }

    fn alloc_scratch(#[comptime] count: u32) -> SharedMemory<Line<u32>> {
        SharedMemory::<u32>::new_lined(comptime!(count as usize), 2usize)
    }
}

#[cube]
impl PairScanLayout for BaseElem<KoalaBearParameters> {
    const LANES: u32 = 2;

    fn pack(value: PairScan<Self>) -> Line<u32> {
        let mut line = Line::<u32>::empty(comptime!(2usize));
        line[0] = value.p.raw;
        line[1] = value.a.raw;
        line
    }

    fn unpack(repr: Line<u32>) -> PairScan<Self> {
        pair::<BaseElem<KoalaBearParameters>>(
            base_elem_from_raw::<KoalaBearParameters>(repr[0]),
            base_elem_from_raw::<KoalaBearParameters>(repr[1]),
        )
    }

    fn alloc_scratch(#[comptime] count: u32) -> SharedMemory<Line<u32>> {
        SharedMemory::<u32>::new_lined(comptime!(count as usize), 2usize)
    }
}

#[cube]
impl PairScanLayout for Ext4<BabyBear4Parameters> {
    const LANES: u32 = 8;

    fn pack(value: PairScan<Self>) -> Line<u32> {
        let mut line = Line::<u32>::empty(comptime!(8usize));
        line[0] = value.p.c0;
        line[1] = value.p.c1;
        line[2] = value.p.c2;
        line[3] = value.p.c3;
        line[4] = value.a.c0;
        line[5] = value.a.c1;
        line[6] = value.a.c2;
        line[7] = value.a.c3;
        line
    }

    fn unpack(repr: Line<u32>) -> PairScan<Self> {
        pair::<Ext4<BabyBear4Parameters>>(
            ext4_from_raws::<BabyBear4Parameters>(repr[0], repr[1], repr[2], repr[3]),
            ext4_from_raws::<BabyBear4Parameters>(repr[4], repr[5], repr[6], repr[7]),
        )
    }

    fn alloc_scratch(#[comptime] count: u32) -> SharedMemory<Line<u32>> {
        SharedMemory::<u32>::new_lined(comptime!(count as usize), 8usize)
    }
}

#[cube]
impl PairScanLayout for Ext4<KoalaBear4Parameters> {
    const LANES: u32 = 8;

    fn pack(value: PairScan<Self>) -> Line<u32> {
        let mut line = Line::<u32>::empty(comptime!(8usize));
        line[0] = value.p.c0;
        line[1] = value.p.c1;
        line[2] = value.p.c2;
        line[3] = value.p.c3;
        line[4] = value.a.c0;
        line[5] = value.a.c1;
        line[6] = value.a.c2;
        line[7] = value.a.c3;
        line
    }

    fn unpack(repr: Line<u32>) -> PairScan<Self> {
        pair::<Ext4<KoalaBear4Parameters>>(
            ext4_from_raws::<KoalaBear4Parameters>(repr[0], repr[1], repr[2], repr[3]),
            ext4_from_raws::<KoalaBear4Parameters>(repr[4], repr[5], repr[6], repr[7]),
        )
    }

    fn alloc_scratch(#[comptime] count: u32) -> SharedMemory<Line<u32>> {
        SharedMemory::<u32>::new_lined(comptime!(count as usize), 8usize)
    }
}

#[cube]
impl PairScanLayout for Ext5<BabyBear5Parameters> {
    // 10 real u32s; padded to 16 so backends with power-of-two-only line
    // sizes accept it. Padding lanes hold zero (Line::empty default) and
    // never participate in combine.
    const LANES: u32 = 16;

    fn pack(value: PairScan<Self>) -> Line<u32> {
        let mut line = Line::<u32>::empty(comptime!(16usize));
        line[0] = value.p.c0;
        line[1] = value.p.c1;
        line[2] = value.p.c2;
        line[3] = value.p.c3;
        line[4] = value.p.c4;
        line[5] = value.a.c0;
        line[6] = value.a.c1;
        line[7] = value.a.c2;
        line[8] = value.a.c3;
        line[9] = value.a.c4;
        line
    }

    fn unpack(repr: Line<u32>) -> PairScan<Self> {
        pair::<Ext5<BabyBear5Parameters>>(
            ext5_from_raws::<BabyBear5Parameters>(repr[0], repr[1], repr[2], repr[3], repr[4]),
            ext5_from_raws::<BabyBear5Parameters>(repr[5], repr[6], repr[7], repr[8], repr[9]),
        )
    }

    fn alloc_scratch(#[comptime] count: u32) -> SharedMemory<Line<u32>> {
        SharedMemory::<u32>::new_lined(comptime!(count as usize), 16usize)
    }
}
