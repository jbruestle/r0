//! Algebraic interface for the generic scan primitives.
//!
//! A [`Monoid`] is the bare minimum a value needs to flow through
//! [`crate::plane_inclusive_scan`], [`crate::block_inclusive_scan`], and
//! [`crate::block_inclusive_reduce`]: an identity, an associative
//! `combine`, and a lossless round-trip through a backing
//! [`CubePrimitive`] type.
//!
//! `combine` need not be commutative — the scan primitives feed `(left,
//! right)` in lane order (lower lane is `left`). Polynomial-division's
//! `PairScan<F>` is the load-bearing non-commutative case; sum and
//! product are both commutative.
//!
//! # Why an associated `Repr: CubePrimitive`
//!
//! cubecl 0.9 only supports a closed set of "primitive" types in three
//! places we need them: [`plane_shuffle_up`](cubecl::prelude::plane_shuffle_up)
//! (warp-level shuffles), [`SharedMemory<T>`](cubecl::prelude::SharedMemory)
//! indexing, and `let mut x: T; x = ...` mutation across loop iterations
//! / `if cond { a } else { b }` expressions. User structs (`#[derive(CubeType)]`)
//! don't satisfy `CubePrimitive` — and there's no derive — so a generic
//! scan written directly over `M: CubeType` doesn't compile.
//!
//! The [`Repr`](Monoid::Repr) escape hatch resolves this: each `Monoid`
//! pairs a friendly host-side struct (named fields, normal arithmetic)
//! with a `CubePrimitive` wire format. The scan code does its mechanics
//! in `Repr`-space, where every cubecl primitive works natively, and
//! crosses back to `M` only for the algebraic `combine`. For one-u32
//! monoids `Repr = u32`; for multi-word monoids `Repr = Line<u32>` (a
//! cubecl-native vector type, mapping to `vec*<u32>` on WGSL / packed
//! `uint*` on CUDA).
//!
//! Implementations live with the type the monoid wraps, not in r0-cube:
//! `Sum<F>` and `PairScan<F>` over `r0-field` elements live alongside
//! `Ext4` / `Ext5` in `r0-field`; recipe-specific monoids live with
//! their recipe. r0-cube intentionally ships no impls — only the shape.

use cubecl::prelude::*;

/// An algebraic monoid usable with the cubecl-side scan primitives
/// ([`crate::plane_inclusive_scan()`], [`crate::block_inclusive_scan()`],
/// [`crate::block_inclusive_reduce()`]).
#[cube]
pub trait Monoid: CubeType + Copy + Clone + Sized + Send + Sync + 'static {
    /// `CubePrimitive` wire format for this monoid value. The scan
    /// primitives shuffle, mutate, and shared-memory-index this type
    /// directly; only `combine` happens at the `Self` level. Use `u32`
    /// for one-word monoids, `Line<u32>` for multi-word.
    type Repr: CubePrimitive;

    /// Identity: `combine(identity(), x) == x == combine(x, identity())`.
    fn identity() -> Self;

    /// Associative binary operation. `left` is to the lower index, `right`
    /// to the higher; need not be commutative.
    fn combine(left: Self, right: Self) -> Self;

    /// Pack this monoid value into its [`Repr`](Self::Repr) wire format.
    fn to_repr(value: Self) -> Self::Repr;

    /// Inverse of [`to_repr`](Self::to_repr).
    fn from_repr(repr: Self::Repr) -> Self;
}
