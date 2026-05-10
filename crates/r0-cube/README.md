# r0-cube

Project-specific helpers on top of [cubecl](https://github.com/tracel-ai/cubecl):
the cross-process device lock, the shared scratch buffer that executors
borrow from, and a generic prefix-scan substrate (`Monoid` trait, plane-
and block-level `#[cube]` primitives, recipe-driven `ScanExec` driver
with recursive spine). Math-object-specific code lives in higher crates
(`r0-field`, `r0-ntt`, `r0-polynomial`); r0-cube knows only about
`CubeType`s and trait-generic associativity.

This document is the design and implementation reference. For API usage
see the rustdoc.

## 1. Scope

In:

- **[`Device<R>`]**: process-shared exclusive lock around a cubecl
  device, plus the shared `ComputeClient<R>` and a fixed-size scratch
  `Handle` that all executors borrow.
- **[`Monoid`] trait** + the three plane / block scan primitives
  ([`plane_inclusive_scan`], [`block_inclusive_scan`],
  [`block_inclusive_reduce`]).
- **[`ScanRecipe`] + [`ScanExec`]**: the application-facing scan driver.
  Runs the standard map → scan → unmap pipeline against a recipe with
  recursive-spine support for transforms larger than `wg_size²`.

Out (lives elsewhere):

- Field arithmetic and `ExtField` → `r0-field`.
- NTT kernels → `r0-ntt`.
- Polynomial-division and other algebra-aware recipes → `r0-polynomial`.
- Monoid impls — they live with the type they wrap (`Sum<F>` /
  `PairScan<F>` for field elements live in `r0-field` / `r0-polynomial`).
  r0-cube intentionally ships none.

## 2. `Device<R>` — lock + client + scratch

cubecl backends like wgpu share a single GPU across the host, and `cargo
test` runs each integration-test binary as its own process **in
parallel**. Concurrent kernel launches from different test binaries can
saturate the device and fail with timeouts. Within a process, a
`parking_lot::Mutex` would suffice; across processes it doesn't.

`Device<R>` wraps `R::Device` together with a process-shared
[`flock(2)`-based](https://docs.rs/fs2) advisory lock keyed per cubecl
runtime, and additionally owns the cubecl `ComputeClient<R>` and a
single fixed-size scratch `Handle`. Executors (e.g. `NttExec`,
`ScanExec`) clone the client and a scratch reference at construction
time instead of allocating their own — one buffer per device, shared
under the same lock that protects the device itself.

```rust
use r0_cube::Device;
use cubecl::wgpu::WgpuRuntime;

#[test]
fn my_kernel_test() {
    let device = Device::<WgpuRuntime>::acquire();        // 64 MiB scratch
    // ... or Device::<R>::acquire_with_scratch(512 * 1024 * 1024) for explicit budget
    let exec = NttExec::<BabyBearParameters, _>::new(&device);
    // … kernel work …
}   // device drops, lock releases
```

The lock is per-runtime (keyed by `core::any::type_name::<R>()`), so
wgpu and CPU tests do not block each other — concurrency is reduced
only where it must be. On `wasm32` (browser builds) the file lock is a
no-op since there are no concurrent processes to coordinate with.

The scratch is sized at acquire time (default 64 MiB; or via
`acquire_with_scratch[_for]` for explicit budget). It's stable for the
device's lifetime — any executor that needs more must size up the
device. A future revision will add grow-on-demand if that turns out to
be too rigid.

## 3. The `Monoid` trait

```rust
#[cube]
pub trait Monoid: CubeType + Copy + Clone + Sized + Send + Sync + 'static {
    type Repr: CubePrimitive;
    fn identity() -> Self;
    fn combine(left: Self, right: Self) -> Self;
    fn to_repr(value: Self) -> Self::Repr;
    fn from_repr(repr: Self::Repr) -> Self;
}
```

Two algebra methods (`identity`, `combine`) plus a lossless round-trip
through a `CubePrimitive` wire format (`to_repr` / `from_repr`).
`combine(left, right)` is associative; need not be commutative — the
scan primitives feed `(left, right)` in lane order so e.g. polynomial
division's `PairScan` works.

### 3.1 Why `Repr`

cubecl 0.9 supports a closed set of "primitive" types in three places
the scan code needs them:

| Operation | Required bound |
|---|---|
| `plane_shuffle_up(value, delta)` | `value: CubePrimitive` |
| `SharedMemory<T>` indexing | `T: CubePrimitive` |
| `let mut x: T; x = …` across loop iterations | `T: CubePrimitive` (in practice) |
| `if cond { a } else { b }` returning `T` | `T: CubePrimitive` |

`#[derive(CubeType)]` user structs don't satisfy `CubePrimitive` and
there's no derive for it. So a generic scan written directly over
`M: CubeType` doesn't compile — the macro can't synthesize the
expand-time wiring for non-primitive `M`.

`Repr` is the escape hatch: each `Monoid` pairs a friendly host-side
struct (named fields, normal arithmetic) with a `CubePrimitive` wire
format. The scan code does its mechanics in `Repr`-space and crosses
back to `Self` only for the algebraic `combine`. For one-u32 monoids
`Repr = u32`; for multi-word monoids `Repr = Line<u32, N>` (a cubecl
SIMD-style vector, mapping to `vec*<u32>` on WGSL / packed `uint*` on
CUDA).

### 3.2 Where impls live

**Not in r0-cube.** Per the dep direction, monoid impls live with the
type they wrap. `Sum<F>` / `PairScan<F>` over `r0-field` elements live
in `r0-field` next to `Ext4` / `Ext5`; recipe-specific monoids live with
their recipe. This crate ships only the trait shape so `r0-field` →
`r0-cube` (not the other way) keeps the dep arrow clean.

## 4. Plane- and block-level scans

```rust
#[cube] pub fn plane_inclusive_scan<M: Monoid>(value: M, log_warp: u32) -> M;
#[cube] pub fn block_inclusive_scan<M: Monoid>(
    value: M,
    scratch: &mut SharedMemory<M::Repr>,
    log_warp: u32,
    log_wg: u32,
) -> M;
#[cube] pub fn block_inclusive_reduce<M: Monoid>(
    value: M,
    scratch: &mut SharedMemory<M::Repr>,
    log_warp: u32,
    log_wg: u32,
) -> M;
```

`plane_inclusive_scan` is Hillis-Steele over `plane_shuffle_up` on
`M::Repr`. It's written as a **comptime-recursive chain** rather than a
mutating `for` loop because `let mut v: M; v = ...` is one of the
patterns cubecl 0.9 doesn't accept for generic `CubeType`.

`block_inclusive_scan` is the standard two-stage thing: scan within
each warp, write per-warp totals to shared scratch, scan those, combine
the carry back. Two `sync_cube`s. Constraint: `wg_size <= warp_size²`
(so warp 0's per-warp-totals scan fits one plane). For `warp_size = 32`
that's `wg_size <= 1024`, which covers every device we target.

`block_inclusive_reduce` is the same shape minus the carry-combine
phase — the block total ends up in slot 0 of `scratch` and is read by
every lane.

Callers allocate `SharedMemory::<M::Repr>::new(num_warps)`. Query the
warp size at host time via `client.properties().hardware.plane_size_max`
and pass the log of that.

## 5. `ScanRecipe` + `ScanExec`

The application-facing scan driver. A **recipe** says how to read an
input element, lift it into a `Monoid` value, and project a scanned
monoid back to an output element; it owns the index transformation
(lets `DivByXMinusZ` read in descending degree order so the Horner-style
scan goes the right way) and the per-batch-row context interpretation.

```rust
#[cube]
pub trait ScanRecipe: 'static + Send + Sync {
    type Monoid: Monoid;
    fn load (ctx: &Array<u32>, input:  &Array<u32>,    batch: u32, scan_pos: u32, n: u32, batch_count: u32) -> Self::Monoid;
    fn store(ctx: &Array<u32>, output: &mut Array<u32>, batch: u32, scan_pos: u32, n: u32, batch_count: u32, value: Self::Monoid);
}
```

Two notes on this shape:

- **Context as `&Array<u32>`, recipe-interpreted layout.** An earlier
  sketch had `type Context: CubeType` baked into the trait, but the
  cubecl 0.9 issues with multi-word `CubeType` plumbing (the same ones
  `Monoid` works around with `Repr`) would have surfaced again here.
  Passing `contexts: &Array<u32>` and letting the recipe pull what it
  needs (typically via `ExtField::load` for field-shaped contexts)
  sidesteps that. For recipes that need no per-batch context (sum,
  product), the buffer is a one-u32 dummy that the recipe's `load`
  ignores.
- **Load fuses lift, store fuses project.** The recipe owns layout end
  to end and `ScanExec` never sees raw element types.

```rust
pub struct ScanExec<R: Runtime, Recipe: ScanRecipe> { /* client, spines, … */ }

impl<R, Recipe> ScanExec<R, Recipe> {
    pub fn new(device: &Device<R>, log_n_max: u32, max_batch: usize) -> Self;
    pub fn run(
        &self,
        contexts: &Handle, input: &Handle, output: &Handle,
        log_n: u32, batch: usize,
    );
}
```

### 5.1 Pipeline

For `n <= wg_size`: a single `k_single_block` kernel does the whole
scan in one workgroup per polynomial.

For `n > wg_size`: pick `L = ceil(log_n / log_wg) - 2` spine recursion
levels (`L = 0` covers `n <= wg_size²`, `L = 1` covers up through
`wg_size³`, etc.) and dispatch `2(L+1) + 1` kernels:

1. `k0_reduce<Recipe>` — recipe-aware. Each level-0 block reduces;
   lane 0 writes its total to `spine[0][block]`.
2. `k_reduce_spine<M>` × L — each upper-spine cell becomes the
   reduction of `wg_size` consecutive lower-spine cells.
3. `spine_top_scan<M>` — one workgroup per polynomial does the
   top-level inclusive scan in place.
4. `k_apply_spine<M>` × L (in reverse) — each group re-scans
   `wg_size` lower-spine cells, combines with the upper-spine carry,
   writes back. After this `spine[0][k]` holds the inclusive prefix of
   the original input through block `k`.
5. `k0_apply<Recipe>` — recipe-aware. Re-loads input, re-scans within
   block, combines with `spine[0][block-1]` as carry, projects via
   `Recipe::store`.

Recipe-aware kernels are monomorphized per `Recipe`; spine kernels are
monomorphized per `Recipe::Monoid` and shared across all spine levels —
only the comptime sizes differ at launch time.

### 5.2 Why "relift" instead of a `correct` method

A naive design adds a fourth recipe method `correct(ctx, carry, pos,
provisional) -> output` so the apply kernel reads a *provisional*
projected output written by an earlier kernel and patches it with the
spine carry. We don't do that.

The relift strategy — re-reading and re-scanning the input in the apply
pass — is cheaper *and* the recipe is one method shorter:

| | reduce work | scan work | apply work |
|---|---|---|---|
| `correct`-method | block scan + project + write | spine scan | read + correct + write |
| **relift**       | block **reduce** + write spine | spine scan | block scan + apply carry + project + write |

K1 is a *reduce* (not a scan) — strictly less work — and never writes
per-element output. K-final does the full per-element scan once.

### 5.3 Spine layout

Each spine level slices into `device.scratch()` at a fixed byte offset
computed at construction time, sized for
`max_batch × num_blocks_at_level(log_n_max, log_wg, level) ×
sizeof(M::Repr)`. Smaller per-call `(log_n, batch)` use only the
leftmost portion of each slice. Spines are typed as `Array<M::Repr>`
(CubePrimitive — index natively).

For `log_n_max = 22` on Mac wgpu (`log_wg = 10`, `Repr = u32`):

| Level | `num_blocks` | Bytes for `max_batch = 4` |
|---|---|---|
| 0 | 4096 | 64 KiB |
| 1 | 4    | 64 B  |

Total ~64 KiB; trivial against the device's default 64 MiB scratch.

### 5.4 Limits

- `wg_size` is fixed at the device's `max_units_per_cube`.
- `log_n_max` is bounded only by scratch budget and device grid-dim
  limits. A future revision will sub-batch when `num_blocks_0` exceeds
  the device's `max_cube_count.0`.
- The single-block fast path uses `wg_size = min(2^log_n, device max)`
  so small-N scans don't waste threads.

## 6. cubecl 0.9 specifics worth knowing

These bit while building the scan and inform the design above. The
[top-level `CUBECL_NOTES.md`](../../CUBECL_NOTES.md) catalogues
workspace-wide cubecl quirks; the ones below are r0-cube-specific.

- **Generic `CubeType` is not first-class.** Mutating a `let mut M`,
  using `if cond { a } else { b }` returning `M`, and indexing
  `SharedMemory<M>` all require `M: CubePrimitive` in 0.9. The `Repr`
  associated type and `combine_via` helper let the scan code stay in
  primitive-space and only cross to `M` for the algebraic `combine`.
- **No comptime arg can be derived from a trait constant in cube
  bodies.** `M::WORDS` (or anything similar) doesn't reach through
  generic bounds inside a `#[cube]` body. Callers compute and pass
  comptime sizes (e.g. `num_warps`, `n`) as `#[comptime] u32` from the
  host instead.
- **`comptime!(N as usize)` to satisfy `SharedMemory::new`.** That
  function takes a `usize` comptime; if your kernel param is a `u32`,
  cast inside `comptime!()`.
- **Recursive `#[cube]` fns work for comptime branching.** The plane
  scan uses `if comptime!(log_warp == 0) { value } else { ... recurse
  with log_warp - 1 ... }` to avoid the `let mut` issue. cubecl
  resolves the recursion at IR build time.

## 7. Testing

- **`tests/scan_smoke.rs`** — exercises the three scan primitives
  directly with a private `(u32, +)` monoid (`SumU32`). Runs single-warp
  + multi-warp + warp-size² wg configs on wgpu.
- **`tests/exec_smoke.rs`** — end-to-end `ScanExec + ScanRecipe`
  smoke. Six configurations covering the single-block fast path, L=0
  multi-block, and L=1 multi-block (up to `n = 2² × wg_size²` ≈ 16M
  elements per row on Mac wgpu).

CPU runtime is skipped: cubecl's CPU emulator reports `plane_size = 1`,
which forces our `wg_size <= warp_size²` constraint to `wg_size = 1`,
a single-thread degenerate case that doesn't exercise anything
interesting. wgpu is the meaningful target until we add a different
single-thread fallback.

## 8. File layout

```
src/
  lib.rs       -- module re-exports + crate-level docs
  device.rs    -- Device<R>: cross-process lock + client + shared scratch
  monoid.rs    -- Monoid trait
  scan.rs      -- plane / block scan + reduce primitives
  recipe.rs    -- ScanRecipe trait
  exec.rs      -- ScanExec driver + the 5 scan-pipeline kernels
tests/
  scan_smoke.rs -- direct tests of the scan primitives
  exec_smoke.rs -- end-to-end ScanExec + ScanRecipe
```
