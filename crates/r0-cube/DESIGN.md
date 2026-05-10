# r0-cube — design

Utilities for using cubecl in this workspace. Not a wrapper *around* cubecl —
cubecl is the engine; this crate is the project-specific helper layer that
sits on top.

This is the design doc, written before implementation. Once the crate is
built it'll be replaced by `README.md` describing what actually shipped
(and any small deviations from this plan). Don't be precious about either —
if the implementation finds a better shape, change it.

---

## 1. Purpose

Two kinds of thing live here:

1. **Process / device hygiene** — anything that's about *managing* cubecl
   resources across many tests, kernels, and runtimes, rather than about
   doing math. Currently `Device<R>` (the cross-process device lock).
2. **Generic kernel primitives** — `#[cube]` helpers that aren't tied to a
   specific math object. Initially: a `Monoid` trait + plane- and
   block-level prefix-scan/reduce + a generic `ScanExec` driver that runs
   the standard map → scan → unmap pipeline for any recipe.

Out of scope: anything tied to a specific algebraic structure. Field
arithmetic stays in `r0-field`; polynomial-shaped operations stay in
`r0-polynomial`; transforms stay in `r0-ntt` (etc.). r0-cube only knows
about `T: CubeType` and trait-generic associativity.

## 2. Phase 1 contents (what to ship first)

```
src/
  lib.rs       -- module re-exports
  device.rs    -- Device<R>, moved from r0-field
  monoid.rs    -- Monoid trait, with proptest-friendly host impls
  scan.rs      -- plane_inclusive_scan, block_inclusive_scan,
                   block_inclusive_reduce
  exec.rs      -- ScanRecipe trait, ScanExec driver, 3-kernel pipeline
                   (with optional recursive spine)
tests/
  monoid_smoke.rs   -- (u32, +) and (u32, *) sanity for the scan primitives
  exec_smoke.rs     -- end-to-end ScanExec with a trivial recipe (sum of u32),
                       verifies the level-0 lift/scan/project + spine work
                       on CPU and wgpu. No field code. Catches all the
                       plumbing bugs before any field-aware recipe touches it.
```

`Device<R>` moves over from `r0-field` in the same commit that creates this
crate, with all `r0_field::Device` imports across the workspace switched to
`r0_cube::Device` in the same change — no transitional re-export.

## 3. The Monoid trait

```rust
#[cube]
pub trait Monoid: CubeType + Copy + Clone + Sized + Send + Sync + 'static {
    fn identity() -> Self;
    /// `combine(left, right)` — left applied first, then right.
    /// Must be associative; need not be commutative.
    fn combine(left: Self, right: Self) -> Self;
}
```

Implementations live with the type they operate on. A few we'll have early:

- An additive `Sum<F: ExtField>` (combine = `F::add`) — used by FRI fold and
  by the smoke tests.
- A scan-pair `PairScan<F: ExtField>` for the polynomial-division recipe
  (combine described in §5).

## 4. Block-level scan primitives

Three orthogonal operations, all generic over `<M: Monoid>`:

```rust
#[cube]
pub fn plane_inclusive_scan<M: Monoid>(value: M) -> M;

#[cube]
pub fn block_inclusive_scan<M: Monoid>(
    value: M,
    scratch: &mut SharedMemory<M>,   // num_warps entries
) -> M;

#[cube]
pub fn block_inclusive_reduce<M: Monoid>(
    value: M,
    scratch: &mut SharedMemory<M>,
) -> M;
```

Implementation strategy:

- Plane scan: Hillis-Steele over `plane_shuffle_up(v, off)` for
  `off ∈ {1, 2, 4, 8, 16}`, with a `lane >= off` guard. cubecl 0.9's
  `plane_inclusive_sum` is fixed-op so we can't reuse it; the
  `plane_shuffle_up` primitive is generic over `CubePrimitive` and we
  build the rest ourselves.
- Block scan: plane scan, then last lane of each warp writes to scratch,
  `sync_cube`, warp 0 scans scratch, `sync_cube`, each warp reads
  `scratch[warpid - 1]` as carry, combines into its lanes.
- Block reduce: same but only carries the final value out; cheaper because
  no per-lane carry phase.

Shared-memory budget: `num_warps × sizeof(M)`. Concrete numbers, e.g. for
`PairScan<Ext4>` (32 B):

| WG size | warps | shared mem |
|---|---|---|
| 1024 | 32 | 1 KB |
| 256  | 8  | 256 B |
| 128  | 4  | 128 B |

Per-thread state (the K elements each thread holds before/after scan)
lives in registers; that's a separate budget.

## 5. The map → scan → unmap pattern (`ScanRecipe`)

Every batched parallel-prefix kernel we'll need follows the same shape:

> 1. **lift** each input element into a monoid value
> 2. **scan** the monoid array
> 3. **project** each scanned monoid back into an output element

`ScanRecipe` captures exactly that, generic over four types:

```rust
#[cube]
pub trait ScanRecipe {
    type Input:   CubeType + Copy;
    type Monoid:  Monoid;
    type Output:  CubeType + Copy;
    type Context: CubeType + Copy;   // per-batch-row data (e.g. z)

    fn lift(ctx: Self::Context, input: Self::Input, pos: u32) -> Self::Monoid;
    fn project(scanned: Self::Monoid, pos: u32) -> Self::Output;
}
```

No `correct`. We use the **relift** strategy (§7).

Three concrete recipes we'll have within the first round of code:

| Recipe | Input | Monoid | Output | Context | Used by |
|---|---|---|---|---|---|
| `SumU32`           | `u32`     | `Sum<u32>`         | `u32`   | `()`     | smoke test |
| `SumExt<F>`        | `F`       | `Sum<F>`           | `F`     | `()`     | (FRI fold inner sum, eventually) |
| `DivByXMinusZ<F>`  | `F`       | `PairScan<F>`      | `F`     | `F` (z)  | `r0-polynomial` |

## 6. ScanExec — the generic 3-kernel driver

```rust
pub struct ScanExec<R: Runtime> {
    client: ComputeClient<R>,
    // Per-level scratch buffers, allocated lazily on first use:
    spines: RefCell<Vec<Handle>>,
}

impl<R: Runtime> ScanExec<R> {
    pub fn new(device: &Device<R>) -> Self;

    pub fn run<Recipe: ScanRecipe>(
        &self,
        input:    &Handle,
        output:   &Handle,         // may alias input for in-place
        contexts: &Handle,         // [batch] × Recipe::Context
        n: u32,
        batch: u32,
    );
}
```

Pipeline:

```
Level 0 (recipe-aware):
  K0_block_reduce<Recipe>  : Input → Monoid (one per block), spine_0[block]
  …
  K0_block_apply<Recipe>   : Input + carry_from_spine_0
                              → relift, block-scan, combine carry, project, write Output

Levels 1..L (Monoid-on-Monoid; recipe drops out):
  Kn_block_scan            : Monoid → both per-position scanned monoid AND
                              per-block summary spine_n[block]
  …
  Kn_block_apply           : combine carry_from_spine_n into per-position
                              scanned monoid, in place

Top level (L+1):
  single_block_scan        : whole spine fits in one workgroup; done
```

Levels 1..L are just "scan an array of `Monoid`" — no `lift`/`project`,
no recipe needed. Level 0 is the only level that touches the recipe.

## 7. Why relift (not a `correct` method)

A naive design adds a fourth recipe method `correct(ctx, carry, pos,
provisional) -> output` so the apply kernel can read a *provisional*
projected output written by K1 and patch it with the spine carry. **We're
not doing that.**

The cleaner strategy:

| | K1 work | K2 work | K3 work | Total scan-equivalents |
|---|---|---|---|---|
| `correct`-method | block scan + project + write | spine scan | read + correct + write | ~2.0 |
| **relift**       | block **reduce** + write spine | spine scan | block scan + apply carry + project + write | ~1.5 |

Relift's K1 is a *reduce* not a scan — strictly less work — and it never
writes per-element output. K3 does the full scan once. Net: relift is
cheaper *and* the recipe is one method shorter. The `pow_by_squaring`
incantation that motivated `correct` was working around a problem that
relift doesn't have.

## 8. Recursive spine, for portability

One-pass spine fits in one workgroup ⟺ `num_blocks ≤ WG_spine × K_spine`,
where `WG_spine` is the spine kernel's workgroup size and `K_spine` is
elements-per-thread there. Plug in for our `n ≤ 2^24`:

| Per-WG max threads | `K = 4` everywhere reaches | spine recursion levels needed |
|---|---|---|
| 1024 | `2^28` ✓ | 0 |
| 256  | `2^20`   | 1 (for `log_n` 21..24) |
| 128  | `2^18`   | 2 |
| 64   | `2^16`   | 3 |

Total kernel count: `2 × (L + 1) + 1`. So 3 / 5 / 7 / 9 kernels for
`L = 0 / 1 / 2 / 3`. The level-1+ kernels are the same shape regardless of
recipe — uniform Monoid-in / Monoid-out — so adding levels is mechanical.

`ScanExec::run` chooses `L` from `client.properties().hardware`. Implement
the L=0 path first, gate L≥1 behind an explicit "needs deep spine" code path
that's exercised by a stress test (a recipe over a deliberately
log-sized-to-force-recursion N).

## 9. Open questions

- **Where do `Recipe::load` / `Recipe::store` live?** ScanExec needs to
  know how to read inputs and write outputs from `&Array<u32>`. Two
  shapes:
  1. `Recipe::Input` is `F: ExtField` and ScanExec calls `F::load(...)` —
     the recipe doesn't deal with layout at all. Simple, but assumes
     transposed layout.
  2. The recipe owns load/store: `Recipe::load(arr, batch_base, pos, n) -> Input`,
     `Recipe::store(arr, batch_base, pos, n, out)`. Lets a recipe scan
     over packed AoS or any other layout.

  Lean toward (1) since it matches our actual usage and ExtField's
  load/store already handle the transposed convention. Revisit if a
  recipe ever wants a different layout.

- **Should `Sum<F>` live in `r0-field` or `r0-cube`?** It's a `Monoid`
  impl over a field, so technically belongs in r0-field. But that creates
  a r0-field → r0-cube dep edge. Probably fine; r0-cube depends on cubecl
  only, r0-field depends on r0-cube + cubecl. Single direction.

- **Sub-batching for ScanExec.** When `batch × work_per_batch_row`
  exceeds what fits in a single launch, NttExec slices the buffer via
  `Handle::offset_start`. Same trick will work here once we hit the
  size — defer until an actual recipe needs it.

## 10. Crate dep direction (post-this-work)

```
r0-cube       --  no internal deps; cubecl only
r0-field      --  → r0-cube (for Device, Monoid via Sum<F>)
r0-ntt        --  → r0-field, r0-cube
r0-polynomial --  → r0-field, r0-cube, possibly r0-ntt
r0-ntt-web    --  → r0-ntt, r0-field
```
