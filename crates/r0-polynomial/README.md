# r0-polynomial

GPU-friendly polynomial-level operations on `r0-field` polynomials,
built on the [`r0-cube`](../r0-cube) `ScanRecipe` substrate. Currently
ships division by `(x − z)` (synthetic division reframed as a parallel
prefix scan); future kernels (evaluation, FRI fold, OOD eval) plug into
the same substrate.

This document is the design and implementation reference. For API usage
see the rustdoc.

## 1. Scope

In:

- **[`PairScan<F>`](src/pair_scan.rs)** — the monoid used by every
  Horner-style scan. Combine `(p_L, a_L) ⊕ (p_R, a_R) =
  (p_L · p_R, p_R · a_L + a_R)`, identity `(1, 0)`, lift `c → (z, c)`.
- **[`PairScanLayout`](src/pair_scan.rs)** — per-`F` trait carrying
  `LANES` and `pack` / `unpack` / `alloc_scratch` over a `Line<u32>`
  wire format. One small impl per concrete field; combine stays
  generic.
- **[`DivByXMinusZ<F>`](src/div_by_x_minus_z.rs)** — `ScanRecipe` over
  `PairScan<F>`. Reads coefficients in descending-degree order so the
  inclusive prefix scan accumulates the Horner recurrence; per-batch
  `z` lives in `contexts`.
- **[`PolyDivExec<F, R>`](src/exec.rs)** — thin wrapper around
  `ScanExec<R, DivByXMinusZ<F>>`. `(device, log_n_max, max_batch)` at
  construction; `div_by_x_minus_z(buf, zs, log_n, batch)` in place.
- **[`host_ref`](src/host_ref.rs)** — serial host reference + the
  `HostField` trait with impls for the five `r0-field` instances. Used
  as the test oracle but available to any consumer that wants a
  CPU-side check.

Out (future work, deferred until needed):

- `Sum<F>` monoid for FRI fold / evaluation. Same `Repr = Line<u32>`
  shape as `PairScan` but `LANES = D` instead of `2D`. We'll add it
  with the first kernel that needs it (probably FRI fold).
- Polynomial evaluation at one or many points.
- FRI fold / out-of-domain evaluation / linear combination.

## 2. Why a separate crate (not in `r0-ntt`)

Different math object. NTT operates on polynomials but the abstraction
it exposes is "transform a buffer of base-field elements". Polynomial-
level operations care about coefficients-vs-evaluations, extension-field
arithmetic, and per-row scalars (the `z` we divide by). Different
surface, different test oracle, different scheduling. The shared piece
is the cube/scan substrate, which lives in `r0-cube`.

## 3. Division by `(x − z)`

### 3.1 Math

Given `p(x) = a_0 + a_1 x + … + a_{n−1} x^{n−1}` and a field element
`z`, synthetic division by `(x − z)` yields a quotient `q(x)` and
remainder `r ∈ F` (which equals `p(z)` by the remainder theorem).

The serial Horner-style recurrence (high coefficients first):

```
Q ← 0
for c in (a_{n−1}, a_{n−2}, …, a_0):    # descending degree
    Q ← z · Q + c
    emit Q                                # → b_{n−2}, b_{n−3}, …, b_0, r
```

The update `Q ← z·Q + c` is left-multiplication by
`M_c = [[z, c]; [0, 1]]` on `[Q; 1]`. These matrices are closed under
(associative) multiplication, so an inclusive prefix scan over the
lifted inputs gives `Q_k` at every position.

### 3.2 The scan recipe

`DivByXMinusZ<F>` implements `r0_cube::ScanRecipe` with
`Monoid = PairScan<F>`. Two methods:

- **`load`** reads `arr[n − 1 − scan_pos]` (descending degree),
  pulls `z` from the contexts buffer, returns `PairScan { p: z, a: c }`.
- **`store`** writes the scanned `a` back to `arr[n − 1 − scan_pos]`,
  giving the natural lowest-degree-first output convention with the
  remainder at position `n−1`.

`ScanExec` handles everything else (single-block fast path, recursive
spine for `n > wg_size²`, relift on apply).

### 3.3 Output convention (`rotate=true`)

In each polynomial slot `[0..n−1]` holds the quotient (lowest degree
first) and slot `[n−1]` holds the remainder `r = p(z)`. This is sppark's
`rotate=true` flavor; we don't need the other direction.

## 4. `PairScan<F>` and `PairScanLayout`

`PairScan<F>` is generic over `F: ExtField` with two field-typed
fields `p` and `a`. Its `Monoid` impl is a single blanket over
`F: PairScanLayout`, so `combine` (which has all the `F::add` /
`F::mul`) is written exactly once.

The per-`F` work — `Repr` lane count, the `pack` / `unpack` between
`(p, a)` and a `Line<u32>` wire format, and the lane-aware
`alloc_scratch` — lives in `PairScanLayout`, which has one tight impl
per concrete field:

| `F` | `F::DEGREE` | u32 words | `LANES` (padded) |
|---|---|---|---|
| `BaseElem<BabyBear>` / `BaseElem<KoalaBear>` | 1 | 2 | 2 |
| `Ext4<BabyBear4>` / `Ext4<KoalaBear4>` | 4 | 8 | 8 |
| `Ext5<BabyBear5>` | 5 | **10** | **16** (37% padding) |

BB^5 packs 10 real `u32`s into a 16-lane `Line<u32>`; the 6 padding
lanes hold zero (`Line::empty` initializer) and never participate in
combine. The trade-off is roughly two `Line<u32>` widths per scratch
slot vs not — the absolute cost stays tiny against the device's 64 MiB
scratch, and shipping non-power-of-two `Line` sizes would require
backend-specific surgery in cubecl.

### 4.1 Why a per-field layout trait, not five copies of the impl

Per the project memory, "5 impls that duplicate the pair scan combine
is slightly annoying, but not a show stopper". The `PairScanLayout`
split avoids that: combine + identity + the `to_repr` / `from_repr`
delegators are in one generic blanket; only the layout-trivial
`pack` / `unpack` / `alloc_scratch` are per-field. Total per-field
boilerplate: ~25 LOC.

## 5. Deviations from DESIGN.md

The shipped implementation differs from
[`DESIGN.md`](https://github.com/r0-prover-rewrite — replaced by this
README) in three ways. Captured here so the diff is first-class.

1. **r0-cube's `Monoid` trait grew `REPR_LANES` and `alloc_scratch`.**
   The original sketch had `type Repr: CubePrimitive` only. cubecl 0.9's
   `Line<P>` doesn't carry lane count in the Rust type — the lane count
   is attached at IR-construction time — so generic
   `SharedMemory::<M::Repr>::new(N)` silently allocates with line size 1
   regardless of how many lanes the impl actually wants. Lifting both
   pieces (`REPR_LANES` for host-side `ArrayArg` sizing, `alloc_scratch`
   for shared-memory creation) into the `Monoid` trait means each impl
   owns its own line-size knowledge and the scan substrate stays
   generic. Cost: a 2-line bump to existing one-word monoids
   (`type Repr = u32; const REPR_LANES = 1; fn alloc_scratch = …new`).

2. **r0-field gained three free `#[cube]` constructors:**
   `base_elem_from_raw`, `ext4_from_raws`, `ext5_from_raws`. The host
   `from_raw` is `pub const fn` — not callable from cube IR — and the
   `_p: PhantomData<P>` field on `BaseElem` / `Ext4` / `Ext5` is
   private. r0-polynomial's `PairScanLayout::unpack` needs both, so
   r0-field exposes the bridge.

3. **`Sum<F>` is not yet shipped.** The DESIGN.md mentioned it; nothing
   in this iteration uses it (poly-div only needs `PairScan`), so
   following the user's directive ("no need to prebuild Sum yet") it's
   deferred.

## 6. Testing

`tests/div_smoke.rs` runs the cube path against a serial host
reference (`host_ref::div_by_x_minus_z_serial`) for all five fields.
Each test:

1. Generates `batch` random host polynomials (length `2^log_n`) and
   per-row `z` values from a deterministic seed.
2. Runs the serial reference per row → expected output.
3. Packs into transposed `u32` buffers, runs `PolyDivExec`, reads back.
4. Compares limb-by-limb.

Sweep covers `log_n ∈ {1, 4, log_wg, log_wg+2, 2·log_wg, 2·log_wg+1}`
with `batch = 2`, exercising the single-block fast path, L=0
multi-block, and L=1 multi-block. On Mac wgpu (`log_wg = 10`) that's
through `n = 2^21` per row.

A trivial pre-check (`check_zero_smoke`) divides the zero polynomial
by `(x − z)` first and asserts the output is also zero, catching gross
plumbing failures before the 60–200s random sweep.

CPU runtime is skipped: cubecl's CPU emulator reports `plane_size = 1`,
which forces `wg_size = 1` in the scan substrate. CUDA is gated behind
the `cuda` feature per the workspace convention.

## 7. File layout

```
src/
  lib.rs              -- module re-exports + crate-level docs
  pair_scan.rs        -- PairScan<F> monoid + PairScanLayout (5 impls)
  div_by_x_minus_z.rs -- DivByXMinusZ<F> recipe
  exec.rs             -- PolyDivExec<F, R> wrapper
  host_ref.rs         -- HostField trait + 5 impls + serial reference
tests/
  div_smoke.rs        -- cube vs serial-host oracle (5 tests, one per F)
```
