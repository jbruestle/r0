# r0-polynomial — design

GPU-friendly operations on polynomials whose coefficients live in a
`r0-field` field (base or extension). Mirrors the role `r0-ntt` plays for
transforms: a small number of executor types, each owning device resources
and dispatching `r0-cube` recipe pipelines under the hood.

This is the pre-implementation design doc. Once the crate is built it
gets replaced by `README.md`.

---

## 1. Scope (initial)

- **Division by `(x - z)`** — `PolyDivExec::div_by_x_minus_z`. Headline
  kernel; the reason this crate exists right now. Implemented as a
  `ScanRecipe` for the `r0-cube` `ScanExec` driver.

Future kernels we know we'll want, deferred to later sessions:

- **Polynomial evaluation** at one or many points (Horner reduction).
- **FRI fold** — interpolate-and-fold, the core of a STARK's commit phase.
- **Out-of-domain evaluation / linear combination** — straightforward but
  needs the same recipe substrate.

All of these are map → scan/reduce → unmap shaped, so they all fall out
of the `r0-cube` `ScanRecipe` machinery.

## 2. Why this is a separate crate (not in r0-ntt)

Different math object. NTT operates on polynomials but the abstraction it
exposes is "transform a buffer of base-field elements". Polynomial-level
operations care about coefficients-vs-evaluations distinction, about
extension-field arithmetic, about per-row scalars (the `z` we divide by).
Different surface, different test oracle (Plonky3's polynomial helpers
rather than its NTTs), different scheduling.

## 3. Division by `(x - z)`

### 3.1 Math

Given `p(x) = a_0 + a_1 x + … + a_{n-1} x^{n-1}` and a field element `z`,
synthetic division by `(x - z)` yields:

- A quotient `q(x) = b_0 + b_1 x + … + b_{n-2} x^{n-2}`.
- A remainder `r ∈ F` (which equals `p(z)` by the remainder theorem).

The serial Horner-style recurrence (high coefficients first):

```
Q ← 0
for c in (a_{n-1}, a_{n-2}, …, a_1, a_0):     # descending degree
    Q ← z·Q + c
    emit Q                                     # → b_{n-2}, b_{n-3}, …, b_0, r
```

Each `Q` after step `k` is the desired output at position `k`.

### 3.2 Reframe as a parallel prefix scan

The update `Q ← z·Q + c` is left-multiplication by `M_c = [[z, c]; [0, 1]]`
on `[Q; 1]`. Composing two `M_c`'s gives another matrix of the same shape;
the set of these matrices is closed under (associative) multiplication.
Encode each as a pair `(p, a)` representing `[[p, a]; [0, 1]]`. The
combine is

```
(p_L, a_L) ⊕ (p_R, a_R) = (p_L · p_R,  p_R · a_L + a_R)
```

with identity `(1, 0)` and lift `c → (z, c)`. After an inclusive prefix
scan over the lifted inputs, position `k`'s `a` component is `Q_k` —
exactly the desired output.

### 3.3 Recipe

The shipped `r0_cube::ScanRecipe` shape (after step-3 implementation —
slightly different from the original sketch in this doc):

- No separate `Input` / `Output` / `Context` associated types. The
  recipe's `load` reads u32s directly and constructs the monoid; its
  `store` projects and writes u32s. Per-batch context flows in as
  `&Array<u32>` and the recipe interprets the layout (we read it via
  `F::load`).
- Recipe owns the index transformation, which is what we want anyway:
  to make the inclusive scan walk the polynomial in *descending* degree
  order (so the Horner-style recurrence accumulates from `a_{n-1}` down
  to `a_0`), the recipe reads `arr[n - 1 - scan_pos]` and writes back
  to the same flipped position.

```rust
pub struct DivByXMinusZ<F: ExtField>;

#[derive(CubeType, Copy, Clone)]
pub struct PairScan<F: ExtField> { pub p: F, pub a: F }

#[cube]
impl<F: ExtField> Monoid for PairScan<F> {
    type Repr = …;  // see §3.6

    fn identity() -> Self { Self { p: F::one(),  a: F::zero() } }
    fn combine(l: Self, r: Self) -> Self {
        Self { p: F::mul(l.p, r.p),
               a: F::add(F::mul(r.p, l.a), r.a) }
    }
    fn to_repr(value: Self) -> Self::Repr { … }
    fn from_repr(repr: Self::Repr) -> Self { … }
}

#[cube]
impl<F: ExtField> ScanRecipe for DivByXMinusZ<F> {
    type Monoid = PairScan<F>;

    fn load(zs: &Array<u32>, input: &Array<u32>, batch: u32, scan_pos: u32, n: u32, batch_count: u32) -> PairScan<F> {
        // descending order: read coefficient n-1-scan_pos
        let c = F::load(input, batch * n * F::DEGREE, n - 1 - scan_pos, n);
        // per-batch z (transposed layout, batch_count rows of F::DEGREE comps each)
        let z = F::load(zs, 0, batch, batch_count);
        PairScan { p: z, a: c }
    }

    fn store(_zs: &Array<u32>, output: &mut Array<u32>, batch: u32, scan_pos: u32, n: u32, _batch_count: u32, m: PairScan<F>) {
        F::store(output, batch * n * F::DEGREE, n - 1 - scan_pos, n, m.a);
    }
}
```

The recipe is the only file that touches the math; `ScanExec` does the
scanning, level-0 relift, and recursive-spine plumbing.

### 3.4 PairScan::Repr and the BB5 padding question

`r0_cube::Monoid` requires a `Repr: CubePrimitive` wire format (cubecl
0.9 only natively supports a closed set of "primitive" types in
`SharedMemory<T>` indexing, `plane_shuffle_up`, generic mutation, and
generic if-else — see r0-cube's README §3.1). We ship `Line<u32, N>`
for `N ∈ {1, 2, 4, 8, 16}` since power-of-two line sizes are what GPU
backends natively support.

PairScan's u32 word count by field:

| F | F::DEGREE | PairScan u32 words | Natural Repr |
|---|---|---|---|
| `BaseElem<BB>` | 1 | 2 | `Line<u32, 2>` |
| `BaseElem<KB>` | 1 | 2 | `Line<u32, 2>` |
| `Ext4<BB>` | 4 | 8 | `Line<u32, 8>` |
| `Ext4<KB>` | 4 | 8 | `Line<u32, 8>` |
| `Ext5<BB>` | 5 | **10** | `Line<u32, 16>` (37% padding) |

BB5 is the awkward one — 10 isn't a power of two. Plan: pad to
`Line<u32, 16>` and waste 6 u32 lanes. Spine memory cost at log_n=20,
batch=32: `32 × 16 × 16 = 8 KiB` (vs 5 KiB unpadded). Trivial against
device scratch. Worth revisiting if perf benchmarking shows the wider
load/store hurts; the alternative is a tuple of two Line types and
fanning out the cubecl plumbing per-field, more intrusive.

Sum<F> (used by FRI fold etc.) has the same shape question with
`F::DEGREE` words instead of `2 × F::DEGREE`; same padding plan for
BB5 (`Line<u32, 8>` for 5 real words).

### 3.5 API and conventions

```rust
pub struct PolyDivExec<R: Runtime> { scan: ScanExec<R> }

impl<R: Runtime> PolyDivExec<R> {
    pub fn new(device: &Device<R>) -> Self;

    /// In-place division by (x - z) for `batch` polynomials of length 2^log_n.
    /// Buffer holds `batch * (1 << log_n) * F::DEGREE` u32s in transposed
    /// layout (per `ExtField::load/store`). Per-batch `z` values are at
    /// `zs` (one extension element per batch row).
    ///
    /// Output convention: rotate=true. In each polynomial:
    ///   - quotient coefficients at positions [0..n-1]
    ///   - remainder at position [n-1]
    pub fn div_by_x_minus_z<F: ExtField>(
        &self,
        buf: &Handle,
        zs:  &Handle,
        log_n: u32,
        batch: usize,
    );
}
```

Sizes: powers of two, `log_n ∈ [1, 24]` (matching `r0-ntt`).

The `rotate=true` output convention puts the quotient at the natural
lowest-degree-first slots, with the remainder taking the otherwise-vacant
top slot. The recipe generates outputs in descending-position order — the
last emitted scan output corresponds to the polynomial's constant-term
input and is `r`. Mapping scan position `k → output index` is just
`k → n - 1 - k` for the quotient slots, and `k = n - 1 → n - 1` for the
remainder. (Sppark's `rotate=false` mode rotates the other way; we don't
need it.)

### 3.6 Generic over the inner field

`F: ExtField` covers all five field instances we ship — `BaseElem<P>`
when the polynomial is base-field, `Ext4<P>` for degree-4 extensions,
`Ext5<P>` for degree-5. The kernel never needs to know which; the
recipe's monoid combine is `F::mul`/`F::add` and load/store inherits the
transposed layout from `ExtField::load/store`.

## 4. Tests

- **Unit (host)**: serial `div_by_x_minus_z` reference function over each
  field, exercised in a `#[test]` against hand-constructed cases (low-N
  polynomials with known quotient/remainder, and the
  "`p(x)·(x − z₀)` divided by `(x − z₀)`" identity).
- **Cube oracle**: kernel result vs the host serial reference. Iterate
  `log_n` 1..=24, each of the five fields, per-batch `z` values drawn
  randomly. CPU + wgpu backends; CUDA gated behind `cuda` feature like
  `r0-ntt`.
- **Random batch sweep**: same as `r0-ntt`'s batch sweep — sizes `[1, 2,
  3, 5, 7, 16, 17, 32, 33, 100]` to exercise sub-batch slicing once we
  add it.
- **Plonky3 cross-check**: Plonky3 has polynomial division in
  `p3-field` (or close to it). If the API line up, use it as a
  third-party oracle; otherwise the serial host reference is sufficient.

## 5. Future work (sketches, deferred)

- `evaluate(p, &[z]) -> Vec<F>` — Horner over each `z`, batched.
  Same `ScanRecipe` machinery, with `Output = ()` and a final reduction.
- `fri_fold` — bisect a polynomial under a beta value, Ext4-coefficient
  scans of pair-of-pair monoids. Definitely also a `ScanRecipe`.
- Out-of-domain evaluations and linear combinations across polynomials.

These will exercise the same `r0-cube` substrate. If the substrate isn't
right we'll find out via the second recipe; that's the test.

## 6. Performance targets

Headline target: BB^4, `log_n = 20`, batch 32, `< 1 ms` end-to-end on
CUDA — slightly behind the sppark baseline is fine for v1; closing the
gap with autotune is a follow-up. wgpu/Metal will be 3–4× slower per the
NTT precedent; tolerable for the browser-prover story.
