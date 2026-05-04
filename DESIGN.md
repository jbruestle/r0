# `hal` — Batched NTT/iNTT over 31-bit fields, on GPU + CPU

A Rust library for batched Number-Theoretic Transforms over highly 2-adic 31-bit
prime fields (BabyBear, KoalaBear), portable across GPU and CPU via
[`cubecl`](https://github.com/tracel-ai/cubecl).

This document specifies the design and grounds each choice in concrete
references to two state-of-the-art implementations:

- **GPU**: Supranational's [`sppark`](https://github.com/supranational/sppark),
  specifically `~/src/sppark/ntt/`. Targets CUDA. Used as the algorithmic
  blueprint.
- **CPU**: Plonky3's `dft` and `monty-31` crates. Used as the field-arithmetic
  blueprint and the CPU-side reference oracle.

---

## 1. Scope and target workload

- **Workload**: hundreds of independent NTT / iNTT instances per second, each
  of size 2^k with k ∈ \[10, 24\]. The headline target is k = 20 (1M points)
  but the library must scale up to k = 24 (16M points) without architectural
  changes — just an extra kernel pass. k = 24 is also the hard ceiling: it
  matches KoalaBear's 2-adicity exactly (the largest power-of-two subgroup
  available), and we have no interest in larger transforms.
- **Fields**: BabyBear (`p = 2^31 − 2^27 + 1`, 2-adicity 27) and KoalaBear
  (`p = 2^31 − 2^24 + 1`, 2-adicity 24). Both fit a single `u32` in Montgomery
  form. See `~/src/Plonky3/baby-bear/src/baby_bear.rs:18-42` and
  `~/src/Plonky3/koala-bear/src/koala_bear.rs:21-67`.
- **Backends**:
  1. **CUDA** (NVIDIA, via cubecl-cuda) — max-speed path. Specialized inner
     kernel using subgroup-shuffle butterflies. This is where we compete with
     sppark on raw throughput.
  2. **Metal** (Apple Silicon, via cubecl-wgpu's WGSL→Metal lowering) and
     **Vulkan** (AMD/Intel/NVIDIA-Linux, via cubecl-wgpu's WGSL→Vulkan
     lowering). Same WGSL kernel; gets the subgroup fast path when the
     subgroups feature is exposed (Metal: yes; Vulkan: yes on RDNA2+ and
     Pascal+; older AMD: portable-fallback path).
  3. **Browser GPU** (cubecl-wgpu's WGSL → WebGPU). Portable-fallback
     butterfly path: workgroup-memory exchange with `workgroupBarrier()`.
     The browser target is WebGPU, not WebGL — WebGL2 has no standard
     compute-shader path and is out of scope.
  4. **CPU** (cubecl's CPU backend) — runs the exact same `#[cube]` kernel
     for laptops, CI, and as a fallback when no GPU is present.
- **Non-goals**: large primes (256-bit BLS / Goldilocks); LDE / coset domains
  in v1 (added later); MSM, polynomial commitments, FRI; PTX/inline-asm; multi-
  GPU; Plonky3 source/runtime compatibility (we use it only as a test oracle).

## 2. Reference walk-through

### 2.1 sppark: what to copy

sppark is the cleanest production-grade GPU NTT for 31-bit fields. The pieces
worth lifting verbatim:

- **Mixed-radix decomposition by domain size.** `~/src/sppark/ntt/ntt.cuh:100-158`
  selects 1, 2, 3, or 4 kernel passes depending on `lg`:
  - `lg ≤ 10`: one monolithic pass.
  - `10 < lg ≤ 18`: two passes of `≈lg/2`.
  - `18 < lg ≤ 30`: three passes of `≈lg/3`.
  - `30 < lg ≤ 40`: four passes.
  Each pass runs entirely in shared memory. For our 1M-point case (lg=20)
  this is two passes of radix 10 — exactly the regime sppark hits hardest.
- **Narrow vs wide kernels.** `kernels/ct_mixed_radix_narrow.cu` packs
  `Z_COUNT = 256/(8·sizeof(field))` field elements per thread (32 for our
  31-bit fields), letting one block handle 32× more elements in the same
  shared memory footprint. This is the high-throughput path for 32-bit
  fields and is what we should imitate.
- **Warp-shuffle butterflies for the inner 6 stages.** `mont32_t::shfl_bfly`
  at `~/src/sppark/ff/mont32_t.cuh:262-263` implements the radix-2
  partner exchange via `__shfl_xor_sync` on a 32-lane warp, avoiding shared
  memory entirely for stages 1–5. `ct_mixed_radix_narrow.cu:92-115`.
- **Partial / windowed twiddle table.** `~/src/sppark/ntt/parameters.cuh:189-209`
  stores `partial_twiddles[WINDOW_NUM][WINDOW_SIZE]`. For BabyBear the window
  is 64 entries × 5 windows ≈ 320 elements (1.3 KiB), and any
  `ω^k` for `k ∈ [0, 2^27)` is reconstructed by 4 multiplications using
  `get_intermediate_roots()` at `kernels.cu:278-298`. This lets us avoid a
  multi-MB twiddle table while keeping work-per-twiddle constant.
- **Bit-reversal as its own kernel**, fused with the outer pass when
  `InputOutputOrder` allows. `kernels.cu:16-129`.

### 2.2 sppark: what to *not* copy

- **Inline PTX.** `mont32_t.cuh:196-211` is hand-written PTX. cubecl writes
  Rust → multiple targets; we cannot ship PTX. cubecl's CUDA backend will
  emit nearly the same code from a clear `mul_hi`/`mul_lo` formulation
  (see §4).
- **Per-GPU NUMA twiddle placement** (`parameters.cuh:219-264` round-robins
  across GPU 0/1/2). Out of scope.
- **Single-transform-per-launch.** sppark's `NTT::Base` (`ntt.cuh:216-244`)
  runs one transform per call. For 100s of small-ish (1M) transforms, that
  burns kernel-launch overhead. We will batch on the grid-Y axis (see §9).

### 2.3 Plonky3: what to copy

- **Field representation.** `~/src/Plonky3/monty-31/src/monty_31.rs` and
  `monty-31/src/utils.rs:99-158` define the Montgomery form we will use bit-
  for-bit. The scalar `monty_reduce`:

  ```rust
  // utils.rs:105-125, paraphrased
  fn monty_reduce<MP>(x: u64) -> u32 {
      let t = x.wrapping_mul(MP::MONTY_MU as u64) & (MP::MONTY_MASK as u64);
      let u = t * (MP::PRIME as u64);
      let (x_sub_u, over) = x.overflowing_sub(u);
      let x_sub_u_hi = (x_sub_u >> MP::MONTY_BITS) as u32;
      let corr = if over { MP::PRIME } else { 0 };
      x_sub_u_hi.wrapping_add(corr)
  }
  ```

  with constants:

  | Field    | `PRIME`      | `MONTY_MU`   | 2-adicity | Generator |
  |----------|--------------|--------------|-----------|-----------|
  | BabyBear | `0x78000001` | `0x88000001` | 27        | 31        |
  | KoalaBear| `0x7f000001` | `0x81000001` | 24        | 3         |

- **Twiddle caching pattern.** `~/src/Plonky3/dft/src/radix_2_dit.rs:25-58`
  guards a `BTreeMap<usize, Arc<[F]>>` with `RwLock`, lazily computing
  `root.powers().take(n)` on first call for a given `log_h`, then handing
  out `Arc<[F]>` clones. For 100s of same-size NTTs this pays for itself
  on call #2.

### 2.4 Plonky3: what to *not* copy

- **Source / runtime compatibility.** Plonky3 bakes bit-reversal into
  several call sites (`radix_2_dit.rs:72`, `BitReversedMatrixView` shuttled
  through `radix_2_dit_parallel.rs:146,165`), and its `dft_batch` API
  always materializes a natural-order output unless the caller threads a
  view through. For our use case we want bit-reversed coefficients
  *natively* and want to skip bit-reversal entirely (see §7), so the
  Plonky3 trait shape would force us to add a kernel pass we don't want.
  We do not depend on Plonky3 at runtime. It exists in this design only
  as a reference for ideas and as a test oracle (§11).
- **Choice of FFT variant** — Plonky3 ships four (`Radix2Dit`,
  `Radix2DitParallel`, `Radix2Bowers`, `Radix2DFTSmallBatch`) because each
  wins on a different cache geometry. For GPUs the geometry is fixed
  (block, warp, registers), so we collapse to one mixed-radix design,
  matching sppark.
- **Column-as-polynomial matrix layout.** Plonky3's `dft_batch` takes a
  `RowMajorMatrix` where each *column* is a polynomial — element `i` of
  poly `j` lives at offset `i*B + j`. That's optimal for CPU cache when
  the batch dimension is small. The GPU-friendly layout is the opposite
  (see §9), so we do not adopt it.

## 3. cubecl and backend constraints

cubecl writes one Rust kernel and lowers to CUDA / WGSL / SPIR-V / CPU.
Constraints that shape the design:

- **No 64-bit native ints in WGSL.** WebGPU mandates `u32` and `i32` only.
  cubecl exposes `u64` via emulation, but the cheap and portable primitives
  are `u32 × u32 → u32` low and `u32 × u32 → u32` high. Our Montgomery
  reduction is written in those terms (see §4).
- **Native warp/subgroup shuffles vary by backend.** Subgroup ops
  (`subgroupShuffleXor`, etc.) are universally available on CUDA but
  conditional on WGSL: Metal (Apple Silicon) exposes them via the
  `subgroups` feature; Vulkan exposes them on RDNA2+ AMD and Pascal+
  NVIDIA; Chromium-stable WebGPU ships subgroup support behind a flag as
  of late 2025. **Decision: ship two butterfly inner loops, deliberately
  duplicated**, since simple-but-slow and fast-but-complex are in tension
  here:
  1. **Fast path — subgroup shuffles.** Mirrors sppark stages 1–5
     (`ct_mixed_radix_narrow.cu:92-115`, `mont32_t.cuh:262-263`). Used
     on CUDA and on WGSL when subgroups are available. This is where we
     try to match sppark on a real GPU.
  2. **Portable path — workgroup-memory butterflies.** All radix-2
     exchanges go through workgroup memory with `workgroupBarrier()`. No
     subgroup primitives. Slower but runs anywhere WGSL runs, including
     unflagged WebGPU.
  `comptime`-selected at codegen, not runtime — each backend gets one
  monomorphized kernel. The two paths share twiddle layout, batching,
  and the outer pass-decomposition logic, so the duplication is confined
  to the inner butterfly.
- **Workgroup memory budget**: WebGPU minimum is 16 KiB; CUDA gives 48 KiB
  (default) or up to 100 KiB; Metal gives ~32 KiB depending on family.
  For BabyBear (4 bytes/elem), 16 KiB = 4096 elements, comfortably above
  our 1024-radix inner kernel (4 KiB).
- **Workgroup size**: 256 invocations is the safe portable maximum.
  sppark's narrow kernel uses 512 (`ct_narrow.cu:226`); we cap at 256 to
  remain WebGPU-compliant. CUDA backend may opt up to 512 via comptime
  if profiling shows it helps.
- **AMD specifics**: wave size is 32 (RDNA) or 64 (GCN/CDNA). Both
  divide 256 cleanly. Subgroup-shuffle path works on RDNA2+ via Vulkan;
  GCN falls back to the portable path.
- **Apple specifics**: simdgroup size is 32. Metal subgroup-shuffle path
  works through cubecl-wgpu's WGSL→Metal compiler.

## 4. Field arithmetic on GPU

### 4.1 Representation

A field element is a `u32` in Montgomery form: `x_mont = x · 2^32 mod p`.
This matches Plonky3 (`monty_31.rs`) and sppark's `mont32_t`.

```rust
#[derive(Clone, Copy, CubeType)]
pub struct M31<P: MontyParams> {
    pub raw: u32,
    _marker: PhantomData<P>,
}
```

`MontyParams` is a `comptime` trait carrying `PRIME`, `MU = -p^{-1} mod 2^32`,
`R2 = 2^64 mod p`, `TWO_ADICITY`, and `TWO_ADIC_GENERATORS[0..k]` (Montgomery
form). Two implementations: `BabyBear` and `KoalaBear`. Constants come
straight from `~/src/sppark/ntt/parameters/baby_bear.h` and
`~/src/Plonky3/baby-bear/src/baby_bear.rs:18-51`.

### 4.2 Multiplication / Montgomery reduction

The hot path. Written entirely in 32-bit multiplies for WGSL portability:

```rust
#[cube]
fn monty_mul<P: MontyParams>(a: u32, b: u32) -> u32 {
    let lo = mul_lo(a, b);          // (a*b) mod 2^32
    let hi = mul_hi(a, b);          // (a*b) >> 32
    let q  = mul_lo(lo, P::MU);     // q = lo * MU mod 2^32
    let h  = mul_hi(q, P::PRIME);   // h = (q * p) >> 32
    // result in [0, p) ; conditional subtract:
    let (r, borrow) = hi.overflowing_sub(h);
    if borrow { r.wrapping_add(P::PRIME) } else { r }
}
```

Note we never form a `u64`. Both `mul_hi` and `mul_lo` lower to:

- CUDA: `__umulhi` / native `*`. Compiles to `mul.hi.u32` / `mul.lo.u32`,
  which is what sppark's hand-PTX emits at `mont32_t.cuh:196-211`.
- WGSL: WGSL has no `mul_hi` builtin, so cubecl emits the schoolbook split
  `(a_lo*b_lo, a_lo*b_hi + a_hi*b_lo, a_hi*b_hi)` with carries. ~10 32-bit
  ops vs CUDA's 2; expensive but correct. This is the dominant cost on
  WebGPU and the main reason a CUDA build will be ~5-10× faster per
  butterfly than a WGSL build.
- CPU: lowers to native `u64` multiply.

`add` and `sub` are straightforward (`utils.rs:54-86`):

```rust
#[cube] fn monty_add<P>(a: u32, b: u32) -> u32 {
    let s = a.wrapping_add(b);
    let (s2, over) = s.overflowing_sub(P::PRIME);
    if over { s } else { s2 }
}
#[cube] fn monty_sub<P>(a: u32, b: u32) -> u32 {
    let (d, borrow) = a.overflowing_sub(b);
    if borrow { d.wrapping_add(P::PRIME) } else { d }
}
```

These three primitives are everything the kernel needs.

### 4.3 Lazy reduction inside butterflies

sppark's `mont32_t::sqr_n` (`mont32_t.cuh:376-395`) keeps intermediate
values in `[0, 2p)` between chained mults and only `final_sub`s every
other iteration — the author's comment claims +20% on `reciprocal()`.
The same trick applies inside our radix-2 butterflies and is one of the
clearest perf wins to lift from sppark *without* using PTX:

- After Montgomery reduction, the canonical output is in `[0, p)`. But
  the natural unreduced output of the standard reduction sits in
  `[0, 2p)` — the final subtract is what canonicalizes it.
- For 31-bit primes, `2p < 2^32`, so `[0, 2p)` still fits a `u32`.
- A radix-2 butterfly does `(a, b) → (a + ω·b, a − ω·b)`. If `ω·b` is
  produced unreduced in `[0, 2p)` and `a` is in `[0, p)`, the sum is in
  `[0, 3p)` — too wide for the next mul to be sound, so we still need a
  reduction step somewhere. The right granularity is per-stage, not
  per-op: keep mults unreduced, reduce after each add/sub. That's what
  sppark does.
- We expose this as `monty_mul_lazy(a, b) -> u32` returning a value in
  `[0, 2p)`, plus `canonicalize(x) -> u32` (a single `final_sub`-style
  step) used at workgroup boundaries and before global stores.

This is a perf optimization, not a correctness requirement — implement
the canonical-form `monty_mul` first, get correctness against Plonky3,
then layer in lazy form as a `comptime` flag once the bounds analysis is
in tests.

### 4.4 Why Montgomery, not Plantard or Barrett?

Plantard reduction is theoretically cheaper for 31-bit fields with a
32-bit word (single `mul_hi` instead of two), but it has tighter input
bounds and is a footgun under signed inputs (which an inverse NTT
produces transiently). Plonky3 uses Montgomery; sppark uses Montgomery;
both BabyBear and KoalaBear's published constants are Montgomery.
Sticking with Montgomery means we can cross-check bit-for-bit against
Plonky3 on every operation — a huge win for testing.

## 5. NTT algorithm

### 5.1 Top-level decomposition

Following sppark `ntt.cuh:100-158`:

```
lg ≤ LG_BLOCK              : 1 kernel pass.
LG_BLOCK < lg ≤ 2*LG_BLOCK : 2 passes (six-step variant).
2*LG_BLOCK < lg ≤ 3*LG_BLOCK : 3 passes.
```

`LG_BLOCK` is the largest radix that fits in workgroup memory. With WebGPU's
16 KiB / 4 B = 4096-elem budget and a 256-thread workgroup processing
`Z_COUNT = 4` elements each (1024 elements per workgroup), `LG_BLOCK = 10`.
This gives the same 2-pass plan as sppark for our 1M (lg=20) target and a
3-pass plan for the 16M (lg=24) ceiling.

Dispatch over the full size range:

| lg  | passes | radices       | Notes                          |
|-----|--------|---------------|--------------------------------|
| 10  | 1      | (10)          | monolithic                     |
| 11  | 2      | (6, 5)        |                                |
| 12  | 2      | (6, 6)        |                                |
| 16  | 2      | (8, 8)        |                                |
| 20  | 2      | (10, 10)      | **headline 1M target**         |
| 21  | 3      | (7, 7, 7)     |                                |
| 22  | 3      | (8, 7, 7)     |                                |
| 23  | 3      | (8, 8, 7)     |                                |
| 24  | 3      | (8, 8, 8)     | **2-adicity ceiling**          |

This matches what sppark would pick (`ntt.cuh:124-158`). For 3-pass cases,
sppark balances `(lg/3, lg/3, lg/3 + rem)` — same here.

### 5.2 Inner kernel — what one workgroup does

Following sppark's narrow kernel (`ct_mixed_radix_narrow.cu:5-183`):

1. **Load**: 1024 elements, coalesced from global into registers (4 per
   thread). Optional transpose through workgroup memory for stride-aware
   coalescing (`ct_narrow.cu:169-174,318-333`).
2. **Stages 1–5**: radix-2 butterflies with partner exchange via subgroup
   shuffle (CUDA / wgpu+subgroups) or via workgroup memory + barrier
   (portable WGSL). Twiddles read from a small constant table —
   `radix6_twiddles[32]` in sppark, baked in as a `comptime` array for us.
3. **Stages 6–10**: radix-2 butterflies with partner exchange via workgroup
   memory and `workgroupBarrier()`. Twiddles for stages 6–9 come from
   `radixX_twiddles_X` precomputed tables (`ct_mixed_radix_wide.cu:153-206`);
   stage 10's twiddle is the *partial-twiddle* reconstruction (see §6).
4. **Index rotation, not bit-reversal.** sppark's `ct_narrow.cu:160-167`
   trick:
   ```
   mask = ((1 << iterations) - 1) << stage;
   rotw = (rotw >> 1) | (rotw << (iterations - 1));
   ```
   A circular shift of `iterations` bits at position `stage`. This is the
   Stockham-equivalent reorder that lets the next pass read with the same
   coalesced pattern as this one. Avoids a global transpose between
   passes.
5. **Store**: coalesced write back to global.

### 5.3 Two butterfly forms — algebraic inverses

We need two butterfly variants — one each direction — because they are
*algebraic inverses*, not the same operation with knobs. Earlier
versions of this section claimed otherwise; that claim was wrong. The
correction, verified by hand-tracing N=4 and confirmed against sppark
(which also ships both `CT_NTT` and `GS_NTT`):

- **Forward (CT-DIT)**: butterfly `(a, b, ω) → (a + ω·b, a − ω·b)`,
  stages with **ascending stride** (`1, 2, …, N/2`), forward twiddles
  `ω^k`. Bit-rev input → natural output.

- **Inverse (GS-DIF)**: butterfly `(a, b, ω) → (a + b, (a − b)·ω)`,
  stages with **descending stride** (`N/2, …, 1`), inverse twiddles
  `ω^{-k}`, plus an `×N⁻¹` scaling absorbed into the kernel by
  pre-multiplying the natural-evaluation input with `N⁻¹`. Natural
  input → bit-rev output.

Why these are inverses: running the CT-DIT graph backward is equivalent
to applying GS-DIF butterflies in reverse stage order. The `(a+ω·b, a−ω·b)`
butterfly's algebraic inverse is `((s+d)/2, (s−d)/(2ω))`. With `1/2`s
folded out to `×N⁻¹` at the end and a substitution `ω → ω⁻¹`, this
becomes exactly `(s+d, (s−d)·ω⁻¹)` — the GS-DIF form with inverse
twiddles.

Practical consequence: two `#[cube]` kernels, one each, with mostly
shared infrastructure (workgroup memory, comptime stage decomposition,
indexing). Per the §7 single-order convention they're each their
direction's only kernel — no NN/NR/RN/RR matrix.

## 6. Twiddle factors

sppark's partial-twiddle technique (`parameters.cuh:189-209`,
`kernels.cu:278-298`) is the right answer here.

Concretely for BabyBear (S = 27):

- Choose `LG_WINDOW = ceil(S / 5) = 6` ⇒ window size 64, 5 windows.
- Precompute on host: `partial[w][i] = ω^{i · 2^(w·LG_WINDOW)}` for
  `i ∈ [0, 64)`, `w ∈ [0, 5)`. Total: 320 field elements ≈ 1.3 KiB.
- For arbitrary `k ∈ [0, 2^27)`, decompose `k = Σ_w k_w · 2^(w·6)` and
  reconstruct `ω^k = Π_w partial[w][k_w]` — 4 multiplies.

This is dramatically smaller than a full `2^27`-entry table (512 MiB
for 32-bit elements). And critically, it generates twiddles
*inside the kernel*, eliminating a global twiddle-load that would
otherwise dominate WebGPU bandwidth.

For the inner kernel's first 5–6 stages we still want a fully
expanded `radix6_twiddles[32]` constant baked into the SPIR-V/PTX —
those are only 128 bytes per direction.

We re-implement (not depend on) Plonky3's caching pattern
(`radix_2_dit.rs:25-58`): an `Arc<RwLock<HashMap<(P, u32),
Arc<TwiddleSet>>>>` keyed on `(field, log_n)`, lazily filled, double-
checked locking. Coset twiddles deferred to a v2.

## 7. Coefficient-order convention

We commit to **one** memory order, eliminating sppark's NN/NR/RN/RR
optionality entirely. Software complexity matters more than letting
callers pick a layout, and the use cases in scope (STARK-flavored
workloads) are all happy with bit-reversed coefficients anyway.

**Convention:**
- **Coefficients live in bit-reversed memory order.**
- **Evaluations live in natural memory order.**
- **Forward NTT** (poly evaluation): bit-rev coeffs → natural evals.
  Implemented by iterating the radix-2 butterfly with stride descending
  from `N/2` to `1`. Equivalent to sppark's RN forward
  (`ntt.cuh:174-194`).
- **Inverse NTT** (poly interpolation): natural evals → bit-rev coeffs.
  Same butterfly, stride ascending from `1` to `N/2`, inverse twiddles,
  final ×`N⁻¹`. Equivalent to sppark's NR inverse.

**Consequences:**
- Zero bit-reversal kernel launches, ever. We don't even compile the
  `bit_reverse` kernel that sppark ships at `kernels.cu:16-129`.
- The public API has no `Order` parameter. Just `forward()` and
  `inverse()`.
- Callers that need natural-order coefficients (e.g. for output, or
  unusual interop) are responsible for bit-reversing themselves. We
  may ship a standalone `bit_reverse_inplace` helper, but it's not
  on the hot path and is not implicit in any other API.
- This does mean the Plonky3 oracle (§11) compares to a value we have
  to bit-reverse before checking. That's a one-time test-side
  permutation, fine.

## 8. Public API

```rust
pub trait Field: 'static + Copy + Send + Sync {
    const PRIME: u32;
    const MU: u32;
    const TWO_ADICITY: u32;
    fn two_adic_generator(log_n: u32) -> Self;
}

pub struct Plan<F: Field> { /* cached twiddles, kernel selection */ }

impl<F: Field> Plan<F> {
    pub fn new(log_n: u32) -> Self;

    /// Forward NTT (poly evaluation): bit-rev coeffs in → natural evals out.
    /// `data` shape: `[batch, 1<<log_n]`, row-major. In-place.
    pub fn forward<R: Runtime>(
        &self,
        client: &ComputeClient<R::Server, R::Channel>,
        data: &mut Tensor<R, F>,
    );

    /// Inverse NTT (poly interpolation): natural evals in → bit-rev coeffs out.
    pub fn inverse<R: Runtime>(...);
}
```

No `Order` enum, no bit-reversal helper on the hot path — see §7.

## 9. Batching strategy and memory layout

### 9.1 Memory layout

A batch of `B` polynomials of length `N = 2^log_n` lives in a tensor of
shape `[B, N]`, **polynomial-major, row-major**:

```
poly0[0], poly0[1], ..., poly0[N-1], poly1[0], poly1[1], ..., poly_{B-1}[N-1]
```

That is:
- **Stride 1**: consecutive elements of the same polynomial.
- **Stride N**: same index across different polynomials.

This is the GPU-friendly layout: each workgroup processes a single
polynomial, and its global-memory loads coalesce naturally because
adjacent threads in the workgroup hit adjacent `u32`s. It is also the
implicit layout in sppark — `NTT::Base` (`ntt.cuh:216-244`) takes a
single contiguous `inout` buffer of `1<<lg_domain_size` elements per
launch, so a sppark-style host loop over polynomials is operating on
exactly this layout, just one row at a time.

Plonky3 uses the *opposite* layout (column-as-polynomial; see §2.4),
which we explicitly do not adopt. If a caller needs to interop with
Plonky3 they transpose at the boundary.

### 9.2 Batching

sppark runs one transform per kernel launch (`ntt.cuh:216-244`). For
hundreds of 1M-point transforms, that is hundreds of launch overheads
on top of milliseconds of compute each — small but measurable, and
*especially* costly on WebGPU where each launch is a command-buffer
round-trip.

Plan: **grid-Y dimension carries the batch index**. Every kernel pass
is launched with `grid_dim = (num_blocks_per_poly, B, 1)`. Each
workgroup picks its polynomial via `grid.y`, then operates on the
contiguous `[grid.y * N, grid.y * N + N)` slab. Twiddle tables and
partial twiddles are batch-invariant — loaded once into
constant/register memory per workgroup, shared across all batch rows.

This collapses what would be `B × passes` kernel launches into just
`passes` (1–3 for our size range), independent of batch size.

## 10. CPU backend

We do **not** ship a Plonky3 runtime fallback. Reasons:

- Plonky3's API and internals assume natural-order coefficients in
  several places (§2.4); routing through it would force a bit-reversal
  pass we explicitly designed out (§7).
- It would split the type system — Plonky3's `BabyBear` and our
  `BabyBear` are different types with different traits, requiring
  conversion at the boundary.
- It would mean shipping ~10× more dependencies than we need.

The CPU path is **cubecl-cpu running the same `#[cube]` kernel** as the
GPU backends. cubecl-cpu is less mature than Plonky3's hand-tuned
AVX2/AVX-512 packed Montgomery (`monty-31/src/x86_64_avx2/packing.rs:360-446`),
so for now this is a correctness-and-CI path, not a "production CPU
NTT" path. If CPU throughput becomes important later, we revisit by
either (a) waiting for cubecl-cpu to vectorize or (b) writing a
direct-Rust path inside this crate, again *not* depending on Plonky3.

Plonky3 still appears in `dev-dependencies` — only as a test oracle
(§11).

## 11. Testing & validation

- **Unit-level field arithmetic.** Property tests: for random `a, b`,
  our `monty_mul` ≡ Plonky3's `BabyBear::mul`. Bit-exact. Same for
  `add`, `sub`, `inv`. Same for the lazy-form `monty_mul_lazy` after
  canonicalize.
- **Single-NTT correctness.** For `log_n ∈ [1, 16]`, random input,
  compare our forward NTT output against Plonky3 `Radix2Dit::dft`,
  with a bit-reverse permutation applied on one side to reconcile the
  ordering convention (§7). Since both libraries use the same
  Montgomery form, the comparison is byte-exact post-permutation.
- **Round-trip.** `inverse(forward(x)) == x` for all `log_n ∈ [1, 24]`.
  This needs no Plonky3 oracle and exercises the full size range
  including the 3-pass kernels.
- **Cross-backend.** Same input through CUDA, WGSL (Vulkan, Metal,
  WebGPU), and CPU backends must agree byte-for-byte.
- **Subgroup vs portable path agreement.** Force-disable subgroup ops
  via comptime flag and compare against subgroup-enabled output on
  the same backend.
- **Benchmarks.** Throughput in million-points-per-second for batch
  sizes \[1, 256\] and `log_n ∈ {16, 20, 24}`. Compared against:
  - sppark's CUDA (rebuild from `~/src/sppark`) — apples-to-apples
    on raw GPU NTT throughput.
  - Plonky3 CPU (`Radix2DitParallel`) — apples-to-apples on CPU.

## 12. Implementation phases

Suggested order, roughly one PR per phase:

1. **Field types.** `BabyBear`, `KoalaBear` as `#[derive(CubeType)]`,
   with `monty_mul`/`add`/`sub` and a CPU-side `Field` trait that
   re-exports Plonky3's. Property-test against Plonky3.
2. **Small NTT, single backend.** `log_n ≤ 10`, monolithic radix-N
   kernel, on cubecl-cpu only. NN order only. Validate against
   Plonky3.
3. **Twiddle precomputation + caching.** Partial-twiddle table on
   host, copy to device, reconstruct in kernel.
4. **Mixed-radix two-pass kernel.** `log_n ∈ [11, 20]`. Inner kernel
   on the **portable workgroup-memory butterfly path** first (works
   everywhere; simplest to debug). Forward only.
5. **iNTT direction.** Same butterfly, ascending stride, inverse
   twiddles, final ×N⁻¹. Round-trip test gates this phase.
6. **Three-pass kernel.** Extends to `log_n ∈ [21, 24]`.
7. **Grid-Y batching**.
8. **Subgroup-shuffle fast path.** `comptime`-gated, CUDA + Vulkan +
   Metal. Deliberately a separate phase from the portable path —
   correctness first, speed second.
9. **Lazy-reduction butterflies.** Drop `final_sub` on the inner
   stages, canonicalize at workgroup boundary. Bounds-analysis tests
   gate this phase.
10. **Benchmark, profile, tune** (`Z_COUNT`, workgroup size, twiddle
    layout). Compare against sppark on CUDA.

## 13. Open questions

- **cubecl subgroup features per backend.** Need to confirm cubecl's
  WGSL backend reliably exposes subgroup-shuffle on Apple Silicon
  (Metal), RDNA2+ (Vulkan), and Chromium-stable WebGPU. If subgroup
  support on Metal/Vulkan turns out to be flaky in cubecl, the
  portable workgroup-memory path is the default everywhere except
  CUDA — which is a tolerable but slower outcome.
- **`Z_COUNT` on WGSL.** sppark's per-thread vector of 32 elements
  (`ct_narrow.cu:5`) requires register space we may not have on
  mobile / integrated WebGPU. Probable starting point: `Z_COUNT = 4`
  on WGSL, `Z_COUNT = 32` on CUDA, `comptime`-selected.
- **Lazy-reduction bounds.** sppark's `sqr_n` skips `final_sub` every
  other iteration in a chain of squarings. The exact range bookkeeping
  in our butterfly's `(a, ω·b) ↦ (a + ω·b, a − ω·b)` pattern needs to
  be worked out before phase 9 — easier with property tests than from
  first principles.
- **Coset-LDE / extension-field NTT** — needed for STARK use, out of
  scope for v1. Architecture stays the same; just an extra pre/post
  scaling pass.

## 14. References (file:line)

**sppark**:
- `~/src/sppark/ntt/ntt.cuh:33` — `InputOutputOrder` enum
- `~/src/sppark/ntt/ntt.cuh:100-158` — pass decomposition
- `~/src/sppark/ntt/ntt.cuh:174-194` — per-pass direction selection (sppark; we don't need)
- `~/src/sppark/ntt/ntt.cuh:216-244` — public `NTT::Base` entry
- `~/src/sppark/ntt/kernels/ct_mixed_radix_narrow.cu:5-183` — narrow CT kernel
- `~/src/sppark/ntt/kernels/ct_mixed_radix_narrow.cu:160-167` — index rotation
- `~/src/sppark/ntt/kernels/ct_mixed_radix_narrow.cu:226-229` — coalesced launch
- `~/src/sppark/ntt/kernels/ct_mixed_radix_wide.cu:153-206` — wide kernel radices
- `~/src/sppark/ntt/kernels.cu:16-129` — bit-reversal kernel
- `~/src/sppark/ntt/kernels.cu:278-298` — `get_intermediate_roots`
- `~/src/sppark/ntt/parameters.cuh:189-209` — partial-twiddle layout
- `~/src/sppark/ntt/parameters/baby_bear.h:5-73` — BabyBear roots, S=27
- `~/src/sppark/ff/mont32_t.cuh:19-44` — `mont32_t` struct
- `~/src/sppark/ff/mont32_t.cuh:46-58,114-126` — `add` / `sub` (with `N==32` carry)
- `~/src/sppark/ff/mont32_t.cuh:176-194` — `final_sub`
- `~/src/sppark/ff/mont32_t.cuh:196-211` — Montgomery mul (PTX)
- `~/src/sppark/ff/mont32_t.cuh:255-355` — fused dot-product
- `~/src/sppark/ff/mont32_t.cuh:262-263` — `shfl_bfly`
- `~/src/sppark/ff/mont32_t.cuh:376-395` — `sqr_n`, lazy reduction

**Plonky3**:
- `~/src/Plonky3/dft/src/traits.rs:27-498` — `TwoAdicSubgroupDft` trait
- `~/src/Plonky3/dft/src/radix_2_dit.rs:25-58` — twiddle cache
- `~/src/Plonky3/dft/src/radix_2_dit.rs:72-78` — DIT main loop
- `~/src/Plonky3/dft/src/radix_2_dit_parallel.rs:139-228` — parallel six-step + coset
- `~/src/Plonky3/dft/src/butterflies.rs` — butterfly primitives
- `~/src/Plonky3/monty-31/src/utils.rs:99-158` — `monty_reduce`
- `~/src/Plonky3/monty-31/src/utils.rs:54-86` — `add`/`sub`
- `~/src/Plonky3/monty-31/src/monty_31.rs:681-689` — scalar `mul`
- `~/src/Plonky3/monty-31/src/x86_64_avx2/packing.rs:360-446` — packed AVX2 mul
- `~/src/Plonky3/baby-bear/src/baby_bear.rs:18-51` — BabyBear params
- `~/src/Plonky3/koala-bear/src/koala_bear.rs:21-76` — KoalaBear params
