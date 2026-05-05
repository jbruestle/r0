# r0-ntt — Batched NTT/iNTT over 31-bit fields, on GPU + CPU

A Rust library for batched Number-Theoretic Transforms over highly 2-adic 31-bit
prime fields (BabyBear, KoalaBear), portable across GPU and CPU via
[`cubecl`](https://github.com/tracel-ai/cubecl).

---

## 1. Scope

- **Fields**: BabyBear (`p = 2^31 − 2^27 + 1`, 2-adicity 27) and KoalaBear
  (`p = 2^31 − 2^24 + 1`, 2-adicity 24). Single `u32` in Montgomery form.
- **Sizes**: 2^k for k ∈ [1, 24]. Headline target: k = 20 (1M points).
  Hard ceiling: k = 24 (KoalaBear's 2-adicity).
- **Backends**: CUDA (cubecl-cuda), Vulkan/Metal/WebGPU (cubecl-wgpu), CPU (cubecl-cpu).
- **Non-goals**: large primes, LDE/coset (v2), MSM, PTX/inline-asm, multi-GPU.

## 2. Ordering convention

One fixed convention — no `InputOutputOrder` enum:

- **Coefficients**: bit-reversed memory order.
- **Evaluations**: natural memory order.
- **Forward NTT** (R→N): bit-rev coeffs in → natural evals out.
- **Inverse NTT** (N→R): natural evals in → bit-rev coeffs out.

No host-side permutations. No bit-reversal kernel. The multi-pass kernel
structure produces the correct output order directly (see §4).

## 3. Field arithmetic

Montgomery form: `x_mont = x · 2^32 mod p`. Three primitives:

```rust
#[cube] fn monty_mul<P>(a: u32, b: u32) -> u32;  // a*b*R^{-1} mod p
#[cube] fn monty_add<P>(a: u32, b: u32) -> u32;  // (a+b) mod p
#[cube] fn monty_sub<P>(a: u32, b: u32) -> u32;  // (a-b) mod p
```

All use 32-bit multiplies only (WGSL-portable). On CUDA, `mul_hi` compiles
to `mul.hi.u32`. On WGSL, emulated via schoolbook split (~10 ops vs 2).

## 4. Kernel architecture

### 4.1 Two dual kernels

The forward and inverse are perfect duals:

| | **Forward (CT-DIT)** | **Inverse (GS-DIF)** |
|---|---|---|
| Load | contiguous | tiled transposed gather |
| Butterflies | ascending stride, `(a,b,ω) → (a+ωb, a−ωb)` | descending stride, `(a,b,ω) → (a+b, (a−b)·ω)` |
| Store | tiled transposed scatter | contiguous |
| Twiddles | forward partial table | inverse partial table |
| Scaling | none | N^{-1} on first pass load |

The transposed store/load uses Z-tiling for coalescing: Z adjacent threads
(corresponding to Z adjacent workgroups) access Z consecutive global addresses.

### 4.2 Why always-transpose works

For the forward kernel, the transposed store writes element `(wg, local_j)` to
position `local_j * N_other + wg`. For the final pass, `N_other = N / N_pass`
and the natural output position of that element is `wg + local_j * N_other` —
identical. So the transposed store on the final pass IS the natural-order store.

For intermediate passes, the transposed store creates exactly the layout that
the next pass's contiguous load reads correctly from.

The inverse is the mirror: transposed LOAD on each pass gathers elements from
the layout that the previous pass's contiguous STORE created.

### 4.3 Multi-pass decomposition

```
log_n ≤ 10  : 1 pass  (in-place; N_other=1 makes transpose = identity)
11 ≤ log_n ≤ 20 : 2 passes
21 ≤ log_n ≤ 24 : 3 passes
```

Pass sizes balanced: `(log_n/k, log_n/k, ..., log_n/k + remainder)`.

Multi-pass uses ping-pong between two buffers (a→b→a→b...). Output lands in:
- buf_a for even pass count (2-pass)
- buf_b for odd pass count (1-pass, 3-pass)

### 4.4 Kernel parameters

```rust
#[cube(launch_unchecked)]
pub fn ntt_fwd_pass<P: MontyParameters>(
    input: &Array<u32>,
    output: &mut Array<u32>,        // must be separate from input for multi-pass
    partial_twiddles: &Array<u32>,  // windowed twiddle table
    #[comptime] log_n: u32,        // total transform size
    #[comptime] log_pass: u32,     // stages this pass handles
    #[comptime] stage_offset: u32, // global stage index where this pass begins
    #[comptime] log_wg: u32,       // workgroup size exponent
    #[comptime] z_count: u32,      // chunks per workgroup
);
```

Inverse has the same signature plus `inv_n: &Array<u32>` (N^{-1} scaling buffer).

### 4.5 Twiddle exponent computation

At butterfly stage `s` (global stage `stage_offset + s`):

- **Forward first pass** (stage_offset=0): `exp = j * 2^(log_n - 1 - s)`
- **Forward non-first pass**: `exp = (wg >> log_remaining) * outer_step + j * inner_step`
  where `log_remaining = log_n - stage_offset - log_pass`

The `>> log_remaining` correction accounts for prior transposes scrambling the
workgroup-to-global-position mapping. For the final pass, `log_remaining = 0`
so the shift is identity.

## 5. Windowed twiddle tables

Instead of a flat N/2-entry table (2MB for log_n=20), we use a compact
windowed partial table:

- **LG_WINDOW = 10**, window size = 1024 entries
- **NUM_WINDOWS = 3** (covers up to 30-bit exponents)
- **Table size**: 3 × 1024 = 3072 entries = 12 KiB

Reconstruction: decompose exponent `k` into 10-bit windows, look up each
window's entry, multiply together. For log_n ≤ 20: only 2 windows needed
(1 multiply per butterfly). For log_n 21-24: 3 windows (2 multiplies).

The 12 KiB table fits entirely in L1 cache on CUDA, making lookups essentially
free. This is faster than the original 2MB flat table (which spilled L1) while
using 170× less memory.

```rust
// Host-side construction
pub fn build_partial_fwd_twiddles<P: MontyParameters>(log_n: u32) -> Vec<u32>;
pub fn build_partial_inv_twiddles<P: MontyParameters>(log_n: u32) -> Vec<u32>;
```

Both forward and inverse share the same table structure — just different roots
(ω vs ω^{-1}).

## 6. Batching

Grid-Y dimension carries the batch index. Each kernel pass is launched with
`grid = (num_workgroups_per_poly / z_count, batch_size, 1)`. Twiddle table is
shared across all batch rows.

This collapses `B × passes` kernel launches into just `passes` (1–3).

## 7. Performance

CUDA, BabyBear, log_n=20 (1M points), z_count=8:

| Config | Time | Notes |
|--------|------|-------|
| Single NTT (ours) | 38 µs | 2-pass, windowed twiddles |
| Single NTT (sppark) | 27 µs | hand-PTX, warp shuffles |
| Batched 100x (ours) | 1.96 ms | grid-Y batching |
| Batched 100x (sppark) | 2.2 ms | sequential launches |

The remaining ~1.4× single-NTT gap vs sppark is:
- PTX codegen quality (~15-25%): cubecl emits ~10 ops per monty_mul vs 6
- Warp-shuffle butterflies: sppark exchanges via `__shfl_xor_sync` for inner
  stages, avoiding shared memory entirely. Not yet implemented.

## 8. File layout

```
crates/r0-field/src/
  lib.rs           -- re-exports
  monty.rs         -- MontyField<P>, monty_mul/add/sub, MontyParameters trait
  baby_bear.rs     -- BabyBearParameters (p = 0x78000001, S = 27)
  koala_bear.rs    -- KoalaBearParameters (p = 0x7f000001, S = 24)

crates/r0-ntt/src/
  lib.rs           -- public API exports
  fwd_pass.rs      -- forward kernel: contiguous load → DIT → transposed store
  inv_pass.rs      -- inverse kernel: transposed load → DIF → contiguous store
  twiddles.rs      -- host-side partial twiddle construction, bit-reversal helper
```

## 9. Testing

- **Single-pass oracle**: forward + inverse for log_n 1..10 against Plonky3's
  `Radix2Dit`, both BabyBear and KoalaBear, on CPU + wgpu.
- **Multi-pass oracle**: forward + inverse for log_n 11..20, against Plonky3,
  on wgpu + CUDA.
- **3-pass forward**: log_n=21, BabyBear, CUDA (exact match vs Plonky3).
- **Unit tests**: partial twiddle reconstruction verified exhaustively against
  flat table for all k in [0, N/2).

## 10. Future work

- **Warp-shuffle butterflies**: inner 5-6 stages via `__shfl_xor_sync` on CUDA,
  subgroup shuffles on Vulkan/Metal. Expected to close the single-NTT gap.
- **Lazy reduction**: skip `final_sub` in monty_mul between chained butterflies,
  canonicalize at workgroup boundaries only.
- **Coset-LDE**: pre/post scaling pass for STARK-style domain extension.
- **3-pass inverse**: untested (kernel supports it, test not yet written).
