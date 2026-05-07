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

Pass count and sizes are determined by the execution plan (see §6). The
default heuristic produces balanced splits:

```
log_n ≤ 10  : 1 pass  (in-place; N_other=1 makes transpose = identity)
11 ≤ log_n ≤ 20 : 2 passes
21 ≤ log_n ≤ 24 : 3 passes
```

The autotuner can explore asymmetric splits and different per-pass
parameters (z_count, log_wg) — see §6.

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

## 6. Planning and autotuning

Execution is split into three independent concerns:

### 6.1 NttPlan — pure data

```rust
pub struct PassConfig {
    pub log_pass: u32,      // stages this pass handles
    pub stage_offset: u32,  // cumulative offset
    pub z_count: u32,       // chunks per workgroup (must be po2)
    pub log_wg: u32,        // workgroup size exponent
}

pub struct NttPlan {
    pub log_n: u32,
    pub passes: Vec<PassConfig>,
    pub sub_batch: usize,   // NTTs per kernel launch group
}
```

A plan is serializable, hand-constructible, and device-agnostic. It can come
from the heuristic, the autotuner, or a saved file.

### 6.2 NttExec — resource owner and executor

```rust
pub struct NttExec<P: MontyParameters, R: Runtime> { /* client, twiddles, scratch */ }

impl NttExec {
    pub fn new(device: &R::Device, scratch_bytes: usize) -> Self;
    pub fn limits(&self) -> &DeviceLimits;
    pub fn forward(&self, buf: &Handle, plan: &NttPlan, batch: usize);
    pub fn inverse(&self, buf: &Handle, plan: &NttPlan, batch: usize);
    pub fn forward_auto(&self, buf: &Handle, log_n: u32, batch: usize);
    pub fn inverse_auto(&self, buf: &Handle, log_n: u32, batch: usize);
}
```

Queries `DeviceLimits` (shared memory, max threads) from cubecl at
construction time. The `_auto` methods call `plan_heuristic` internally.

### 6.3 Planning strategies

**Heuristic** (`plan_heuristic`): tries 1–4 pass balanced splits, picks the
best by a scoring function that weights log_wg (dominant), pass count, and
z_count. Zero device interaction, instant. Suitable as a default.

**Autotuning** (`enumerate_valid_plans` + benchmarking): generates all
constraint-valid plans (pass decompositions × z_count × log_wg per pass),
sorts by heuristic score, benchmarks in that order. The heuristic sort
ensures the best plans are found early — exhaustive search is optional.

### 6.4 Constraints

| Constraint | Formula |
|---|---|
| Shared memory | `z_count × 2^log_pass × 4 ≤ max_shared_mem_bytes` |
| Threads per WG | `2^log_wg ≤ max_threads_per_wg` |
| Min elements/thread | `log_wg ≤ log_pass − 1` |
| Grid divisibility | `z_count ≤ 2^(log_n − log_pass)` |
| Scratch | `2^log_n × sub_batch × 4 ≤ scratch_bytes` (multi-pass only) |
| z_count | must be a power of two |

`validate_plan()` checks all of the above and returns detailed errors.

### 6.5 Autotuning results

CUDA, BabyBear, log_n=20, batch=32 (2320 2-pass plans scanned):

- **Best plan**: `(lp=10 z=8 wg=9) → (lp=10 z=4 wg=9)`, **611 µs** (~19 µs/NTT)
- **Heuristic pick**: `(lp=10 z=8 wg=9) → (lp=10 z=8 wg=9)`, 613 µs (within 0.3%)
- **Old hardcoded**: `(lp=10 z=8 wg=8) → (lp=10 z=8 wg=8)`, ~775 µs

Key findings:
- **log_wg is the dominant knob**: 5× improvement from wg=6 to wg=9. Always
  use the largest value the device supports.
- **z=4-8 beats z=16**: lower shared memory usage improves occupancy (more
  concurrent workgroups). The heuristic caps z at 8.
- **Performance landscape is flat near the optimum**: top-20 plans are within
  7% of each other. The heuristic only needs to be roughly right.
- **Heuristic sort is effective**: best plan found at position 20 out of 2320.

## 7. Batching

Grid-Y dimension carries the batch index. Each kernel pass is launched with
`grid = (num_workgroups_per_poly / z_count, batch_size, 1)`. Twiddle table is
shared across all batch rows.

This collapses `B × passes` kernel launches into just `passes` (1–3).
For multi-pass NTTs, the sub-batch size determines how many polynomials are
processed per ping-pong iteration, bounded by scratch memory.

## 8. Performance

BabyBear, log_n=20 (1M points), batch=32:

| Backend | Time | Per-NTT | Throughput |
|---------|------|---------|------------|
| CUDA (autotuned) | 611 µs | 19 µs | 54.1 Gelem/s |
| CUDA (heuristic) | 613 µs | 19 µs | 54.0 Gelem/s |
| Metal/wgpu (M-series Mac) | 2.29 ms | 72 µs | 14.6 Gelem/s |
| sppark (hand-PTX) | ~860 µs* | 27 µs | — |

\* sppark estimated from 27 µs single × 32, since it launches sequentially.

**CUDA vs Metal gap (~3.7×)** is largely explained by WGSL `mul_hi`
emulation: ~5× more ALU ops per `monty_mul` vs native CUDA `mul.hi.u32`.
Metal MSL has native `mulhi` but cubecl currently emits WGSL for the wgpu
backend regardless of the underlying GPU API.

**CUDA vs sppark gap** on single-NTT (~32 µs vs 27 µs = 1.2×):
- PTX codegen quality: cubecl emits ~10 ops per monty_mul vs 6
- Warp-shuffle butterflies: sppark uses `__shfl_xor_sync` for inner stages

## 9. File layout

```
crates/r0-field/src/
  lib.rs           -- re-exports
  monty.rs         -- MontyField<P>, monty_mul/add/sub, MontyParameters trait
  baby_bear.rs     -- BabyBearParameters (p = 0x78000001, S = 27)
  koala_bear.rs    -- KoalaBearParameters (p = 0x7f000001, S = 24)

crates/r0-ntt/src/
  lib.rs           -- public API exports
  plan.rs          -- NttPlan, PassConfig, DeviceLimits, validate, heuristic, enumerate
  exec.rs          -- NttExec: device resources, kernel launching
  fwd_pass.rs      -- forward kernel: contiguous load → DIT → transposed store
  inv_pass.rs      -- inverse kernel: transposed load → DIF → contiguous store
  twiddles.rs      -- host-side partial twiddle construction, bit-reversal helper
```

## 10. Testing

All tests gated behind feature flags (`cuda`, `wgpu`, `cpu`).

- **Oracle sweep** (tests/p3_oracle.rs): forward + inverse against Plonky3's
  `Radix2Dit` for log_n 1..=24 on CUDA and wgpu, BabyBear full sweep +
  KoalaBear spot-check at 20. Covers 1-pass, 2-pass, and 3-pass (21-24).
- **Batch-size sweep**: roundtrip at log_n=20, batch sizes [1,2,3,5,7,16,17,
  32,33,100] on CUDA and [1..17] on wgpu. Exercises sub_batch remainder
  handling (sub_batch=16 with default 64 MiB scratch).
- **CPU oracle**: forward + inverse at log_n=10 (ignored by default, slow).
- **Plan validation**: 20 unit tests for constraint checking, heuristic
  properties, z_count/split math, enumeration.
- **Autotune**: full 2-pass parameter scan on CUDA and wgpu (tests/autotune.rs,
  ignored by default).
- **Diagnostics**: adapter enumeration and device limit dump (tests/diagnostics.rs).
- **Twiddle unit tests**: partial twiddle reconstruction verified exhaustively
  against flat table for all k in [0, N/2).

## 11. Future work

- **WebGPU browser target**: test wgpu path in actual browser environment.
- **Autotune persistence**: save/load best plans per (device, log_n, batch).
- **3-pass autotune**: needs smarter search (enumeration combinatorics too
  large for exhaustive scan).
- **Native MSL mulhi**: cubecl currently emits WGSL schoolbook emulation for
  `mul_hi` even on Metal. Native MSL `mulhi` would close ~3× of the
  CUDA-vs-Metal gap.
- **Warp-shuffle butterflies**: inner 5-6 stages via `__shfl_xor_sync` on CUDA,
  subgroup shuffles on Vulkan/Metal.
- **Lazy reduction**: skip `final_sub` in monty_mul between chained butterflies,
  canonicalize at workgroup boundaries only.
- **Coset-LDE**: pre/post scaling pass for STARK-style domain extension.
