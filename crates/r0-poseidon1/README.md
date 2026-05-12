# r0-poseidon1

Poseidon1 permutation over KoalaBear, width 16, as a set of `#[cube]`
subroutines callable from any cubecl kernel. Single source compiles to
host Rust, CUDA, WGSL (wgpu / Vulkan / Metal / WebGPU), and the cubecl
CPU emulator. Bit-for-bit compatible with Plonky3's
[`default_koalabear_poseidon1_16`](../../../Plonky3/koala-bear/src/poseidon1.rs).

This document is the design and implementation reference.

## 1. Scope

In:

- **Permutation** over `[KoalaBear; 16]` — raw Montgomery `u32`s in
  cubecl IR; `MontyField<KB>` on host.
- **Three call modes** sharing a common round structure:
  1. Pure permutation (compute mode).
  2. Permutation + per-S-box witness write (witgen mode).
  3. Per-row constraint contribution into a fiat-shamir-mixed
     accumulator (constraint mode, KB-state with KB^4 accumulator).
- **Host shadow** of the constraint mode for use as a test oracle and
  for any host-side use that wants the same algorithm.

Out (deferred until needed):

- Other widths (24).
- BB base field (same shape, different RC/MDS).
- Poseidon2.
- Sponge/duplex wrappers, Merkle tree drivers.
- Pure-Ext4 host verifier (the OOD-evaluation case where input/witness
  are KB^4 throughout). The current `host_constraint_kb_witness` covers
  the test-shadow case; an Ext4-throughout version is a small variant
  that adds when the verifier needs it.

## 2. Algorithm — Plonky3-compatible Poseidon1 KB16

| Parameter | Value |
|---|---|
| Width | 16 |
| S-box | `x^3` (`gcd(3, p_KB - 1) = 1` makes this an injective monomial) |
| Initial full rounds | 4 |
| Partial rounds | 20 |
| Terminal full rounds | 4 |
| Total rounds | 28 |
| Round constants | [`ROUND_CONSTANTS_CANONICAL`](src/host_ref.rs) (Grain LFSR, matches Plonky3) |
| MDS | Circulant, first column [`MDS_CIRC_COL_CANONICAL`](src/host_ref.rs) = `[1, 3, 13, 22, 67, 2, 15, 63, 101, 1, 2, 17, 11, 1, 51, 1]` |

Per round: AddRC → S-box (all 16 slots in full rounds, slot 0 only in
partial rounds) → MDS multiply.

**Oracle test vector** (matches Plonky3 + leanMultisig):

```text
input  = [0, 1, 2, …, 15]
output = [610090613, 935319874, 1893335292, 796792199,
          356405232,  552237741, 55134556,   1215104204,
          1823723405, 1133298033,1780633798, 1453946561,
          710069176,  1128629550,1917333254, 1175481618]
```

This is the load-bearing correctness check — every variant (host
serial, history-formulation host, FFT-MDS host, cube compute, cube
witgen) reproduces it.

## 3. Three call modes via the SBox-as-the-only-difference principle

The 28-round walk (AddRC → sbox slot(s) → MDS) is structurally identical
across the three modes. What differs is *only* the per-S-box primitive:

```text
compute:    sbox(x) = x³
witgen:     sbox(x) = let r = x³; write r to wit[col·stride + row]; r
constraint: sbox(x) = let w = read wit[col·stride + row];
                       acc += alpha_pow · (x³ - w);
                       alpha_pow *= alpha;
                       w
```

The constraint variant returns `w` (the witness value), not `x³`. Using
the witness keeps the rolling state computation linear in the unknown
— easier to reason about, and matches what the verifier sees.

**In practice**: cubecl 0.9's `CubeType` derive doesn't accept fields
with lifetimes or `Array<u32>` references, so a `trait SBoxOp { type
Ctx: CubeType; }` can't bundle "the witness buffer". The shipped shape
is **three top-level `#[cube] fn`s sharing mode-agnostic helpers**
(`bt`, `dit`, `neg_dif`, `mds_fft_16`). The mode-specific bit is the
per-S-box call site (~5 lines per variant inside per-round helpers).

## 4. State representation

Per-thread 16-element working set lives in `Array::<u32>::new(16)`
allocated inside the `#[cube] fn`. Verified by the spike at
[`r0-cube/tests/array16_spike.rs`](../r0-cube/tests/array16_spike.rs):
lowers to `var a_0: array<u32, 16>` in WGSL with all indices
comptime-resolved, which downstream compilers (Naga → Metal/SPIR-V)
scalarize into register file.

Caller composition: the caller allocates its own `Array::<u32>::new(16)`,
fills it from wherever (input buffer, prior permutation output,
computed values), and passes `&mut state` in. The permutation mutates
in place.

## 5. MDS path — FFT for full rounds

Circulant 16×16 multiply via the convolution theorem:

```text
C · x = DIT_FFT( λ ⊙ DIF_IFFT(x) )
```

Where λ are precomputed eigenvalues with the 1/16 inverse-FFT scaling
absorbed (`λ = DIF_IFFT(MDS_CIRC_COL) · 16⁻¹`).

Per MDS multiply: ~17 + 16 + 17 = **~50 monty_muls**, vs ~256 for naive
matvec. Across the 8 full rounds that's 400 vs 2048 muls — half the
permutation's total multiply budget.

The FFT butterflies and twiddles match leanMultisig's `dif_ifft_16_mut` /
`dit_fft_16_mut`. Implementation is a fully-unrolled chain of
`bt(lo, hi)` and `neg_dif(lo, hi, twiddle)` / `dit(lo, hi, twiddle)`
operations on the 16-element local array. Twiddles `ω^1..ω^7` and the
λ table are baked into the shader via `Array::<u32>::from_data`.

Cross-checked: `host_mds_fft` matches `mds_naive` on random inputs and
on the unit-basis check `e_0 → MDS_CIRC_COL` (see `tests/mds.rs`).

## 6. Partial-round optimization — history-of-16+r

Standard "sparse decomposition" of partial rounds (Plonky3, leanMultisig)
precomputes a dense `m_i` transformation plus per-round `(v[r], w_hat[r])`
such that each partial round is O(width). It works but introduces a
256-mul dense init step and lives in a transformed basis where
intermediate state isn't directly the "true" Poseidon state — making
it unusable for the constraint engine which needs per-round S-box values.

The **history-of-16+r** formulation captures the same linearity insight
in a cleaner, mode-agnostic form: every partial-round S-box input is a
linear combination of `(s_full_end[16] ++ partial_sbox_outputs[r])`.

```text
# After 4 initial full rounds, state is `s_full_end` (16 KB values).
history = [s_full_end[0..16], 0, 0, …, 0]   # 36 slots; tail zero

for r in 0..20:
    pre_sbox = Σ_{k=0..16+r} weights[r][k] · history[k]   + offset[r]
    sbox_out = SBOX(pre_sbox)                              # mode-specific
    history[16 + r] = sbox_out

# Recover 16-element state for terminal full rounds:
for i in 0..16:
    state[i] = Σ_{k=0..36} terminal_weights[i][k] · history[k]   + terminal_offset[i]
```

`weights[r]` and `terminal_weights[i]` and the offsets are precomputed
host-side from the MDS matrix and the partial-round AddRC constants
(see [`src/partial.rs`](src/partial.rs)). The AddRC additions get
folded into the dot product: each `weights[r]` absorbs the appropriate
scalar contributions from prior rounds' AddRC constants.

Op count for the partial phase: `Σ_{r=0..20}(16 + r)` + 20·2 + 16·36 ≈
**1130 muls**. Comparable to standard sparse decomposition (~900 muls)
but works uniformly for compute, witgen, and constraint modes without
basis transformations.

The derivation is cross-checked against the naive serial reference:
`tests/precompute.rs` runs `host_permute_via_history` (uses the
precomputed weights) against `host_permute` (naive matvec) on the
oracle vector and 32 random inputs.

## 7. Witness layout — round-major

148 sbox-output columns total per permutation:

| Range | Cols | Contents |
|---|---|---|
| `[0, 64)` | 64 | 4 initial full rounds × 16 slots, round-major then slot-major |
| `[64, 84)` | 20 | 20 partial rounds × slot 0 |
| `[84, 148)` | 64 | 4 terminal full rounds × 16 slots, round-major then slot-major |

Each column `c` is stored at `witness[(witness_col_base + c) · stride + row]`
in transposed layout (column-contiguous across rows). Same convention as
`r0-field`'s `ExtField::store`.

The witgen kernel writes only the 148 sbox values. The 16 inputs are
the caller's responsibility (already in their input buffer); the 16
outputs are recoverable from the last 16 sbox values via one MDS
multiply, so the witness doesn't store them either.

## 8. Public API

### 8.1 Permutation (compute mode)

```rust
#[cube]
pub fn poseidon1_kb16_permute(state: &mut Array<u32>);
```

Caller allocates `Array::<u32>::new(16)`, fills with raw Montgomery
KB values, calls, reads results.

### 8.2 Permutation with witness (witgen mode)

```rust
#[cube]
pub fn poseidon1_kb16_permute_with_witness(
    state: &mut Array<u32>,
    witness: &mut Array<u32>,
    witness_col_base: u32,
    row: u32,
    stride: u32,
);
```

In place over `state`. Additionally writes 148 sbox outputs in
round-major layout (see §7).

### 8.3 Constraint contribution

```rust
#[derive(CubeType, Copy, Clone)]
pub struct ConstraintAccumulator {
    pub alpha:     Ext4<KoalaBear4Parameters>, // mixing param (read-only)
    pub acc:       Ext4<KoalaBear4Parameters>, // running Σ α^i · diff_i
    pub alpha_pow: Ext4<KoalaBear4Parameters>, // current α^k
}

#[cube]
pub fn poseidon1_kb16_constraint(
    input_state: &Array<u32>,
    witness: &Array<u32>,
    witness_col_base: u32,
    row: u32,
    stride: u32,
    cstate: ConstraintAccumulator,
) -> ConstraintAccumulator;
```

Caller seeds `cstate.alpha_pow` with the desired starting α-power and
threads through subsequent constraint subroutines via shadow-let
chaining: `let cstate = poseidon1_kb16_constraint(…, cstate);`.

### 8.4 Host shadow

```rust
pub fn host_constraint_kb_witness(
    input_state: &[KoalaBear; 16],
    witness:     &[KoalaBear; N_WITNESS_SBOXES],
    cstate: ConstraintAccumulator,
) -> ConstraintAccumulator;
```

Plain Rust mirror of the cube path: KB rolling state, KB witness, KB^4
accumulator with diff lifted to KB^4 before mixing. Used as the cube
test oracle and available to any host-side caller that wants the same
algorithm. (An Ext4-throughout variant for the OOD-evaluation verifier
case is straightforward to add when needed.)

### 8.5 Other host-side surface

- [`host_permute`] — naive 28-round serial walk; the canonical reference.
- [`host_permute_with_trace`] — same as above, additionally returns the
  148 sbox-output trace in round-major layout.
- [`host_permute_via_history`] — history-of-16+r partial rounds; matches
  `host_permute` on every input (cross-check for the precompute).
- [`host_mds_fft`], [`mds_naive`] — both MDS paths exposed for comparison.
- [`dif_ifft_16`], [`dit_fft_16`] — host-side FFT halves.

## 9. Constants story

All compile-time-fixed:

- **28 × 16 round constants** (KB Montgomery form) — generated host-side
  at IR build time via `Array::<u32>::from_data(comptime!(rc_montgomery_flat()))`.
- **16 FFT λ eigenvalues** (with 1/16 absorbed) — same.
- **8 twiddle powers** `ω^0..ω^7` — same.
- **Partial-round weights** — 20 × 36 weights + 20 offsets + 16 × 36
  terminal weights + 16 terminal offsets. All baked via `from_data`.

`from_data` lowers to a WGSL module-scope `const arrays_N: array<u32, M>
= array(...)`; reads with constant indices fold to literals downstream.
Total baked constants: ~1750 u32s ≈ 7 KiB per kernel — trivially small.

## 10. Testing

Twelve tests across six files; all run on wgpu (the workspace's default
testable backend) and all pass.

| File | Tests | Coverage |
|---|---|---|
| `tests/oracle.rs` | 1 | `host_permute([0..15])` matches Plonky3 |
| `tests/precompute.rs` | 2 | history-formulation == naive on oracle + 32 random |
| `tests/mds.rs` | 2 | FFT MDS == naive on 32 random + e_0 → MDS column |
| `tests/permute_cube.rs` | 2 | cube compute matches host on oracle + 32 random |
| `tests/witgen_cube.rs` | 1 | cube witgen outputs + 148 sbox cols match host trace |
| `tests/constraint_cube.rs` | 3 | valid → 0; flipped → ≠0; cube == host shadow |

Plus two cubecl-pattern probe spikes at
`tests/cubetype_mut_ext_spike.rs` (this crate) and
[`r0-cube/tests/array16_spike.rs`](../r0-cube/tests/array16_spike.rs) /
[`r0-cube/tests/cubetype_mut_spike.rs`](../r0-cube/tests/cubetype_mut_spike.rs)
documenting the cubecl 0.9 limits that shaped this crate's API.

CPU runtime is not exercised: cubecl's CPU emulator reports
`plane_size = 1` which the rest of the workspace's tests already work
around; we don't add it back here unless we hit a wgpu-only codegen
issue.

## 11. Performance (Mac wgpu, M-series)

Throughput at 2^18 perms (criterion median, `cargo bench --features wgpu -p r0-poseidon1`):

| Mode | Time / call | Per-perm | Estimated muls/perm |
|---|---|---|---|
| permute    | 2.08 ms | 7.9 ns  | ~1780 |
| witgen     | 2.17 ms | 8.3 ns  | ~1780 (+ 148 cached writes) |
| constraint | 5.85 ms | 22.3 ns | ~5330 |

Mul-count breakdown (per perm):

- **Permute** ≈ 400 (8 × FFT MDS) + 256 (8 × 16 cubes) + 40 (20 partial cubes) + 510 (partial dot products) + 576 (state recovery) = **~1780**.
- **Witgen** = same muls, +148 transposed writes (~5% measured overhead — basically free, confirming compute-bound).
- **Constraint** ≈ 400 (MDS, same) + 8·16·26 + 20·26 (per-S-box: 2 cube + 4 lift-mul + 20 α-update) + 510 + 576 = **~5330**.

The constraint/permute ratio (~2.81× measured vs ~2.99× predicted) confirms
the WGSL compiler folds the literal-zero limbs in `lift(diff) = (diff, 0, 0, 0)`
from a full 20-mul `ext4_mul` down to ~4 muls. The remaining bottleneck is
`alpha_pow *= alpha` at every S-box (20 muls × 148 = ~56% of constraint
total). Skippable via a 148-element precomputed α-power table if
constraint perf becomes critical.

CUDA / sppark numbers TBD — `r0-poseidon1` builds against `--features cuda`
identically; benches just need an NVIDIA host.

## 12. Deviations from DESIGN.md

The shipped implementation differs from
the original `DESIGN.md` (replaced by this README) in two places. Both
are workarounds for cubecl 0.9 constraints that surfaced during
implementation; documented here so the diff is first-class.

1. **Constraint subroutine takes `cstate` by value and returns it,
   instead of `&mut ConstraintAccumulator`.** cubecl 0.9 supports
   field assignment on `&mut <CubeType>` only when the field type is
   itself a `CubePrimitive` (u32 etc.). For CubeType-typed fields like
   `Ext4` the macro errors with `From<…Expand>` trait-bound failures
   referencing unrelated cubecl primitives. The
   [`tests/cubetype_mut_ext_spike.rs`](tests/cubetype_mut_ext_spike.rs)
   probe documents this. Caller pattern:
   `let cstate = poseidon1_kb16_constraint(…, cstate);` — one extra
   reassignment per call vs the originally-planned `&mut`. Functionally
   equivalent; mutation through nested helper calls works the same.

2. **Per-round S-box chains use comptime-recursive `#[cube] fn`s
   instead of `for ... { cs = helper(cs) }` reassignment.** Same
   underlying limitation: `let mut cs: ConstraintAccumulator = …; cs =
   …;` reassignment also doesn't work for CubeType-with-CubeType-fields.
   The shipped pattern is `if comptime!(i >= N) { cs } else { let cs =
   step(cs); recurse(cs, comptime!(i + 1)) }`. cubecl resolves the
   recursion at IR build time, generating an inlined chain of N calls.
   Spike-verified pattern. See `full_round_chain` and `partial_chain`
   in [`src/constraint.rs`](src/constraint.rs).

Neither deviation changes the public API meaningfully — point 1 means
caller writes `cstate = f(…, cstate)` instead of `f(…, &mut cstate)`,
point 2 is purely internal.

## 13. File layout

```
src/
  lib.rs            -- module re-exports + crate-level docs
  host_ref.rs       -- naive 28-round walk (Plonky3 oracle); RC + MDS column
                       canonical constants; lifted-form helpers; host_permute
                       and host_permute_with_trace
  partial.rs        -- partial-round weight precomputation; host_permute_via_history
  mds.rs            -- DIF/DIT FFT halves, lambda eigenvalues, host_mds_fft
  permute.rs        -- poseidon1_kb16_permute and ..._with_witness (cube)
  constraint.rs     -- ConstraintAccumulator, poseidon1_kb16_constraint (cube),
                       host_constraint_kb_witness (host shadow)
tests/
  oracle.rs                     -- Plonky3 oracle [0..15] → expected
  precompute.rs                 -- history formulation == naive
  mds.rs                        -- FFT MDS == naive
  permute_cube.rs               -- cube compute on oracle + random batch
  witgen_cube.rs                -- cube witgen sbox-by-sbox vs host trace
  constraint_cube.rs            -- valid → 0; flipped → ≠0; cube == host
  cubetype_mut_ext_spike.rs     -- cubecl 0.9 probe (documented limitation)
```

## 14. Crate dep direction

```
r0-poseidon1 — → r0-field, r0-cube, cubecl
```

Sibling to r0-polynomial. Single backend feature flag (`cuda` or
`wgpu`), forwarded through `r0-cube`.

## 15. Future work (deferred)

- **Pure-Ext4 host verifier** for the OOD-evaluation case (input/witness
  in KB^4 throughout). Trivial variant of `host_constraint_kb_witness`;
  add when the verifier needs it.
- **Width 24** (same shape, different RC + RP=23).
- **BB base field** (same shape, different RC + W=11/2 for extensions).
- **Sponge wrapper** for variable-length absorption.
- **Merkle-tree compression-mode wrapper**: `compress(left, right) =
  permute(left ++ right)[..left.len()] + left`.
- **Performance pass** on the FFT butterflies (potential in-register
  twiddle precomputation, fusion across rounds).
- **Partial-round perf**: 256-mul shave possible by switching to the
  `m_i + sparse_v + sparse_w_hat` formulation if witness/constraint
  paths can co-exist with it (currently they can't because of the
  basis-transformation issue).
