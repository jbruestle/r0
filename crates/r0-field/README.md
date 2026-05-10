# r0-field

31-bit Montgomery prime fields for [BabyBear](https://eprint.iacr.org/2024/1037)
and [KoalaBear](https://github.com/Plonky3/Plonky3), their degree-4/5 binomial
extensions, and the abstractions higher-level cubecl kernels use to stay
generic over the choice. Single-source `#[cube]` arithmetic — one definition
compiles to host Rust, CUDA, WGSL, and the cubecl CPU backend.

This document is the design and implementation reference. For API usage see
the rustdoc.

## 1. Scope

- **Base fields**: BabyBear (`p = 2^31 − 2^27 + 1`, 2-adicity 27) and
  KoalaBear (`p = 2^31 − 2^24 + 1`, 2-adicity 24). One `u32` in Montgomery
  form per element.
- **Binomial extensions**: `BabyBear4` (`X^4 − 11`), `KoalaBear4` (`X^4 − 3`),
  `BabyBear5` (`X^5 − 2`). `W` constants match Plonky3's
  `BinomialExtensionData<D>` for the same base.
- **`ExtField` trait**: lets kernels (e.g. polynomial division) be generic
  over the inner field, with `BaseElem<P>` as the degree-1 bridge so the
  same kernel handles base and extension uniformly.
Out of scope: non-binomial extensions (KoalaBear^5 doesn't exist as a
binomial extension since `gcd(5, p_KB − 1) = 1`); we'd reach for trinomials
if the security story ever demanded it.

## 2. Field zoo

| Type | Base | `D` | `W` | Source |
|---|---|---|---|---|
| `BabyBear` | — | 1 | — | `MontyField<BabyBearParameters>` |
| `KoalaBear` | — | 1 | — | `MontyField<KoalaBearParameters>` |
| `BabyBear4` | BB | 4 | 11 | `Ext4<BabyBear4Parameters>` |
| `KoalaBear4` | KB | 4 |  3 | `Ext4<KoalaBear4Parameters>` |
| `BabyBear5` | BB | 5 |  2 | `Ext5<BabyBear5Parameters>` |

Each parameter set is a zero-sized marker type; arithmetic happens in
`MontyField<P>` (base) or `Ext4<P>` / `Ext5<P>` (extensions). Construct
from canonical limbs via `from_canonical(x: u32)` / `from_canonical([u32; D])`,
read back with `to_canonical()`. All limbs are stored in Montgomery form;
host operator overloads forward to the same `#[cube]` free functions
(`monty_add/sub/mul/neg`, `ext4_add/sub/mul/neg`, `ext5_…`) that kernels
call directly on raw `u32`s.

Base-field constants and the extensions' `W_MONT` are cross-checked
against Plonky3 in `tests/ext_p3_oracle.rs`.

## 3. The `ExtField` abstraction

Some kernels — NTT — only need base-field arithmetic. Others — polynomial
division, evaluation, FRI folding — care about the polynomial structure
but not whether the coefficients are in `F_p` or `F_{p^4}`. Those kernels
take `<F: ExtField>`:

```rust
#[cube]
pub trait ExtField: CubeType + Copy + Clone + … {
    type Base: MontyParameters;
    const DEGREE: u32;

    fn add/sub/mul/neg(…) -> Self;
    fn zero() -> Self;
    fn one()  -> Self;
    fn from_base_raw(x: u32) -> Self;

    /// Read element `i` from a transposed-layout buffer of `n` logical
    /// elements, starting at u32 offset `base`.
    fn load(arr: &Array<u32>, base: u32, i: u32, n: u32) -> Self;
    fn store(arr: &mut Array<u32>, base: u32, i: u32, n: u32, v: Self);
}
```

Three impls ship: `BaseElem<P>` (degree 1), `Ext4<P>` (degree 4), `Ext5<P>`
(degree 5). `BaseElem<P>` is a `u32` newtype tagged by its parameters —
zero-cost; the only reason it exists is so a single `<F: ExtField>` kernel
covers both base and extension polynomials.

### 3.1 Memory layout: transposed

A length-`N` extension polynomial occupies `D · N` u32s with **component
`c` of element `i` at offset `c · N + i`**. So a `BabyBear4` polynomial of
length `2^20` is bitwise identical to four contiguous `BabyBear`
polynomials of length `2^20`.

This layout buys two things at once:

- **Free extension NTT**: the four base sub-polynomials sit at consecutive
  NTT batch rows, so `NttExec::forward(buf, log_n, batch * D)` *is* the
  extension NTT — no new kernel, no permutation pass.
- **Coalesced GPU loads**: 32 warp threads reading element-`i` for varying
  `i` get `D` separate coalesced 32-byte loads (one per component) instead
  of `D` strided non-coalesced loads.

`ExtField::load` / `store` carry `base` (the polynomial's u32 offset) and
`n` (the stride) so the kernel folds in batch indexing cleanly.

## 4. cubecl 0.9 quirks worth knowing

The crate works around these in-place; they bite anyone writing new
`#[cube]` code in the workspace, so they're worth being aware of.

- **`u32 % u32` is broken on Metal.** cubecl lowers it to an ambiguous
  `metal::select(...)` call that fails compilation through wgpu. We never
  use `%` in cube bodies — Montgomery reduction is conditional subtract
  (`if x >= p { x - p } else { x }`).
- **`u32::mul_hi` panics on host.** Bridged via the `mul_hi_u32`
  function-plus-sister-module pattern in `monty.rs`; the cubecl macro
  resolves the call differently in host vs cube context.
- **No `From<u64> for ConstantValue` in cubecl 0.9.** All `u64` in cube
  bodies must come from a runtime `as u64` cast on a `u32` local.
- **Trait-const defaults referencing `Self::OTHER_CONST` aren't
  reachable through generic bounds (E0790).** Why `MontyParameters::MONT_ONE`
  is a required impl const — each impl spells out the same
  `(((1u64 << 32) % Self::PRIME as u64) as u32)`.
- **`PhantomData<P>` inside a `CubeType`-derived struct needs the
  `#[cube(comptime)]` attribute** so cubecl knows it carries no IR data.

## 5. Testing

- **`tests/ext_p3_oracle.rs`** — proptest cross-check vs Plonky3's
  `BinomialExtensionField<F, D>` for BB^4 / KB^4 / BB^5 (add, sub, mul,
  neg, distributivity, canonical round-trip, `W_MONT` consistency).
- **`tests/cube_smoke.rs`** — host-vs-kernel agreement for the base
  `monty_mul` `#[cube]` function on CPU and wgpu backends.
- **`tests/ext_cube_smoke.rs`** — generic `<F: ExtField>` kernel running
  `(a + b) * b` element-wise on transposed-layout buffers, for all five
  field instances on CPU and wgpu. Catches both dispatch bugs and
  load/store-layout bugs in one go.
- **Lib unit tests** — `MONT_ONE` consistency, `BaseElem<P>` zero-cost
  size assertion.

## 6. File layout

```
src/
  lib.rs        -- module re-exports + crate-level docs
  monty.rs      -- MontyParameters trait, MontyField<P>, monty_* #[cube] fns
  baby_bear.rs  -- BabyBearParameters + BabyBear4Parameters + BabyBear5Parameters
  koala_bear.rs -- KoalaBearParameters + KoalaBear4Parameters
  ext.rs        -- ExtField trait, BaseElem<P> (degree-1 bridge)
  ext4.rs       -- Ext4<P>, ext4_* #[cube] fns, ExtField impl
  ext5.rs       -- Ext5<P>, ext5_* #[cube] fns, ExtField impl
```

The cross-process device lock used by every kernel-launching test
(`Device<R>`) lives in [`r0-cube`](../r0-cube), not here.
