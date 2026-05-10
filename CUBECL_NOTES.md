# cubecl — orientation for future agents

Notes for someone (human or agent) walking into this codebase having read
the standard cubecl docs but not yet hit the rough edges. cubecl is the
foundation of every kernel here; understanding what it does and doesn't
do well is half the work.

We're on **cubecl 0.9** at time of writing. Quirks below are version-
specific; check the workspace `Cargo.toml` and re-test before assuming
they still apply.

---

## What cubecl gives us

- A `#[cube]` proc macro that turns a Rust function into a kernel that
  compiles to **CUDA, WGSL (wgpu / Vulkan / Metal / WebGPU), and a CPU
  emulator**. Same source, three back-ends, one runtime trait `R:
  Runtime` to pick at launch.
- A `#[derive(CubeType)]` attribute for user structs that flow through
  kernels.
- A small standard library: `SharedMemory<T>`, `Array<T>`, the cube
  intrinsics (`UNIT_POS`, `CUBE_POS_X`, `ABSOLUTE_POS`, `sync_cube()`),
  warp-level shuffles, and a few fixed-op reductions/scans.
- `#[cube] trait` and `#[cube] impl` for trait-generic kernels (we lean
  on this hard for `ExtField` in `r0-field`).

What it does **not** give us:
- Cooperative groups / grid-wide barriers. (CUDA has them; cubecl 0.9
  doesn't expose them; WGSL/Metal can't have them at all.) Multi-stage
  algorithms are multi-launch.
- Generic-op subgroup scans. The plane scans only do `+`, `*`, `min`,
  `max`. For custom operations roll your own from `plane_shuffle_up`.
- Native `u64` arithmetic in cube IR. Some host-side `as u64` is fine in
  a `#[cube]` body but only as a runtime cast on a u32 local — see the
  `mul_hi_u32` pattern in `r0-field/src/monty.rs`.

---

## Patterns that work

### One CubeType struct, host + cube usage

`Ext4<P>` / `Ext5<P>` / `BaseElem<P>` derive `CubeType` *and* are the
host-side wrapper users write code against. Operator overloads on the
host forward to the same `#[cube] fn`s the kernels call. Single source,
no aliasing layer. `MontyField<P>` predates this pattern — it's
`#[repr(transparent)]` over `u32` and isn't (yet) `CubeType`. The newer
extension types are the model to copy.

### `#[cube] trait` for kernel genericity

```rust
#[cube]
pub trait ExtField: CubeType + … {
    type Base: MontyParameters;
    const DEGREE: u32;
    fn add(a: Self, b: Self) -> Self;
    // …
}
```

Then `#[cube(launch_unchecked)] fn k<F: ExtField>(…)` monomorphizes per
impl through `launch_unchecked::<F, R>`. Tested end-to-end on CPU + wgpu
in `r0-field/tests/ext_cube_smoke.rs`.

Constraint defaults need to be specified explicitly; see the trait above
for the whole list (`CubeType + Copy + Clone + Sized + Send + Sync +
'static`).

### `PhantomData<P>` inside a CubeType struct

Annotate the field `#[cube(comptime)]`. Without that, cubecl tries to
materialise it in IR.

```rust
#[derive(CubeType, Copy, Clone)]
pub struct Ext4<P: BinomialExt4Parameters> {
    pub c0: u32,  pub c1: u32,  pub c2: u32,  pub c3: u32,
    #[cube(comptime)]
    _p: PhantomData<P>,
}
```

### `mul_hi_u32` — the function-and-module name pattern

`u32::mul_hi` from cubecl-core panics in host context (host body is
`unexpanded!()`). The `#[cube]` macro rewrites a call `foo(a, b)` inside
a cube body into `foo::expand(scope, a, b)`. So we pair a free
`fn mul_hi_u32` (host body) with a sibling `mod mul_hi_u32` (containing
the `expand` IR-builder). Same name, different namespace, both contexts
resolve correctly. Pattern lives in `r0-field/src/monty.rs`; copy it if
you ever need to bridge another panicking-on-host primitive.

### Cross-process device locking

cubecl's wgpu backend hands out one device per process; `cargo test` runs
each integration-test binary as its own process *in parallel*. Multiple
binaries fighting for one GPU = timeouts and flaky failures. Solution:
`r0_field::Device<R>` (will move to `r0_cube`) wraps `R::Device` with a
flock-based file lock keyed per runtime. Acquire one per test, pass
`&device` to executor constructors.

### Layout via `Array<u32>` + load/store helpers

cubecl 0.9 packs CubeType structs reasonably, but we don't trust
struct-of-`u32`-fields layout for memory I/O. Kernels take `&Array<u32>`
and use `F::load(arr, base, i, n)` / `F::store(...)` helpers built into
the field trait, which compute the right offsets for transposed layout
(component `c` of element `i` at offset `c·n + i`). Removes a foot-gun
and lets one `Array<u32>` carry any extension shape.

---

## Quirks we've hit

| Problem | Workaround |
|---|---|
| `u32 % u32` lowers to ambiguous `metal::select(...)` on Metal — fails to compile through wgpu. | Don't use `%` in cube bodies. Field reductions use conditional subtract (`if x >= p { x - p } else { x }`). |
| `u32::mul_hi` panics on host. | Function-and-module bridge (see above). |
| No `From<u64> for ConstantValue` in cubecl 0.9; u64 literals can't appear in cube IR. | All `u64` in cube bodies must come from a runtime `as u64` cast on a `u32` local. Done only inside `mul_hi_u32`'s host body. |
| Trait const defaults referencing `Self::OTHER_CONST` aren't reachable through generic bounds (Rust E0790). | Don't use defaults that reference other trait consts. `MontyParameters::MONT_ONE` is required on each impl, with the same one-line `(((1u64 << 32) % Self::PRIME as u64) as u32)` — duplicated by design. |
| Cube fn parameters are picky about `usize` vs `u32`. `ABSOLUTE_POS` is `usize` in 0.9. Mixing produces opaque `From<ExpandElementTyped<…>>` errors from the macro. | Pick one and stick to it inside any given kernel. Cast at the boundary. |
| `cargo test --workspace --features cuda` panics on non-NVIDIA hosts because cubecl-cuda dynamically loads `libcuda`. | `cuda` is **off** by default in `r0-ntt`. CUDA developers run `cargo test --workspace --features r0-ntt/cuda`. |
| The cubecl macro emits both a free `fn name` and a sibling `mod name` for `#[cube] fn name`. Rustdoc complains about the ambiguous link if you write `[`name`]`. | Use `[`name()`]` for the function in doc links. |

---

## What's available, briefly (cubecl 0.9 surface we use)

- **Topology intrinsics**: `UNIT_POS` (thread within block, `usize`),
  `UNIT_POS_PLANE` (lane within warp), `CUBE_POS_X/Y/Z` (block in grid),
  `CUBE_DIM_X/Y/Z` (block size), `ABSOLUTE_POS` (global thread id).
- **Memory**: `SharedMemory::<T>::new(size)`, `Array<T>` (kernel
  parameter), `client.create_from_slice`, `client.empty`, `client.read_one`,
  `Handle::offset_start` (for sub-batch slicing).
- **Sync**: `sync_cube()` (workgroup barrier).
- **Plane (warp) ops** (`crates/.cargo/registry/src/.../cubecl-core-0.9.0/src/frontend/plane.rs`):
  `plane_broadcast`, `plane_shuffle`, `plane_shuffle_xor`,
  `plane_shuffle_up`, `plane_shuffle_down`, `plane_sum`, `plane_prod`,
  `plane_max`, `plane_min`, `plane_inclusive_sum`, `plane_exclusive_sum`,
  `plane_inclusive_prod`, `plane_exclusive_prod`, `plane_all`,
  `plane_any`, `plane_ballot`. **No generic-op scan.**
- **Launch**: `kernel::launch_unchecked::<P, R>(client, count, dim,
  args…)`. We use `launch_unchecked` everywhere to skip cubecl's runtime
  argument validation; the type system catches mistakes earlier.
- **`#[cube] trait`** is documented in cubecl-core's
  `runtime_tests/traits.rs`. That's the canonical example.
- **Test utilities**: cubecl exposes `cubecl::cpu::CpuRuntime` (a CPU
  emulator backend, slow but always available) and `cubecl::wgpu::WgpuRuntime`
  (the wgpu backend). Cross-running our tests on both catches a lot of
  codegen-divergence bugs.

---

## Where to look when something's weird

1. **Compile error inside a `#[cube]` macro body that mentions
   `ExpandElementTyped<...>` or "From not implemented"** — almost always
   a `usize`/`u32` mixing problem. Check the types of every operand.
2. **wgpu/Metal-only failure** — first suspect: `%` in a cube body.
   Second: emitting an unsupported-on-WGSL operation (cubecl's WGSL
   codegen is the youngest backend and has the most rough edges). Run
   the same kernel under `CpuRuntime` to confirm the algorithm is right.
3. **Test passes alone but fails in `cargo test --workspace`** —
   GPU contention. Make sure the test acquires a `Device<R>` (or
   `Device::<R>::acquire()` once that's in `r0-cube`). The cross-process
   flock serializes binaries.
4. **Mysterious panic from `cudarc` mentioning `libcuda.dylib`** — you're
   on a non-CUDA host with the `cuda` feature on. Drop the feature.

---

## When in doubt

- The cubecl source is your friend; it's well-organized. Start at
  `~/.cargo/registry/src/index.crates.io-*/cubecl-core-0.9.0/src/`.
- `cubecl-core/src/runtime_tests/` has small canonical examples for
  every primitive surface. We use these as templates.
- Everything in this workspace's tests/cube_smoke.rs is a working
  end-to-end example you can copy.
