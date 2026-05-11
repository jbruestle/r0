# r0

A Rust workspace exploring CubeCL as the basis for a portable ZKP
prover. Kernels are written once in cubecl `#[cube]` and compile to
CUDA, wgpu (Vulkan/Metal/WebGPU), and CPU.

## Crates

- [`r0-cube`](crates/r0-cube) — project-specific helpers on top of
  cubecl: the `Device` lock + shared scratch, the `Monoid` trait
  with plane- and block-level scan / reduce primitives, and the
  recipe-driven `ScanExec` driver (with recursive spine for
  `n > wg_size²`). Also exports the compile-time `Runtime` type alias
  selected by the backend feature. See
  [the crate README](crates/r0-cube/README.md) for design.
- [`r0-field`](crates/r0-field) — 31-bit Montgomery prime fields
  (BabyBear, KoalaBear) plus their degree-4/5 binomial extensions
  (BB^4, KB^4, BB^5), and the `ExtField` `#[cube]` trait that lets
  later kernels stay generic over the inner field. See
  [the crate README](crates/r0-field/README.md) for design.
- [`r0-ntt`](crates/r0-ntt) — batched forward/inverse NTT over those
  fields. See [the crate README](crates/r0-ntt/README.md) for design
  and performance.
- [`r0-polynomial`](crates/r0-polynomial) — polynomial-level operations
  on `r0-field` polynomials, on top of `r0-cube`'s scan substrate.
  Currently division by `(x − z)`; future work includes evaluation,
  FRI fold, and OOD evaluation. See
  [the crate README](crates/r0-polynomial/README.md) for design.
- [`r0-ntt-web`](crates/r0-ntt-web) — browser WebGPU benchmark demo
  for `r0-ntt`.

## Build & test

You must select exactly one GPU backend via feature flags. The workspace
will not compile without one — `r0-cube` emits a `compile_error!`.

**NVIDIA (CUDA):**

```sh
cargo build --features cuda
cargo test  --features cuda
```

**Vulkan / Metal / WebGPU (wgpu):**

```sh
cargo build --features wgpu
cargo test  --features wgpu
```

The feature name is the same across all default-member crates (`cuda`
or `wgpu`). Omit `--workspace` — `r0-ntt-web` hardcodes wgpu and
would conflict with `--features cuda`.

## Benchmarks

```sh
cargo bench --features cuda -p r0-ntt
cargo bench --features cuda -p r0-polynomial
```

Substitute `wgpu` for `cuda` on non-NVIDIA hardware.

## Notes

Autotune and diagnostics integration tests are gated behind `r0-ntt`'s
`unstable-planner` feature and run only with `--ignored`.

GPU tests acquire a process-shared file lock (`r0_cube::Device`) so
multiple test binaries can't fight for the same device under cargo's
default per-binary parallelism — `cargo test --workspace` works without
a `--test-threads=1` workaround.

## License

Apache-2.0.
