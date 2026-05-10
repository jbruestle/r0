# r0

A Rust workspace exploring CubeCL as the basis for a portable ZKP
prover. Kernels are written once in cubecl `#[cube]` and compile to
CUDA, wgpu (Vulkan/Metal/WebGPU), and CPU.

## Crates

- [`r0-field`](crates/r0-field) — 31-bit Montgomery prime fields
  (BabyBear, KoalaBear). Single-source `#[cube]` arithmetic shared by
  host and kernel code.
- [`r0-ntt`](crates/r0-ntt) — batched forward/inverse NTT over those
  fields. See [the crate README](crates/r0-ntt/README.md) for design
  and performance.
- [`r0-ntt-web`](crates/r0-ntt-web) — browser WebGPU benchmark demo
  for `r0-ntt`.

## Build & test

```sh
cargo build --workspace
cargo test --workspace
```

Default features enable the wgpu and CPU backends. The CUDA backend is
opt-in (it links `libcuda` at runtime, which would make `cargo test
--workspace` panic on machines without an NVIDIA driver). On a CUDA
host:

```sh
cargo test --workspace --features r0-ntt/cuda
```

Autotune and diagnostics integration tests are gated behind `r0-ntt`'s
`unstable-planner` feature and run only with `--ignored`.

GPU tests acquire a process-shared file lock (`r0_field::Device`) so
multiple test binaries can't fight for the same device under cargo's
default per-binary parallelism — `cargo test --workspace` works without
a `--test-threads=1` workaround.

## License

Apache-2.0.
