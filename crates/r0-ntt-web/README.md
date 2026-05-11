# r0-ntt-web — Browser WebGPU benchmark demo

A throwaway-grade web page that runs the `r0-ntt` forward NTT on the browser's
WebGPU device (via `cubecl-wgpu`), reports adapter/limit info, and times a
configurable benchmark sweep. Built to answer one question:

> Is `cubecl` viable for client-side (in-browser) ZKP proving?

## Result: yes

Run on Mac M-series, Chrome stable, log_n=20, BabyBear forward NTT:

| Batch | Cycles | Median (µs) | Per-NTT (µs) |
|-------|--------|-------------|--------------|
| 32    | 1×32   | 3000        | 93.8         |
| 100   | 2×(64+36) | 8300     | **83.0**     |
| 128   | 1×128  | 11200       | 87.5         |

Plan: `(log_pass=10, z=8, log_wg=9) → (log_pass=10, z=8, log_wg=9)` (heuristic
plan, identical to the CUDA-autotuned best). For reference, CUDA autotuned
on the same plan is ~19 µs/NTT. The 4–5× gap has at least two plausible
contributors and we haven't separated them: (a) memory bandwidth — Apple
unified memory ~400 GB/s vs NVIDIA discrete ~1 TB/s+, and (b) Montgomery-mul
codegen — `mul_hi` lowers to a single `mul.hi.u32` on CUDA but is emulated
via schoolbook split (~10 ops) on WGSL (see r0-ntt README §3). A like-for-like
run on native (non-browser) `cubecl-wgpu` on this same Mac would help
attribute the gap.

Per-NTT cost is non-monotone in batch: the GPU saturates around 64 polys ×
~128 workgroups, so doubling further adds memory pressure without helping
throughput. Sweet spot on this hardware is batch ≈ 64.

## Running

```sh
cargo install trunk wasm-bindgen-cli       # one-time
rustup target add wasm32-unknown-unknown   # one-time

cd crates/r0-ntt-web
trunk serve --open
```

Browser support: Chrome/Edge stable, Safari 18+, Firefox Nightly (with
`dom.webgpu.enabled` flag). The page detects WebGPU absence and prints an
error.

## Files

- `src/lib.rs` — two `#[wasm_bindgen]` async exports: `diagnose()` and
  `run_benchmark(log_n, batch, warmups, samples)`.
- `index.html` — UI: device pane (auto-populated on load), input controls,
  Run button, results table.
- `Trunk.toml` — `filehash = false` so the JS module name is predictable.

## Notes for the next person to bring up cubecl on `wasm32-unknown-unknown`

The build path is straightforward but has several feature-flag papercuts.
All of these are pinned in `Cargo.toml`:

- `getrandom` needs the `wasm_js` cargo feature. (Pulled in transitively;
  defaults don't compile for `wasm32-unknown-unknown`.)
- `serde_json` needs `std` enabled. `cubecl-core` depends on it with no
  features and serde_json refuses `no_std` without `alloc` either.
- `cubecl-common` needs `serde` and `std` features. Without `serde`,
  `cubecl-runtime` fails to find `cubecl_common::bytes`.
- `cubecl` itself: `default-features = false, features = ["wgpu", "std"]`.

Async readback is mandatory:
- `client.sync()` returns `DynFut<...>` — `.await` it via `wasm-bindgen-futures`.
- The convenience `read_sync` / `read_one` panic on wasm (their futures don't
  block on a single-threaded event loop).

Device init quirk:
- `init_setup_async::<AutoGraphicsApi>(&device, ...)` registers a client in
  cubecl's global registry keyed by device id. **Calling it twice for the
  same device panics** (`"A server is still registered for device …"`).
  `diagnose()` does the one-time init; `run_benchmark()` just constructs a
  fresh `NttExec` (which finds the already-registered client).

Timing:
- `std::time::Instant::now()` panics on `wasm32-unknown-unknown`. Use the
  `web-time` crate as a drop-in (it dispatches to `performance.now()`).

Scratch buffer:
- `Device::acquire_with_scratch_for` allocates a single shared GPU scratch buffer. We
  use 512 MiB so `sub_batch` reaches 128 at log_n=20 (4 MiB/poly). WebGPU's
  reported `max_buffer_size` on this hardware was ~4 GiB, so plenty of
  headroom.

## What this is NOT

- Not a productionized prover entrypoint. No service worker, no error UI
  beyond a status line, no test coverage.
- Not a stable benchmark methodology. Browser timing has variable resolution
  (Chrome clamps `performance.now()` to ~5 µs without cross-origin
  isolation), and other tabs / GC pauses show up as outliers in the max.
