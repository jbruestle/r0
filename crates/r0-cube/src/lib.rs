//! Project-specific helpers on top of cubecl: the cross-process
//! [`Device<R>`] lock and shared scratch, the algebraic [`Monoid`]
//! trait, the plane- and block-level scan primitives
//! ([`plane_inclusive_scan()`] / [`block_inclusive_scan()`] /
//! [`block_inclusive_reduce()`]), and the recipe-driven [`ScanExec`]
//! driver ([`ScanRecipe`]) used by `r0-polynomial` and any other
//! prefix-scan-shaped pipeline.
//!
//! r0-cube is intentionally type-agnostic: it knows about `CubeType`s
//! and trait-generic associativity but never touches a specific math
//! object. Monoid impls live with the type they wrap (`PairScan<F>`
//! lives in `r0-polynomial`; future monoids will live with their
//! recipe crate, not here).
//!
//! See the crate README for design and a full pipeline walkthrough.
//!
//! # Backend selection
//!
//! Enable exactly one backend feature: `--features cuda` (NVIDIA) or
//! `--features wgpu` (Vulkan / Metal / WebGPU). The chosen backend is
//! re-exported as [`Runtime`].
//!
//! # Quick start
//!
//! ```ignore
//! use r0_cube::{Device, Runtime, ScanExec};
//!
//! let device = Device::<Runtime>::acquire();
//! let exec = ScanExec::<Runtime, MyRecipe>::new(&device, /*log_n_max*/ 20, /*max_batch*/ 32);
//! exec.run(&contexts, &input, &output, /*log_n*/ 20, /*batch*/ 32);
//! ```

// cubecl's #[cube] macro expansion produces `0u32.into()` and identity casts
// that clippy flags. These are intentional — the expanded IR types need them.
#![allow(clippy::useless_conversion)]

// ---------------------------------------------------------------------------
// Backend selection: exactly one of `cuda` or `wgpu` must be enabled.
// ---------------------------------------------------------------------------

#[cfg(all(feature = "cuda", feature = "wgpu"))]
compile_error!(
    "Enable exactly one backend: `cuda` or `wgpu`, not both. \
     Use `--features cuda` on NVIDIA or `--features wgpu` on Vulkan/Metal/WebGPU."
);

#[cfg(not(any(feature = "cuda", feature = "wgpu")))]
compile_error!(
    "No backend selected. Enable one with `--features cuda` (NVIDIA) \
     or `--features wgpu` (Vulkan/Metal/WebGPU)."
);

/// The cubecl runtime selected at compile time via feature flags.
#[cfg(feature = "cuda")]
pub type Runtime = cubecl::cuda::CudaRuntime;

/// The cubecl runtime selected at compile time via feature flags.
#[cfg(feature = "wgpu")]
pub type Runtime = cubecl::wgpu::WgpuRuntime;

mod device;
pub use device::Device;

mod monoid;
pub use monoid::Monoid;

mod scan;
pub use scan::{block_inclusive_reduce, block_inclusive_scan, plane_inclusive_scan};

mod recipe;
pub use recipe::ScanRecipe;

mod exec;
pub use exec::ScanExec;
