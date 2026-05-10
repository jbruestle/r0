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
//! object. Monoid impls live with the type they wrap (`Sum<F>` and
//! `PairScan<F>` over `r0-field` elements live in `r0-field` /
//! `r0-polynomial`, not here).
//!
//! See the crate README for design and a full pipeline walkthrough.
//!
//! # Quick start
//!
//! ```ignore
//! use r0_cube::{Device, ScanExec};
//! use cubecl::wgpu::WgpuRuntime;
//!
//! let device = Device::<WgpuRuntime>::acquire();
//! let exec = ScanExec::<WgpuRuntime, MyRecipe>::new(&device, /*log_n_max*/ 20, /*max_batch*/ 32);
//! exec.run(&contexts, &input, &output, /*log_n*/ 20, /*batch*/ 32);
//! ```

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
