//! Project-specific helpers on top of cubecl.
//!
//! This crate is the workspace's home for cubecl-related machinery that
//! isn't tied to a specific math object — process / device hygiene, and
//! (forthcoming) generic kernel primitives like a `Monoid` trait,
//! plane- and block-level scans, and the `ScanRecipe` / `ScanExec`
//! driver used by `r0-polynomial`.
//!
//! Currently shipped: [`Device<R>`], the process-shared exclusive lock
//! around a cubecl device, used by every kernel-launching test in the
//! workspace.

mod device;
pub use device::{Device, DEFAULT_SCRATCH_BYTES};
