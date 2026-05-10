//! Project-specific helpers on top of cubecl.
//!
//! This crate is the workspace's home for cubecl-related machinery that
//! isn't tied to a specific math object: process / device hygiene
//! ([`Device<R>`]) and generic kernel primitives ([`Monoid`] +
//! [`plane_inclusive_scan`] / [`block_inclusive_scan`] /
//! [`block_inclusive_reduce`]). The forthcoming `ScanRecipe` /
//! `ScanExec` driver used by `r0-polynomial` builds on these.
//!
//! Trait *implementations* live with the type they operate on, not here:
//! e.g. `Sum<F>` for additive scans over `r0-field` elements lives in
//! `r0-field` next to `Ext4` / `Ext5`.

mod device;
pub use device::{Device, DEFAULT_SCRATCH_BYTES};

mod monoid;
pub use monoid::Monoid;

mod scan;
pub use scan::{block_inclusive_reduce, block_inclusive_scan, plane_inclusive_scan};

mod recipe;
pub use recipe::ScanRecipe;

mod exec;
pub use exec::ScanExec;
