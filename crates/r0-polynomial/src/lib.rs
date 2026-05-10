//! Polynomial-level operations for `r0-field` polynomials, built on the
//! `r0-cube` `ScanRecipe` substrate.
//!
//! Currently ships [`PolyDivExec`] for division by `(x − z)` (synthetic
//! division reframed as a parallel prefix scan). See the crate README
//! for design.

mod pair_scan;
pub use pair_scan::{PairScan, PairScanLayout};

mod div_by_x_minus_z;
pub use div_by_x_minus_z::DivByXMinusZ;

mod exec;
pub use exec::PolyDivExec;

pub mod host_ref;
