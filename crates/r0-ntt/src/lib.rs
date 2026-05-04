//! Number-theoretic transforms over 31-bit Montgomery prime fields.
//!
//! Bit-reversed coefficient input -> natural-order evaluation output
//! (the design's canonical convention -- see `DESIGN.md` S7).

mod fwd_pass;
mod inv_pass;
mod twiddles;

pub use fwd_pass::ntt_fwd_pass;
pub use inv_pass::ntt_inv_pass;
pub use twiddles::{bit_reverse_in_place, build_fwd_twiddles, build_inv_twiddles, n_inv};
