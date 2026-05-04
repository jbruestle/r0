//! Number-theoretic transforms over 31-bit Montgomery prime fields.
//!
//! Bit-reversed coefficient input → natural-order evaluation output
//! (the design's canonical convention — see `DESIGN.md` §7).

mod inv_pass;
mod pass;
mod twiddles;

pub use inv_pass::intt_pass;
pub use pass::ntt_pass;
pub use twiddles::{bit_reverse_in_place, build_inv_twiddles, build_twiddles, n_inv};
