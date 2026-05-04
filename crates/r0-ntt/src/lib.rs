//! Number-theoretic transforms over 31-bit Montgomery prime fields.
//!
//! Bit-reversed coefficient input → natural-order evaluation output
//! (the design's canonical convention — see `DESIGN.md` §7).

mod monolithic;
mod pass;
mod two_pass;
mod twiddles;

pub use monolithic::{ntt_monolithic, ntt_monolithic_inverse};
pub use pass::ntt_pass;
pub use two_pass::{intt_pass1, intt_pass2};
pub use twiddles::{bit_reverse_in_place, build_inv_twiddles, build_twiddles, n_inv};
