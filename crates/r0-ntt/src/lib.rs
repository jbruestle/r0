//! Number-theoretic transforms over 31-bit Montgomery prime fields.
//!
//! Phase 2: monolithic forward NTT for `log_n ≤ 10`. Bit-reversed
//! coefficient input → natural-order evaluation output (the design's
//! canonical convention — see `DESIGN.md` §7).

mod monolithic;
mod twiddles;

pub use monolithic::ntt_monolithic;
pub use twiddles::{bit_reverse_in_place, build_twiddles};
