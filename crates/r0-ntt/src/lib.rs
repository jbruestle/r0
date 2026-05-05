//! Number-theoretic transforms over 31-bit Montgomery prime fields.
//!
//! Bit-reversed coefficient input -> natural-order evaluation output
//! (the design's canonical convention -- see `DESIGN.md` §2).
//!
//! # Quick start
//!
//! ```ignore
//! use r0_ntt::NttPlanner;
//! use r0_field::BabyBearParameters;
//! use cubecl::cuda::CudaRuntime;
//!
//! let planner = NttPlanner::<BabyBearParameters, CudaRuntime>::new(&Default::default(), 0);
//! planner.forward(&buf, 20, 100);  // 100 polynomials of size 2^20
//! ```

mod fwd_pass;
mod inv_pass;
mod planner;
mod twiddles;

pub use fwd_pass::ntt_fwd_pass;
pub use inv_pass::ntt_inv_pass;
pub use planner::NttPlanner;
pub use twiddles::{
    bit_reverse_in_place, build_fwd_twiddles, build_inv_twiddles,
    build_partial_fwd_twiddles, build_partial_inv_twiddles, n_inv,
    LG_WINDOW, NUM_WINDOWS, PARTIAL_TWIDDLE_LEN, WINDOW_SIZE,
};
