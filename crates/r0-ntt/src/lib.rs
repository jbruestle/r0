//! Number-theoretic transforms over 31-bit Montgomery prime fields.
//!
//! Bit-reversed coefficient input -> natural-order evaluation output
//! (the design's canonical convention -- see `DESIGN.md` §2).
//!
//! # Quick start
//!
//! ```ignore
//! use r0_ntt::{NttExec, plan_heuristic};
//! use r0_field::BabyBearParameters;
//! use cubecl::cuda::CudaRuntime;
//!
//! let exec = NttExec::<BabyBearParameters, CudaRuntime>::new(&Default::default(), 0);
//!
//! // Convenience path (heuristic plan):
//! exec.forward_auto(&buf, 20, 100);
//!
//! // Explicit plan path:
//! let plan = plan_heuristic(20, 100, exec.limits());
//! exec.forward(&buf, &plan, 100);
//! ```

mod exec;
mod fwd_pass;
mod inv_pass;
pub mod plan;
mod twiddles;

pub use exec::NttExec;
pub use fwd_pass::ntt_fwd_pass;
pub use inv_pass::ntt_inv_pass;
pub use plan::{
    enumerate_valid_plans, heuristic_score, plan_heuristic, validate_plan, DeviceLimits, NttPlan,
    PassConfig, PlanError,
};
pub use twiddles::{
    bit_reverse_in_place, build_fwd_twiddles, build_inv_twiddles, build_partial_fwd_twiddles,
    build_partial_inv_twiddles, n_inv, reconstruct_twiddle, LG_WINDOW, NUM_WINDOWS,
    PARTIAL_TWIDDLE_LEN, WINDOW_SIZE,
};
