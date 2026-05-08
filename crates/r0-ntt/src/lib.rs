//! Number-theoretic transforms over 31-bit Montgomery prime fields.
//!
//! Bit-reversed coefficient input -> natural-order evaluation output
//! (see the crate README for the ordering convention).
//!
//! # Quick start
//!
//! ```ignore
//! use r0_ntt::NttExec;
//! use r0_field::BabyBearParameters;
//! use cubecl::cuda::CudaRuntime;
//!
//! let exec = NttExec::<BabyBearParameters, CudaRuntime>::new(&Default::default(), 0);
//! exec.forward(&buf, 20, 100);
//! exec.inverse(&buf, 20, 100);
//! ```

mod exec;
mod fwd_pass;
mod inv_pass;
mod pass_common;
mod plan;
mod twiddles;

pub use exec::NttExec;
pub use twiddles::bit_reverse_in_place;

#[cfg(feature = "unstable-planner")]
pub use plan::{
    enumerate_valid_plans, heuristic_score, plan_heuristic, validate_plan, DeviceLimits, NttPlan,
    PassConfig, PlanError,
};
