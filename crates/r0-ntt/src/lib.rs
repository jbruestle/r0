//! Batched number-theoretic transforms over 31-bit Montgomery prime
//! fields, on GPU and CPU via cubecl.
//!
//! Forward NTT (R→N) takes bit-reversed coefficients and produces
//! natural-order evaluations; inverse (N→R) is the dual. There is no
//! ordering toggle — use [`bit_reverse_in_place`] to prepare R-side
//! input or interpret R-side output. Polynomials are sized `2^log_n`,
//! up to `log_n = 24` (capped by KoalaBear's 2-adicity); BabyBear's
//! 2-adicity 27 also caps at 24 in this crate.
//!
//! # Quick start
//!
//! ```ignore
//! use r0_ntt::NttExec;
//! use r0_cube::Device;
//! use r0_field::BabyBearParameters;
//! use cubecl::cuda::CudaRuntime;
//!
//! let device = Device::<CudaRuntime>::acquire();
//! let exec = NttExec::<BabyBearParameters, CudaRuntime>::new(&device);
//!
//! // 100 NTTs of size 2^20, in place on `buf`.
//! exec.forward(&buf, 20, 100);
//! exec.inverse(&buf, 20, 100);
//! ```
//!
//! # Stable surface
//!
//! - [`NttExec`]: device-resident executor; one per `(device, field)`.
//! - [`bit_reverse_in_place`]: in-place bit-reversal helper.
//!
//! Kernel internals, twiddle construction, and planning are not part
//! of the stable API. The planner is exposed under the
//! `unstable-planner` feature for autotuning experiments; that surface
//! is still in flux.
//!
//! See the crate README for the multi-pass kernel design and
//! performance results.

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
