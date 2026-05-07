//! NTT execution plans and planning strategies.
//!
//! This module contains pure data types and logic — no device interaction,
//! no cubecl dependency. Plans can be constructed via [`plan_heuristic`],
//! manually, or (eventually) via autotuning.

/// Size of a field element in bytes (u32 Montgomery form).
const ELEM_BYTES: usize = 4;

/// Configuration for a single pass of a multi-pass NTT.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PassConfig {
    /// Number of NTT stages this pass handles (N_pass = 2^log_pass).
    pub log_pass: u32,
    /// Global stage index where this pass begins (cumulative sum of prior log_pass values).
    pub stage_offset: u32,
    /// Independent chunks each workgroup processes. Must be a power of two.
    pub z_count: u32,
    /// Workgroup size exponent (workgroup has 2^log_wg threads).
    pub log_wg: u32,
}

/// A complete execution plan for an NTT of size 2^log_n.
#[derive(Debug, Clone)]
pub struct NttPlan {
    /// Total transform size exponent (N = 2^log_n).
    pub log_n: u32,
    /// Per-pass configurations, in execution order.
    pub passes: Vec<PassConfig>,
    /// Number of NTTs to process per kernel launch group.
    /// Bounded by scratch memory for multi-pass plans.
    pub sub_batch: usize,
}

/// Device capability limits relevant to NTT planning.
#[derive(Debug, Clone)]
pub struct DeviceLimits {
    /// Maximum shared memory per workgroup, in bytes.
    pub max_shared_mem_bytes: usize,
    /// Maximum threads per workgroup.
    pub max_threads_per_wg: u32,
    /// Scratch buffer size in bytes (user-configured, should be po2).
    pub scratch_bytes: usize,
}

/// Errors from [`validate_plan`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanError {
    LogNZero,
    EmptyPlan,
    LogPassZero { pass: usize },
    BitsMismatch { expected: u32, got: u32 },
    StageOffsetMismatch { pass: usize, expected: u32, got: u32 },
    SharedMemExceeded { pass: usize, needed: usize, limit: usize },
    WorkgroupTooLarge { pass: usize, threads: u32, limit: u32 },
    LogWgTooLarge { pass: usize, log_wg: u32, max: u32 },
    ZCountNotPo2 { pass: usize, z_count: u32 },
    ZCountExceedsNOther { pass: usize, z_count: u32, n_other: u32 },
    ScratchExceeded { needed: usize, limit: usize },
    SubBatchZero,
}

/// Validate a plan against device limits. Returns all constraint violations found.
pub fn validate_plan(plan: &NttPlan, limits: &DeviceLimits) -> Result<(), Vec<PlanError>> {
    let mut errors = Vec::new();

    if plan.log_n == 0 {
        errors.push(PlanError::LogNZero);
        return Err(errors);
    }
    if plan.passes.is_empty() {
        errors.push(PlanError::EmptyPlan);
        return Err(errors);
    }

    // Bits must sum to log_n.
    let total_bits: u32 = plan.passes.iter().map(|p| p.log_pass).sum();
    if total_bits != plan.log_n {
        errors.push(PlanError::BitsMismatch {
            expected: plan.log_n,
            got: total_bits,
        });
    }

    // Per-pass checks.
    let mut cumulative = 0u32;
    for (i, pass) in plan.passes.iter().enumerate() {
        // stage_offset must match cumulative sum.
        if pass.stage_offset != cumulative {
            errors.push(PlanError::StageOffsetMismatch {
                pass: i,
                expected: cumulative,
                got: pass.stage_offset,
            });
        }

        if pass.log_pass == 0 {
            errors.push(PlanError::LogPassZero { pass: i });
            cumulative += pass.log_pass;
            continue;
        }

        // log_wg <= log_pass - 1 (need ≥2 elements per thread for butterflies).
        if pass.log_wg > pass.log_pass - 1 {
            errors.push(PlanError::LogWgTooLarge {
                pass: i,
                log_wg: pass.log_wg,
                max: pass.log_pass - 1,
            });
        }

        // Thread count within device limit.
        let threads = 1u32 << pass.log_wg;
        if threads > limits.max_threads_per_wg {
            errors.push(PlanError::WorkgroupTooLarge {
                pass: i,
                threads,
                limit: limits.max_threads_per_wg,
            });
        }

        // z_count must be a positive power of two.
        if pass.z_count == 0 || !pass.z_count.is_power_of_two() {
            errors.push(PlanError::ZCountNotPo2 {
                pass: i,
                z_count: pass.z_count,
            });
        }

        // Shared memory: z_count * 2^log_pass * elem_size.
        let shared_needed = (pass.z_count as usize) * (1usize << pass.log_pass) * ELEM_BYTES;
        if shared_needed > limits.max_shared_mem_bytes {
            errors.push(PlanError::SharedMemExceeded {
                pass: i,
                needed: shared_needed,
                limit: limits.max_shared_mem_bytes,
            });
        }

        // z_count <= n_other = 2^(log_n - log_pass).
        if pass.log_pass <= plan.log_n {
            let n_other = 1u32.checked_shl(plan.log_n - pass.log_pass).unwrap_or(u32::MAX);
            if pass.z_count > n_other {
                errors.push(PlanError::ZCountExceedsNOther {
                    pass: i,
                    z_count: pass.z_count,
                    n_other,
                });
            }
        }

        cumulative += pass.log_pass;
    }

    if plan.sub_batch == 0 {
        errors.push(PlanError::SubBatchZero);
    }

    // Scratch check (only for multi-pass).
    if plan.passes.len() > 1 && plan.sub_batch > 0 {
        let n = 1usize << plan.log_n;
        let scratch_needed = n * plan.sub_batch * ELEM_BYTES;
        if scratch_needed > limits.scratch_bytes {
            errors.push(PlanError::ScratchExceeded {
                needed: scratch_needed,
                limit: limits.scratch_bytes,
            });
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Heuristic score for a plan. Lower = predicted faster.
///
/// Used by the autotuner to order trial plans: score all candidates, sort
/// ascending, benchmark in that order. The heuristic doesn't need to be
/// precise — it just needs to avoid burying the best plan deep in the list.
pub fn heuristic_score(plan: &NttPlan) -> f64 {
    score_passes(&plan.passes)
}

fn score_passes(passes: &[PassConfig]) -> f64 {
    // Lower is better.
    //
    // Dominant factors observed empirically (log_n=20, CUDA):
    //   1. log_wg: 5× difference from wg=6 to wg=9. More threads = better
    //      latency hiding. This is by far the most important per-pass knob.
    //   2. num_passes: each pass is a full global memory round-trip.
    //   3. z_count: affects coalescing quality. ~15% difference between z=8 and z=16.
    let pass_cost = 10.0;
    let coalescing_weight = 2.0;
    let wg_weight = 5.0;

    let mut score = passes.len() as f64 * pass_cost;
    for p in passes {
        // Penalize low z_count (coalescing).
        score += coalescing_weight / p.z_count as f64;
        // Penalize low log_wg (latency hiding). Measured as distance from 10
        // (log2(1024), the practical max on most GPUs).
        score += wg_weight * (10u32.saturating_sub(p.log_wg)) as f64;
    }
    score
}

/// Produce a reasonable plan without benchmarking.
///
/// Tries 1–4 pass decompositions with balanced splits, picks the best
/// by [`heuristic_score`]. Each pass gets the largest power-of-two z_count
/// that fits shared memory (capped at 32) and the largest valid log_wg.
pub fn plan_heuristic(log_n: u32, batch: usize, limits: &DeviceLimits) -> NttPlan {
    assert!(log_n >= 1, "log_n must be >= 1");

    let mut best: Option<(Vec<PassConfig>, f64)> = None;

    for num_passes in 1..=4u32.min(log_n) {
        let pass_sizes = balanced_split(log_n, num_passes as usize);

        let mut configs = Vec::with_capacity(num_passes as usize);
        let mut stage_offset = 0u32;
        let mut valid = true;

        for &lp in &pass_sizes {
            if lp == 0 {
                valid = false;
                break;
            }

            let log_wg = best_log_wg(lp, limits);
            let z = heuristic_z_count(lp, log_n, limits);

            if z == 0 {
                valid = false;
                break;
            }

            configs.push(PassConfig {
                log_pass: lp,
                stage_offset,
                z_count: z,
                log_wg,
            });
            stage_offset += lp;
        }

        if !valid {
            continue;
        }

        let score = score_passes(&configs);
        if best.is_none() || score < best.as_ref().unwrap().1 {
            best = Some((configs, score));
        }
    }

    let passes = best
        .expect("no valid plan found for given log_n and device limits")
        .0;

    let sub_batch = compute_sub_batch(log_n, batch, &passes, limits);

    NttPlan {
        log_n,
        passes,
        sub_batch,
    }
}

/// Enumerate all constraint-valid plans for the given parameters.
///
/// Generates plans with 1 through `max_passes` passes, all valid
/// pass-size compositions, and all valid (z_count, log_wg) combinations
/// per pass. Plans are returned unsorted.
pub fn enumerate_valid_plans(
    log_n: u32,
    batch: usize,
    limits: &DeviceLimits,
    max_passes: u32,
) -> Vec<NttPlan> {
    assert!(log_n >= 1);
    let mut plans = Vec::new();

    // Max log_pass that fits z=1 in shared memory.
    let max_log_pass = max_fitting_log_pass(limits);

    for k in 1..=max_passes.min(log_n) {
        let compositions = compositions(log_n, k as usize, 1, max_log_pass);
        for comp in &compositions {
            enumerate_params(log_n, comp, batch, limits, &mut plans);
        }
    }

    plans
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Largest log_pass where z_count=1 fits in shared memory.
fn max_fitting_log_pass(limits: &DeviceLimits) -> u32 {
    // 2^log_pass * 4 <= shared_mem
    if limits.max_shared_mem_bytes < ELEM_BYTES {
        return 0;
    }
    let max_elems = limits.max_shared_mem_bytes / ELEM_BYTES;
    31 - (max_elems as u32).leading_zeros()
}

/// Generate all compositions of `n` into `k` parts, each in `[min_part, max_part]`.
fn compositions(n: u32, k: usize, min_part: u32, max_part: u32) -> Vec<Vec<u32>> {
    if k == 0 {
        return if n == 0 { vec![vec![]] } else { vec![] };
    }
    if k == 1 {
        return if n >= min_part && n <= max_part {
            vec![vec![n]]
        } else {
            vec![]
        };
    }
    let mut result = Vec::new();
    // Upper bound for first part: leave room for remaining k-1 parts of at least min_part each.
    let upper = max_part.min(n.saturating_sub((k as u32 - 1) * min_part));
    for first in min_part..=upper {
        for mut rest in compositions(n - first, k - 1, min_part, max_part) {
            rest.insert(0, first);
            result.push(rest);
        }
    }
    result
}

/// For a given pass-size composition, enumerate all valid (z_count, log_wg)
/// combinations per pass and add valid plans to `out`.
fn enumerate_params(
    log_n: u32,
    pass_sizes: &[u32],
    batch: usize,
    limits: &DeviceLimits,
    out: &mut Vec<NttPlan>,
) {
    // For each pass, compute the set of valid (z_count, log_wg) pairs.
    let mut per_pass_options: Vec<Vec<(u32, u32)>> = Vec::new();

    for &lp in pass_sizes {
        let max_z = best_z_count(lp, log_n, limits);
        if max_z == 0 {
            return; // this composition is infeasible
        }

        let max_lwg = best_log_wg(lp, limits);
        // log_wg range: from max(0, max_lwg - 3) to max_lwg.
        let min_lwg = max_lwg.saturating_sub(3);

        let mut options = Vec::new();
        // z_count: all po2 from 1 up to max_z.
        let mut z = 1u32;
        while z <= max_z {
            for lwg in min_lwg..=max_lwg {
                options.push((z, lwg));
            }
            z *= 2;
        }
        per_pass_options.push(options);
    }

    // Cross-product of per-pass options.
    let mut indices = vec![0usize; pass_sizes.len()];
    loop {
        // Build plan from current indices.
        let mut passes = Vec::with_capacity(pass_sizes.len());
        let mut so = 0u32;
        for (i, &lp) in pass_sizes.iter().enumerate() {
            let (z, lwg) = per_pass_options[i][indices[i]];
            passes.push(PassConfig {
                log_pass: lp,
                stage_offset: so,
                z_count: z,
                log_wg: lwg,
            });
            so += lp;
        }

        let sub_batch = compute_sub_batch(log_n, batch, &passes, limits);
        let plan = NttPlan {
            log_n,
            passes,
            sub_batch,
        };

        if validate_plan(&plan, limits).is_ok() {
            out.push(plan);
        }

        // Advance indices (odometer-style).
        let mut carry = true;
        for i in (0..indices.len()).rev() {
            if carry {
                indices[i] += 1;
                if indices[i] < per_pass_options[i].len() {
                    carry = false;
                } else {
                    indices[i] = 0;
                }
            }
        }
        if carry {
            break;
        }
    }
}

/// Split log_n into num_passes balanced parts. Remainder bits go to the
/// earlier passes (front-loading).
fn balanced_split(log_n: u32, num_passes: usize) -> Vec<u32> {
    let base = log_n / num_passes as u32;
    let rem = log_n % num_passes as u32;
    (0..num_passes)
        .map(|i| base + if (i as u32) < rem { 1 } else { 0 })
        .collect()
}

/// Largest valid log_wg for a pass of size log_pass on this device.
fn best_log_wg(log_pass: u32, limits: &DeviceLimits) -> u32 {
    let device_max = if limits.max_threads_per_wg <= 1 {
        0
    } else {
        31 - limits.max_threads_per_wg.leading_zeros()
    };
    // Need log_wg <= log_pass - 1 (≥2 elements per thread).
    (log_pass - 1).min(device_max)
}

/// Largest valid power-of-two z_count for a pass, capped.
///
/// Used by both enumeration (returns the hard max) and the heuristic.
/// The heuristic caps at 8: empirically, z=4-8 tends to beat higher values
/// because lower shared memory usage improves occupancy (more concurrent
/// workgroups). The enumeration uses this to set the upper bound for search.
fn best_z_count(log_pass: u32, log_n: u32, limits: &DeviceLimits) -> u32 {
    max_z_count(log_pass, log_n, limits, 32)
}

fn heuristic_z_count(log_pass: u32, log_n: u32, limits: &DeviceLimits) -> u32 {
    // Cap at 8: autotune data (log_n=20, CUDA) shows z=4-8 beats z=16
    // due to occupancy effects. Conservative cap avoids over-subscribing
    // shared memory.
    max_z_count(log_pass, log_n, limits, 8)
}

fn max_z_count(log_pass: u32, log_n: u32, limits: &DeviceLimits, cap: u32) -> u32 {
    let n_pass_bytes = (1usize << log_pass) * ELEM_BYTES;
    if n_pass_bytes > limits.max_shared_mem_bytes {
        return 0; // can't fit even z=1
    }
    let max_from_shared = (limits.max_shared_mem_bytes / n_pass_bytes) as u32;
    let n_other = 1u32.checked_shl(log_n - log_pass).unwrap_or(u32::MAX);
    let max_z = max_from_shared.min(n_other);
    if max_z == 0 {
        return 0;
    }
    // Floor to power of two.
    let po2 = 1u32 << (31 - max_z.leading_zeros());
    po2.min(cap)
}

fn compute_sub_batch(
    log_n: u32,
    batch: usize,
    passes: &[PassConfig],
    limits: &DeviceLimits,
) -> usize {
    if passes.len() <= 1 {
        // Single-pass: no scratch needed, process everything at once.
        batch
    } else {
        let n = 1usize << log_n;
        let max_sub = limits.scratch_bytes / (n * ELEM_BYTES);
        batch.min(max_sub).max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cuda_like_limits() -> DeviceLimits {
        DeviceLimits {
            max_shared_mem_bytes: 49152, // 48 KiB
            max_threads_per_wg: 1024,
            scratch_bytes: 64 * 1024 * 1024, // 64 MiB
        }
    }

    #[test]
    fn balanced_split_even() {
        assert_eq!(balanced_split(20, 2), vec![10, 10]);
        assert_eq!(balanced_split(21, 3), vec![7, 7, 7]);
    }

    #[test]
    fn balanced_split_remainder() {
        assert_eq!(balanced_split(21, 2), vec![11, 10]);
        assert_eq!(balanced_split(20, 3), vec![7, 7, 6]);
    }

    #[test]
    fn validate_valid_plan() {
        let limits = cuda_like_limits();
        let plan = NttPlan {
            log_n: 20,
            passes: vec![
                PassConfig { log_pass: 10, stage_offset: 0, z_count: 8, log_wg: 8 },
                PassConfig { log_pass: 10, stage_offset: 10, z_count: 8, log_wg: 8 },
            ],
            sub_batch: 4,
        };
        assert!(validate_plan(&plan, &limits).is_ok());
    }

    #[test]
    fn validate_bits_mismatch() {
        let limits = cuda_like_limits();
        let plan = NttPlan {
            log_n: 20,
            passes: vec![
                PassConfig { log_pass: 10, stage_offset: 0, z_count: 8, log_wg: 8 },
                PassConfig { log_pass: 11, stage_offset: 10, z_count: 4, log_wg: 8 },
            ],
            sub_batch: 4,
        };
        let errs = validate_plan(&plan, &limits).unwrap_err();
        assert!(errs.iter().any(|e| matches!(e, PlanError::BitsMismatch { .. })));
    }

    #[test]
    fn validate_shared_mem_exceeded() {
        let limits = cuda_like_limits();
        // log_pass=10, z_count=16: 16 * 1024 * 4 = 65536 > 49152
        let plan = NttPlan {
            log_n: 20,
            passes: vec![
                PassConfig { log_pass: 10, stage_offset: 0, z_count: 16, log_wg: 8 },
                PassConfig { log_pass: 10, stage_offset: 10, z_count: 8, log_wg: 8 },
            ],
            sub_batch: 4,
        };
        let errs = validate_plan(&plan, &limits).unwrap_err();
        assert!(errs.iter().any(|e| matches!(e, PlanError::SharedMemExceeded { .. })));
    }

    #[test]
    fn validate_z_count_exceeds_n_other() {
        let limits = cuda_like_limits();
        // Single pass: n_other = 1, z_count must be 1.
        let plan = NttPlan {
            log_n: 8,
            passes: vec![
                PassConfig { log_pass: 8, stage_offset: 0, z_count: 2, log_wg: 7 },
            ],
            sub_batch: 1,
        };
        let errs = validate_plan(&plan, &limits).unwrap_err();
        assert!(errs.iter().any(|e| matches!(e, PlanError::ZCountExceedsNOther { .. })));
    }

    #[test]
    fn heuristic_prefers_fewer_passes_when_z_ok() {
        // 2-pass with z=8, wg=9 should score better than 3-pass with z=8, wg=6.
        let plan2 = NttPlan {
            log_n: 20,
            passes: vec![
                PassConfig { log_pass: 10, stage_offset: 0, z_count: 8, log_wg: 9 },
                PassConfig { log_pass: 10, stage_offset: 10, z_count: 8, log_wg: 9 },
            ],
            sub_batch: 1,
        };
        let plan3 = NttPlan {
            log_n: 20,
            passes: vec![
                PassConfig { log_pass: 7, stage_offset: 0, z_count: 8, log_wg: 6 },
                PassConfig { log_pass: 7, stage_offset: 7, z_count: 8, log_wg: 6 },
                PassConfig { log_pass: 6, stage_offset: 14, z_count: 8, log_wg: 5 },
            ],
            sub_batch: 1,
        };
        assert!(heuristic_score(&plan2) < heuristic_score(&plan3));
    }

    #[test]
    fn heuristic_penalizes_low_wg() {
        // Same plan but wg=9 should beat wg=6 (log_wg is the dominant factor).
        let plan_wg9 = NttPlan {
            log_n: 20,
            passes: vec![
                PassConfig { log_pass: 10, stage_offset: 0, z_count: 8, log_wg: 9 },
                PassConfig { log_pass: 10, stage_offset: 10, z_count: 8, log_wg: 9 },
            ],
            sub_batch: 1,
        };
        let plan_wg6 = NttPlan {
            log_n: 20,
            passes: vec![
                PassConfig { log_pass: 10, stage_offset: 0, z_count: 8, log_wg: 6 },
                PassConfig { log_pass: 10, stage_offset: 10, z_count: 8, log_wg: 6 },
            ],
            sub_batch: 1,
        };
        assert!(heuristic_score(&plan_wg9) < heuristic_score(&plan_wg6));
    }

    #[test]
    fn heuristic_penalizes_low_z() {
        // Same log_wg, z=8 should beat z=1.
        let plan_z8 = NttPlan {
            log_n: 20,
            passes: vec![
                PassConfig { log_pass: 10, stage_offset: 0, z_count: 8, log_wg: 9 },
                PassConfig { log_pass: 10, stage_offset: 10, z_count: 8, log_wg: 9 },
            ],
            sub_batch: 1,
        };
        let plan_z1 = NttPlan {
            log_n: 20,
            passes: vec![
                PassConfig { log_pass: 10, stage_offset: 0, z_count: 1, log_wg: 9 },
                PassConfig { log_pass: 10, stage_offset: 10, z_count: 1, log_wg: 9 },
            ],
            sub_batch: 1,
        };
        assert!(heuristic_score(&plan_z8) < heuristic_score(&plan_z1));
    }

    #[test]
    fn plan_heuristic_produces_valid_plan() {
        let limits = cuda_like_limits();
        for log_n in 1..=24u32 {
            let plan = plan_heuristic(log_n, 10, &limits);
            assert!(
                validate_plan(&plan, &limits).is_ok(),
                "heuristic produced invalid plan for log_n={log_n}: {:?}",
                validate_plan(&plan, &limits).unwrap_err()
            );
        }
    }

    #[test]
    fn plan_heuristic_single_pass_for_small() {
        let limits = cuda_like_limits();
        for log_n in 1..=10u32 {
            let plan = plan_heuristic(log_n, 1, &limits);
            assert_eq!(plan.passes.len(), 1, "expected single pass for log_n={log_n}");
            assert_eq!(plan.passes[0].z_count, 1, "single pass must have z=1");
        }
    }

    #[test]
    fn best_z_count_respects_shared_mem() {
        let limits = cuda_like_limits();
        // log_pass=10: n_pass = 1024, bytes = 4096. 48K/4K = 12 → floor to 8.
        // Enumeration cap=32: returns 8. Heuristic cap=8: returns 8.
        assert_eq!(best_z_count(10, 20, &limits), 8);
        // log_pass=12: n_pass = 4096, bytes = 16384. 48K/16K = 3 → floor to 2.
        assert_eq!(best_z_count(12, 20, &limits), 2);
        // log_pass=14: n_pass = 16384, bytes = 65536 > 48K → 0 (can't fit).
        assert_eq!(best_z_count(14, 20, &limits), 0);
    }

    #[test]
    fn best_z_count_respects_n_other() {
        let limits = cuda_like_limits();
        // log_n=12, log_pass=10: n_other = 4. Shared allows 8 but n_other caps at 4.
        assert_eq!(best_z_count(10, 12, &limits), 4);
    }

    #[test]
    fn heuristic_z_count_caps_at_8() {
        // On a device with lots of shared mem, heuristic still caps z at 8.
        let big_shared = DeviceLimits {
            max_shared_mem_bytes: 100 * 1024,
            max_threads_per_wg: 1024,
            scratch_bytes: 64 * 1024 * 1024,
        };
        // log_pass=10: 100K / 4K = 25 → best_z_count returns 16, but
        // heuristic_z_count caps at 8.
        assert_eq!(best_z_count(10, 20, &big_shared), 16);
        assert_eq!(heuristic_z_count(10, 20, &big_shared), 8);
    }

    #[test]
    fn sub_batch_capped_by_scratch() {
        let limits = DeviceLimits {
            max_shared_mem_bytes: 49152,
            max_threads_per_wg: 1024,
            scratch_bytes: 4 * 1024 * 1024, // 4 MiB
        };
        // log_n=20: N = 1M elements = 4 MiB. Only 1 fits in scratch.
        let plan = plan_heuristic(20, 100, &limits);
        assert_eq!(plan.sub_batch, 1);
    }
}
