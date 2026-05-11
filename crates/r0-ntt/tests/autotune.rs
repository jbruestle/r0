//! Autotune: enumerate valid plans, benchmark each, report results.
//!
//! Run with:
//!   cargo test -p r0-ntt --features cuda,unstable-planner --test autotune -- \
//!       --ignored --nocapture

#![cfg(feature = "unstable-planner")]

use std::time::{Duration, Instant};

use cubecl::prelude::*;

use r0_cube::{Device, Runtime};
use r0_field::{BabyBearParameters, MontyParameters};
use r0_ntt::{enumerate_valid_plans, heuristic_score, NttExec, NttPlan};

fn autotune<P: MontyParameters + Send + Sync + 'static>(
    log_n: u32,
    batch: usize,
    warmup: usize,
    samples: usize,
    max_passes: u32,
    max_trials: Option<usize>,
) -> Vec<(NttPlan, Duration)> {
    let device = Device::<Runtime>::acquire();
    let exec = NttExec::<P, Runtime>::new(&device);
    let limits = exec.limits().clone();

    eprintln!("Device limits: {:?}", limits);

    let mut plans = enumerate_valid_plans(log_n, batch, &limits, max_passes);
    eprintln!("Enumerated {} valid plans", plans.len());

    plans.sort_by(|a, b| heuristic_score(a).partial_cmp(&heuristic_score(b)).unwrap());

    if let Some(max) = max_trials {
        plans.truncate(max);
    }

    let num_plans = plans.len();
    eprintln!("Benchmarking {} plans...", num_plans);

    let handle = std::thread::Builder::new()
        .name("autotune-worker".into())
        .stack_size(64 * 1024 * 1024)
        .spawn(move || {
            let client = exec.client();
            let n = 1usize << log_n;
            let total = batch * n;
            let input: Vec<u32> = (0..total as u32)
                .map(|i| i.wrapping_mul(0x9E3779B1) % P::PRIME)
                .collect();
            let buf = client.create_from_slice(u32::as_bytes(&input));

            let mut results: Vec<(NttPlan, Duration)> = Vec::with_capacity(num_plans);
            let mut global_best = Duration::MAX;

            for (i, plan) in plans.into_iter().enumerate() {
                for _ in 0..warmup {
                    exec.forward_with_plan(&buf, &plan, batch);
                    cubecl_common::reader::read_sync(client.sync()).expect("sync failed");
                }

                let mut best_time = Duration::MAX;
                for _ in 0..samples {
                    let start = Instant::now();
                    exec.forward_with_plan(&buf, &plan, batch);
                    cubecl_common::reader::read_sync(client.sync()).expect("sync failed");
                    best_time = best_time.min(start.elapsed());
                }

                if best_time < global_best {
                    global_best = best_time;
                    eprintln!("  [{:4}/{}] NEW BEST: {:>8.1?}  {:?}",
                        i + 1, num_plans, best_time,
                        plan.passes.iter()
                            .map(|p| format!("(lp={} z={} wg={})", p.log_pass, p.z_count, p.log_wg))
                            .collect::<Vec<_>>().join(" "));
                }

                results.push((plan, best_time));
            }

            results.sort_by_key(|(_, t)| *t);
            results
        })
        .expect("failed to spawn autotune thread");

    handle.join().expect("autotune thread panicked")
}

#[test]
#[ignore]
fn autotune_log20_batch32() {
    eprintln!("\n=== Autotuning: log_n=20, batch=32, BabyBear ===\n");
    let results = autotune::<BabyBearParameters>(20, 32, 3, 10, 2, None);

    eprintln!("\n=== Top 20 plans ===");
    for (i, (plan, time)) in results.iter().take(20).enumerate() {
        let passes: Vec<String> = plan.passes.iter()
            .map(|p| format!("(lp={:>2} z={:>2} wg={:>2})", p.log_pass, p.z_count, p.log_wg))
            .collect();
        eprintln!("{:>4}  {:>10.1?}  {:>6.1}  {}", i + 1, time, heuristic_score(plan), passes.join(" "));
    }
}
