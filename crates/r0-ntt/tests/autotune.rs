//! Autotune: enumerate valid plans, benchmark each, report results.
//!
//! Run with:
//!   cargo test -p r0-ntt --features unstable-planner --test autotune -- \
//!       --ignored --nocapture

#![cfg(feature = "unstable-planner")]

use std::time::{Duration, Instant};

use cubecl::prelude::*;

use r0_cube::Device;
use r0_field::{BabyBearParameters, MontyParameters};
use r0_ntt::{enumerate_valid_plans, heuristic_score, NttExec, NttPlan};

fn autotune<P: MontyParameters, R: Runtime>(
    log_n: u32,
    batch: usize,
    warmup: usize,
    samples: usize,
    max_passes: u32,
    max_trials: Option<usize>,
) -> Vec<(NttPlan, Duration)>
where
    R::Device: Default,
    P: Send + Sync + 'static,
    R: 'static,
    R::Device: Send,
{
    let device = Device::<R>::acquire();
    let exec = NttExec::<P, R>::new(&device);
    let limits = exec.limits().clone();

    eprintln!("Device limits: {:?}", limits);

    // Enumerate valid plans.
    let mut plans = enumerate_valid_plans(log_n, batch, &limits, max_passes);
    eprintln!("Enumerated {} valid plans", plans.len());

    // Sort by heuristic score (best predicted first).
    plans.sort_by(|a, b| {
        heuristic_score(a)
            .partial_cmp(&heuristic_score(b))
            .unwrap()
    });

    if let Some(max) = max_trials {
        plans.truncate(max);
    }

    let num_plans = plans.len();
    eprintln!("Benchmarking {} plans...", num_plans);

    // Run benchmarks on a thread with a large stack to handle cubecl's
    // JIT compilation (each unique comptime tuple expands unrolled IR on
    // the stack; log_pass=10 with z=16 can use several MB per compilation).
    let handle = std::thread::Builder::new()
        .name("autotune-worker".into())
        .stack_size(64 * 1024 * 1024) // 64 MB
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
                // Warmup.
                for _ in 0..warmup {
                    exec.forward_with_plan(&buf, &plan, batch);
                    cubecl_common::reader::read_sync(client.sync()).expect("sync failed");
                }

                // Measure (take minimum).
                let mut best_time = Duration::MAX;
                for _ in 0..samples {
                    let start = Instant::now();
                    exec.forward_with_plan(&buf, &plan, batch);
                    cubecl_common::reader::read_sync(client.sync()).expect("sync failed");
                    let elapsed = start.elapsed();
                    best_time = best_time.min(elapsed);
                }

                if best_time < global_best {
                    global_best = best_time;
                    eprintln!(
                        "  [{:4}/{}] NEW BEST: {:>8.1?}  passes={} {:?}",
                        i + 1,
                        num_plans,
                        best_time,
                        plan.passes.len(),
                        plan.passes
                            .iter()
                            .map(|p| format!(
                                "(lp={} z={} wg={})",
                                p.log_pass, p.z_count, p.log_wg
                            ))
                            .collect::<Vec<_>>()
                            .join(" "),
                    );
                }

                results.push((plan, best_time));
            }

            results.sort_by_key(|(_, t)| *t);
            results
        })
        .expect("failed to spawn autotune thread");

    handle.join().expect("autotune thread panicked")
}

fn print_results(results: &[(NttPlan, Duration)], top_n: usize) {
    eprintln!("\n=== Top {} plans ===", top_n.min(results.len()));
    eprintln!(
        "{:>4}  {:>10}  {:>6}  {:>5}  {}",
        "Rank", "Time", "Score", "Batch", "Passes"
    );
    eprintln!("{}", "-".repeat(80));

    for (i, (plan, time)) in results.iter().take(top_n).enumerate() {
        let pass_desc: Vec<String> = plan
            .passes
            .iter()
            .map(|p| {
                format!(
                    "(lp={:>2} z={:>2} wg={:>2})",
                    p.log_pass, p.z_count, p.log_wg
                )
            })
            .collect();
        eprintln!(
            "{:>4}  {:>10.1?}  {:>6.1}  {:>5}  {}",
            i + 1,
            time,
            heuristic_score(plan),
            plan.sub_batch,
            pass_desc.join(" "),
        );
    }

    if results.len() > top_n {
        eprintln!("\n=== Bottom 5 plans ===");
        for (plan, time) in results.iter().rev().take(5).collect::<Vec<_>>().into_iter().rev() {
            let pass_desc: Vec<String> = plan
                .passes
                .iter()
                .map(|p| {
                    format!(
                        "(lp={:>2} z={:>2} wg={:>2})",
                        p.log_pass, p.z_count, p.log_wg
                    )
                })
                .collect();
            eprintln!(
                "      {:>10.1?}  {:>6.1}  {:>5}  {}",
                time,
                heuristic_score(plan),
                plan.sub_batch,
                pass_desc.join(" "),
            );
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn autotune_cuda_log20_batch32() {
    type P = BabyBearParameters;
    type R = cubecl::cuda::CudaRuntime;

    eprintln!("\n=== Autotuning: log_n=20, batch=32, BabyBear, CUDA ===\n");

    let results = autotune::<P, R>(20, 32, 3, 10, 2, None);
    print_results(&results, 20);

    let (best_plan, best_time) = &results[0];
    eprintln!(
        "\nBest: {:?} in {:?}",
        best_plan.passes
            .iter()
            .map(|p| format!(
                "lp={} z={} wg={}",
                p.log_pass, p.z_count, p.log_wg
            ))
            .collect::<Vec<_>>(),
        best_time
    );
}

#[cfg(feature = "wgpu")]
#[test]
#[ignore]
fn autotune_wgpu_log20_batch32() {
    type P = BabyBearParameters;
    type R = cubecl::wgpu::WgpuRuntime;

    eprintln!("\n=== Autotuning: log_n=20, batch=32, BabyBear, wgpu ===\n");

    let results = autotune::<P, R>(20, 32, 3, 10, 2, None);
    print_results(&results, 20);

    let (best_plan, best_time) = &results[0];
    eprintln!(
        "\nBest: {:?} in {:?}",
        best_plan.passes
            .iter()
            .map(|p| format!(
                "lp={} z={} wg={}",
                p.log_pass, p.z_count, p.log_wg
            ))
            .collect::<Vec<_>>(),
        best_time
    );
}
