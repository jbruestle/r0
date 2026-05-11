//! Quick diagnostic: dump device info, limits, and heuristic plans.
//!
//! Run with:
//!   cargo test -p r0-ntt --features cuda,unstable-planner --test diagnostics -- \
//!       --ignored --nocapture

#![cfg(feature = "unstable-planner")]

use r0_cube::{Device, Runtime};
use r0_field::BabyBearParameters;
use r0_ntt::{plan_heuristic, NttExec};

#[test]
#[ignore]
fn dump_device() {
    let device = Device::<Runtime>::acquire();
    let exec = NttExec::<BabyBearParameters, Runtime>::new(&device);
    let limits = exec.limits();
    let client = device.client();
    let props = client.properties();

    eprintln!("=== Device ===");
    eprintln!("  max_shared_memory_size: {} bytes ({} KiB)",
        props.hardware.max_shared_memory_size,
        props.hardware.max_shared_memory_size / 1024);
    eprintln!("  max_units_per_cube:     {}", props.hardware.max_units_per_cube);
    eprintln!("  plane_size_max:         {}", props.hardware.plane_size_max);
    if let Some(sms) = props.hardware.num_streaming_multiprocessors {
        eprintln!("  SMs:                    {sms}");
    }
    eprintln!("  DeviceLimits: {:?}", limits);

    for log_n in [20u32, 22] {
        let plan = plan_heuristic(log_n, 1, limits);
        eprintln!("  Heuristic plan for log_n={log_n}, batch=1:");
        for (i, p) in plan.passes.iter().enumerate() {
            eprintln!("    pass {i}: log_pass={} z_count={} log_wg={} stage_offset={}",
                p.log_pass, p.z_count, p.log_wg, p.stage_offset);
        }
        eprintln!("    sub_batch: {}", plan.sub_batch);
    }
}
