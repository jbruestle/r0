//! Quick diagnostic: dump device info, limits, and heuristic plans for each runtime.
//!
//! Run with:
//!   cargo test -p r0-ntt --features unstable-planner --test diagnostics -- \
//!       --ignored --nocapture

#![cfg(feature = "unstable-planner")]

use cubecl::prelude::*;
use r0_field::{BabyBearParameters, Device};
use r0_ntt::{plan_heuristic, NttExec};

fn dump_runtime<R: Runtime>(label: &str)
where
    R::Device: Default,
{
    let device = Device::<R>::acquire();
    let exec = NttExec::<BabyBearParameters, R>::new(&device, 0);
    let limits = exec.limits();
    let client = R::client(device.inner());
    let props = client.properties();

    eprintln!("=== {label} ===");
    eprintln!("  Hardware:");
    eprintln!("    max_shared_memory_size: {} bytes ({} KiB)",
        props.hardware.max_shared_memory_size,
        props.hardware.max_shared_memory_size / 1024);
    eprintln!("    max_units_per_cube:     {}", props.hardware.max_units_per_cube);
    eprintln!("    max_cube_dim:           {:?}", props.hardware.max_cube_dim);
    eprintln!("    max_cube_count:         {:?}", props.hardware.max_cube_count);
    eprintln!("    plane_size_min:         {}", props.hardware.plane_size_min);
    eprintln!("    plane_size_max:         {}", props.hardware.plane_size_max);
    if let Some(sms) = props.hardware.num_streaming_multiprocessors {
        eprintln!("    SMs:                    {sms}");
    }

    eprintln!("  DeviceLimits used by planner:");
    eprintln!("    max_shared_mem_bytes:   {}", limits.max_shared_mem_bytes);
    eprintln!("    max_threads_per_wg:     {}", limits.max_threads_per_wg);
    eprintln!("    scratch_bytes:          {}", limits.scratch_bytes);

    for log_n in [20u32, 22] {
        let plan = plan_heuristic(log_n, 1, limits);
        eprintln!("  Heuristic plan for log_n={log_n}, batch=1:");
        for (i, p) in plan.passes.iter().enumerate() {
            eprintln!("    pass {i}: log_pass={} z_count={} log_wg={} stage_offset={}",
                p.log_pass, p.z_count, p.log_wg, p.stage_offset);
        }
        eprintln!("    sub_batch: {}", plan.sub_batch);
    }
    eprintln!();
}

#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn dump_cuda() {
    dump_runtime::<cubecl::cuda::CudaRuntime>("CUDA");
}

#[cfg(feature = "wgpu")]
#[test]
#[ignore]
fn dump_wgpu() {
    // List all wgpu adapters to see what devices are available.
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    let adapters: Vec<_> = instance.enumerate_adapters(wgpu::Backends::all());
    eprintln!("=== wgpu adapters ({}) ===", adapters.len());
    for (i, adapter) in adapters.iter().enumerate() {
        let info = adapter.get_info();
        eprintln!("  [{i}] name={:?} vendor=0x{:04x} device=0x{:04x} type={:?} backend={:?}",
            info.name, info.vendor, info.device, info.device_type, info.backend);
        let limits = adapter.limits();
        eprintln!("       max_compute_workgroup_storage_size={} max_subgroup_size={}",
            limits.max_compute_workgroup_storage_size, limits.max_subgroup_size);
    }
    eprintln!();

    dump_runtime::<cubecl::wgpu::WgpuRuntime>("wgpu (DefaultDevice)");
}
