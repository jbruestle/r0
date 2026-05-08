//! WebGPU NTT benchmark demo.
//!
//! Exposes two #[wasm_bindgen] entry points called from index.html:
//! - `diagnose()` -- initialize the wgpu device and return adapter/limit info.
//! - `run_benchmark(log_n, batch, warmups, samples)` -- run the NTT and
//!   return per-sample timings + summary stats.

use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;
use cubecl_wgpu::{init_setup_async, AutoGraphicsApi, RuntimeOptions, WgpuDevice};
use r0_field::{BabyBearParameters, MontyField, MontyParameters};
use r0_ntt::{plan_heuristic, NttExec};
use wasm_bindgen::prelude::*;
use web_time::Instant;

/// Scratch buffer size for the executor. 512 MiB lets sub_batch reach 128
/// at log_n=20 (4 MiB/poly), so batch up to 128 fits in a single ping-pong cycle.
const SCRATCH_BYTES: usize = 512 * 1024 * 1024;

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
}

/// Initialize the wgpu device and return device/limit info as JSON.
#[wasm_bindgen]
pub async fn diagnose() -> Result<JsValue, JsValue> {
    let device = WgpuDevice::DefaultDevice;
    let setup = init_setup_async::<AutoGraphicsApi>(&device, RuntimeOptions::default()).await;

    let info = setup.adapter.get_info();
    let wgpu_limits = setup.device.limits();

    let exec = NttExec::<BabyBearParameters, WgpuRuntime>::new(&device, SCRATCH_BYTES);
    let limits = exec.limits();
    let plan = plan_heuristic(20, 32, limits);
    let plan_str = plan
        .passes
        .iter()
        .map(|p| format!("(lp={} z={} wg={})", p.log_pass, p.z_count, p.log_wg))
        .collect::<Vec<_>>()
        .join(" → ");

    let json = format!(
        r#"{{
  "adapter": {{
    "name": "{name}",
    "vendor": "0x{vendor:04x}",
    "device": "0x{dev:04x}",
    "device_type": "{dtype:?}",
    "backend": "{backend:?}",
    "driver": "{driver}",
    "driver_info": "{driver_info}"
  }},
  "wgpu_limits": {{
    "max_compute_workgroup_storage_size": {wg_storage},
    "max_compute_invocations_per_workgroup": {invocations},
    "max_storage_buffer_binding_size": {sbuf},
    "max_buffer_size": {buf}
  }},
  "ntt_planner_limits": {{
    "max_shared_mem_bytes": {shmem},
    "max_threads_per_wg": {threads},
    "scratch_bytes": {scratch}
  }},
  "default_plan_log20_b32": "{plan_str}",
  "sub_batch": {sub_batch}
}}"#,
        name = info.name,
        vendor = info.vendor,
        dev = info.device,
        dtype = info.device_type,
        backend = info.backend,
        driver = info.driver,
        driver_info = info.driver_info,
        wg_storage = wgpu_limits.max_compute_workgroup_storage_size,
        invocations = wgpu_limits.max_compute_invocations_per_workgroup,
        sbuf = wgpu_limits.max_storage_buffer_binding_size,
        buf = wgpu_limits.max_buffer_size,
        shmem = limits.max_shared_mem_bytes,
        threads = limits.max_threads_per_wg,
        scratch = limits.scratch_bytes,
        sub_batch = plan.sub_batch,
    );
    Ok(JsValue::from_str(&json))
}

/// Run a forward-NTT benchmark and return timings as JSON.
#[wasm_bindgen]
pub async fn run_benchmark(
    log_n: u32,
    batch: u32,
    warmups: u32,
    samples: u32,
) -> Result<JsValue, JsValue> {
    // The device must already be registered by `diagnose()` — re-calling
    // init_setup_async panics ("server already registered").
    let device = WgpuDevice::DefaultDevice;
    let exec = NttExec::<BabyBearParameters, WgpuRuntime>::new(&device, SCRATCH_BYTES);
    let client = exec.client().clone();

    let n = 1usize << log_n;
    let total = batch as usize * n;

    // Pseudo-random Montgomery-form input (matches the bench in benches/ntt.rs).
    let input: Vec<u32> = (0..total as u32)
        .map(|i| {
            MontyField::<BabyBearParameters>::from_canonical(
                i.wrapping_mul(0x9E3779B1) % BabyBearParameters::PRIME,
            )
            .raw()
        })
        .collect();
    let buf = client.create_from_slice(u32::as_bytes(&input));

    let plan = plan_heuristic(log_n, batch as usize, exec.limits());
    let plan_str = plan
        .passes
        .iter()
        .map(|p| format!("(lp={} z={} wg={})", p.log_pass, p.z_count, p.log_wg))
        .collect::<Vec<_>>()
        .join(" → ");

    // Warmups.
    for _ in 0..warmups {
        exec.forward_with_plan(&buf, &plan, batch as usize);
        client
            .sync()
            .await
            .map_err(|e| JsValue::from_str(&format!("warmup sync failed: {e:?}")))?;
    }

    // Measured samples.
    let mut times_us = Vec::with_capacity(samples as usize);
    for _ in 0..samples {
        let t0 = Instant::now();
        exec.forward_with_plan(&buf, &plan, batch as usize);
        client
            .sync()
            .await
            .map_err(|e| JsValue::from_str(&format!("sync failed: {e:?}")))?;
        times_us.push(t0.elapsed().as_secs_f64() * 1e6);
    }

    let mut sorted = times_us.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = sorted[0];
    let max = *sorted.last().unwrap();
    let median = sorted[sorted.len() / 2];
    let mean: f64 = sorted.iter().sum::<f64>() / sorted.len() as f64;
    // Use median for per-NTT to reduce noise.
    let per_ntt_us = median / batch as f64;

    let samples_json = times_us
        .iter()
        .map(|t| format!("{t:.1}"))
        .collect::<Vec<_>>()
        .join(", ");

    let json = format!(
        r#"{{
  "log_n": {log_n},
  "batch": {batch},
  "warmups": {warmups},
  "samples": {samples},
  "plan": "{plan_str}",
  "sub_batch": {sub_batch},
  "min_us": {min:.1},
  "median_us": {median:.1},
  "mean_us": {mean:.1},
  "max_us": {max:.1},
  "per_ntt_us": {per_ntt_us:.2},
  "samples_us": [{samples_json}]
}}"#,
        sub_batch = plan.sub_batch,
    );
    Ok(JsValue::from_str(&json))
}
