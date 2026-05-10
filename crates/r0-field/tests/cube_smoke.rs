//! First end-to-end exercise of the `#[cube]` IR path. Launches a
//! generic element-wise `monty_mul` kernel on cubecl-cpu and asserts
//! its output matches the host `monty_mul` byte-for-byte. Runs for
//! both BabyBear and KoalaBear to verify trait-generic kernel dispatch.

use cubecl::cpu::CpuRuntime;
use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;

use r0_field::{
    monty_mul, BabyBear, BabyBearParameters, Device, KoalaBearParameters, MontyField,
    MontyParameters,
};

#[cube(launch_unchecked)]
fn vec_monty_mul<P: MontyParameters>(a: &Array<u32>, b: &Array<u32>, out: &mut Array<u32>) {
    if ABSOLUTE_POS < a.len() {
        out[ABSOLUTE_POS] = monty_mul::<P>(a[ABSOLUTE_POS], b[ABSOLUTE_POS]);
    }
}

fn run_vec_monty_mul<P: MontyParameters, R: Runtime>(
    device: &Device<R>,
    a: &[u32],
    b: &[u32],
) -> Vec<u32> {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    let client = R::client(device.inner());

    let a_handle = client.create_from_slice(u32::as_bytes(a));
    let b_handle = client.create_from_slice(u32::as_bytes(b));
    let out_handle = client.empty(n * core::mem::size_of::<u32>());

    let block_size: u32 = 64;
    let num_blocks: u32 = ((n as u32) + block_size - 1) / block_size;

    unsafe {
        vec_monty_mul::launch_unchecked::<P, R>(
            &client,
            CubeCount::Static(num_blocks, 1, 1),
            CubeDim::new_1d(block_size),
            ArrayArg::from_raw_parts::<u32>(&a_handle, n, 1),
            ArrayArg::from_raw_parts::<u32>(&b_handle, n, 1),
            ArrayArg::from_raw_parts::<u32>(&out_handle, n, 1),
        )
        .expect("kernel launch failed");
    }

    let bytes = client.read_one(out_handle);
    u32::from_bytes(&bytes).to_vec()
}

/// Build two pseudo-random vectors of `n` Montgomery-form values for
/// field `P`, run the kernel on runtime `R`, and assert agreement with
/// host.
fn check_vec_mul<P: MontyParameters, R: Runtime>(n: usize)
where
    R::Device: Default,
{
    // Deterministic but non-trivial inputs.
    let a: Vec<u32> = (0..n)
        .map(|i| MontyField::<P>::from_canonical((i as u32).wrapping_mul(0x9E3779B1) ^ 0xDEADBEEF).raw())
        .collect();
    let b: Vec<u32> = (0..n)
        .map(|i| MontyField::<P>::from_canonical((i as u32).wrapping_mul(0x85EBCA77) ^ 0xC2B2AE3D).raw())
        .collect();

    let expected: Vec<u32> = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| monty_mul::<P>(x, y))
        .collect();

    let device = Device::<R>::acquire();
    let actual = run_vec_monty_mul::<P, R>(&device, &a, &b);

    assert_eq!(actual.len(), n);
    assert_eq!(actual, expected);
}

#[test]
fn babybear_vec_mul_cpu_runtime() {
    check_vec_mul::<BabyBearParameters, CpuRuntime>(1024);
}

#[test]
fn koalabear_vec_mul_cpu_runtime() {
    check_vec_mul::<KoalaBearParameters, CpuRuntime>(1024);
}

#[test]
fn babybear_vec_mul_wgpu_runtime() {
    check_vec_mul::<BabyBearParameters, WgpuRuntime>(1024);
}

#[test]
fn koalabear_vec_mul_wgpu_runtime() {
    check_vec_mul::<KoalaBearParameters, WgpuRuntime>(1024);
}

/// Spot-check: kernel agrees with `MontyField`'s operator overload on
/// a hand-picked case (verifies the round-trip from canonical → raw
/// → kernel → raw → canonical isn't lying somewhere).
#[test]
fn babybear_vec_mul_spot_check() {
    let n = 8;
    let a_field: Vec<BabyBear> = (1..=n).map(|i| BabyBear::from_canonical(i as u32)).collect();
    let b_field: Vec<BabyBear> = (1..=n).map(|i| BabyBear::from_canonical((100 + i) as u32)).collect();

    let a: Vec<u32> = a_field.iter().map(|x| x.raw()).collect();
    let b: Vec<u32> = b_field.iter().map(|x| x.raw()).collect();

    let device = Device::<CpuRuntime>::acquire();
    let actual_raw = run_vec_monty_mul::<BabyBearParameters, CpuRuntime>(&device, &a, &b);

    for (i, &raw) in actual_raw.iter().enumerate() {
        let kernel_result = BabyBear::from_raw(raw).to_canonical();
        let expected = (a_field[i] * b_field[i]).to_canonical();
        let pure_canonical_product = ((i + 1) as u64 * (100 + i + 1) as u64
            % BabyBearParameters::PRIME as u64) as u32;
        assert_eq!(kernel_result, expected);
        assert_eq!(kernel_result, pure_canonical_product);
    }
}
