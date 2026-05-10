//! End-to-end cube test: launches a generic kernel `<F: ExtField>` on
//! both CpuRuntime and WgpuRuntime for all five field instances, computes
//! `out[i] = (a[i] + b[i]) * b[i]` element-wise on a transposed-layout
//! buffer, and compares against a host loop using `MontyField` /
//! `Ext4` / `Ext5` host operators (already cross-checked against Plonky3
//! by the sibling oracle test).
//!
//! Failures from this would indicate one of:
//!   - `#[cube] trait ExtField` dispatch through `launch_unchecked` is
//!     misbehaving under monomorphization,
//!   - the `c·N + i` transposed load/store math is wrong,
//!   - or a backend codegen surprise (cubecl 0.9 has a few — e.g. `%`
//!     was broken on Metal in our earlier smoke).

use cubecl::cpu::CpuRuntime;
use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;

use r0_cube::Device;
use r0_field::{
    BabyBear4, BabyBear4Parameters, BabyBear5, BabyBear5Parameters, BabyBearParameters, BaseElem,
    Ext4, Ext5, ExtField, KoalaBear4, KoalaBear4Parameters, KoalaBearParameters, MontyField,
    MontyParameters,
};

// ---------------------------------------------------------------------------
// Generic kernel: out[i] = (a[i] + b[i]) * b[i] for each logical i in 0..n.
// Buffers are in transposed layout: component c of element i is at base+c*n+i.
// ---------------------------------------------------------------------------

#[cube(launch_unchecked)]
fn add_then_mul<F: ExtField>(
    a: &Array<u32>,
    b: &Array<u32>,
    out: &mut Array<u32>,
    #[comptime] n: u32,
) {
    let i = ABSOLUTE_POS as u32;
    if i < n {
        let av = F::load(a, 0u32, i, n);
        let bv = F::load(b, 0u32, i, n);
        let s = F::add(av, bv);
        let p = F::mul(s, bv);
        F::store(out, 0u32, i, n, p);
    }
}

fn run<F: ExtField, R: Runtime>(a: &[u32], b: &[u32], n: u32) -> Vec<u32>
where
    R::Device: Default,
{
    assert_eq!(a.len(), b.len());
    let device = Device::<R>::acquire();
    let client = R::client(device.inner());

    let a_h = client.create_from_slice(u32::as_bytes(a));
    let b_h = client.create_from_slice(u32::as_bytes(b));
    let out_h = client.empty(a.len() * core::mem::size_of::<u32>());

    let block_size = 64u32;
    let num_blocks = (n + block_size - 1) / block_size;

    unsafe {
        add_then_mul::launch_unchecked::<F, R>(
            &client,
            CubeCount::Static(num_blocks, 1, 1),
            CubeDim::new_1d(block_size),
            ArrayArg::from_raw_parts::<u32>(&a_h, a.len(), 1),
            ArrayArg::from_raw_parts::<u32>(&b_h, b.len(), 1),
            ArrayArg::from_raw_parts::<u32>(&out_h, a.len(), 1),
            n,
        )
        .expect("launch failed");
    }

    let bytes = client.read_one(out_h);
    u32::from_bytes(&bytes).to_vec()
}

// ---------------------------------------------------------------------------
// Host references. Same algebra as the kernel, on the host.
// ---------------------------------------------------------------------------

fn host_addsq_base<P: MontyParameters>(a: &[u32], b: &[u32]) -> Vec<u32> {
    a.iter()
        .zip(b)
        .map(|(&ax, &bx)| {
            let av = MontyField::<P>::from_raw(ax);
            let bv = MontyField::<P>::from_raw(bx);
            ((av + bv) * bv).raw()
        })
        .collect()
}

fn host_addsq_ext4<P: r0_field::BinomialExt4Parameters>(a: &[u32], b: &[u32], n: usize) -> Vec<u32> {
    let mut out = vec![0u32; n * 4];
    for i in 0..n {
        let av = Ext4::<P>::from_raw([a[i], a[n + i], a[2 * n + i], a[3 * n + i]]);
        let bv = Ext4::<P>::from_raw([b[i], b[n + i], b[2 * n + i], b[3 * n + i]]);
        let raw = ((av + bv) * bv).raw();
        out[i] = raw[0];
        out[n + i] = raw[1];
        out[2 * n + i] = raw[2];
        out[3 * n + i] = raw[3];
    }
    out
}

fn host_addsq_ext5<P: r0_field::BinomialExt5Parameters>(a: &[u32], b: &[u32], n: usize) -> Vec<u32> {
    let mut out = vec![0u32; n * 5];
    for i in 0..n {
        let av = Ext5::<P>::from_raw([
            a[i],
            a[n + i],
            a[2 * n + i],
            a[3 * n + i],
            a[4 * n + i],
        ]);
        let bv = Ext5::<P>::from_raw([
            b[i],
            b[n + i],
            b[2 * n + i],
            b[3 * n + i],
            b[4 * n + i],
        ]);
        let raw = ((av + bv) * bv).raw();
        out[i] = raw[0];
        out[n + i] = raw[1];
        out[2 * n + i] = raw[2];
        out[3 * n + i] = raw[3];
        out[4 * n + i] = raw[4];
    }
    out
}

// ---------------------------------------------------------------------------
// Input generation. Produces deterministic but non-trivial Montgomery-form
// inputs spanning all components.
// ---------------------------------------------------------------------------

fn make_input_base<P: MontyParameters>(n: usize, seed: u32) -> Vec<u32> {
    (0..n)
        .map(|i| {
            MontyField::<P>::from_canonical(
                (i as u32)
                    .wrapping_mul(0x9E37_79B1)
                    .wrapping_add(seed),
            )
            .raw()
        })
        .collect()
}

fn make_input_ext<P: MontyParameters>(n: usize, degree: usize, seed: u32) -> Vec<u32> {
    // Transposed layout: D contiguous blocks of N, each filled with
    // pseudo-random Montgomery-form values seeded by component index.
    let mut buf = Vec::with_capacity(n * degree);
    for c in 0..degree {
        for i in 0..n {
            let canonical = (i as u32)
                .wrapping_mul(0x9E37_79B1)
                .wrapping_add(seed)
                .wrapping_add(c as u32 * 0xC0FF_EE17);
            buf.push(MontyField::<P>::from_canonical(canonical).raw());
        }
    }
    buf
}

// ---------------------------------------------------------------------------
// Cases.
// ---------------------------------------------------------------------------

const N: u32 = 64; // small enough to be fast, large enough to span warps

fn check_base<P: MontyParameters, R: Runtime>()
where
    R::Device: Default,
{
    let n = N as usize;
    let a = make_input_base::<P>(n, 0x1111_1111);
    let b = make_input_base::<P>(n, 0x2222_2222);
    let actual = run::<BaseElem<P>, R>(&a, &b, N);
    let expected = host_addsq_base::<P>(&a, &b);
    assert_eq!(actual, expected, "BaseElem<{}>", core::any::type_name::<P>());
}

fn check_ext4<P: r0_field::BinomialExt4Parameters, R: Runtime>()
where
    R::Device: Default,
{
    let n = N as usize;
    let a = make_input_ext::<P::Base>(n, 4, 0x3333_3333);
    let b = make_input_ext::<P::Base>(n, 4, 0x4444_4444);
    let actual = run::<Ext4<P>, R>(&a, &b, N);
    let expected = host_addsq_ext4::<P>(&a, &b, n);
    assert_eq!(actual, expected, "Ext4<{}>", core::any::type_name::<P>());
}

fn check_ext5<P: r0_field::BinomialExt5Parameters, R: Runtime>()
where
    R::Device: Default,
{
    let n = N as usize;
    let a = make_input_ext::<P::Base>(n, 5, 0x5555_5555);
    let b = make_input_ext::<P::Base>(n, 5, 0x6666_6666);
    let actual = run::<Ext5<P>, R>(&a, &b, N);
    let expected = host_addsq_ext5::<P>(&a, &b, n);
    assert_eq!(actual, expected, "Ext5<{}>", core::any::type_name::<P>());
}

// --- CpuRuntime ---

#[test]
fn base_bb_cpu() {
    check_base::<BabyBearParameters, CpuRuntime>();
}
#[test]
fn base_kb_cpu() {
    check_base::<KoalaBearParameters, CpuRuntime>();
}
#[test]
fn ext4_bb4_cpu() {
    check_ext4::<BabyBear4Parameters, CpuRuntime>();
    let _ = BabyBear4::default(); // touch type alias to ensure it compiles
}
#[test]
fn ext4_kb4_cpu() {
    check_ext4::<KoalaBear4Parameters, CpuRuntime>();
    let _ = KoalaBear4::default();
}
#[test]
fn ext5_bb5_cpu() {
    check_ext5::<BabyBear5Parameters, CpuRuntime>();
    let _ = BabyBear5::default();
}

// --- WgpuRuntime ---

#[test]
fn base_bb_wgpu() {
    check_base::<BabyBearParameters, WgpuRuntime>();
}
#[test]
fn base_kb_wgpu() {
    check_base::<KoalaBearParameters, WgpuRuntime>();
}
#[test]
fn ext4_bb4_wgpu() {
    check_ext4::<BabyBear4Parameters, WgpuRuntime>();
}
#[test]
fn ext4_kb4_wgpu() {
    check_ext4::<KoalaBear4Parameters, WgpuRuntime>();
}
#[test]
fn ext5_bb5_wgpu() {
    check_ext5::<BabyBear5Parameters, WgpuRuntime>();
}
