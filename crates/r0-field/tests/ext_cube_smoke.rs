//! End-to-end cube test: launches a generic kernel `<F: ExtField>`,
//! computes `out[i] = (a[i] + b[i]) * b[i]` element-wise on a
//! transposed-layout buffer, and compares against a host loop.

use cubecl::prelude::*;

use r0_cube::{Device, Runtime};
use r0_field::{
    BabyBear5Parameters, Ext4, Ext5, ExtField, KoalaBear4Parameters, MontyField, MontyParameters,
};

#[cube(launch_unchecked)]
fn add_then_mul<F: ExtField>(
    a: &Array<u32>, b: &Array<u32>, out: &mut Array<u32>, #[comptime] n: u32,
) {
    let i = ABSOLUTE_POS as u32;
    if i < n {
        let av = F::load(a, 0u32, i, n);
        let bv = F::load(b, 0u32, i, n);
        F::store(out, 0u32, i, n, F::mul(F::add(av, bv), bv));
    }
}

fn run<F: ExtField>(a: &[u32], b: &[u32], n: u32) -> Vec<u32> {
    let device = Device::<Runtime>::acquire();
    let client = device.client();

    let a_h = client.create_from_slice(u32::as_bytes(a));
    let b_h = client.create_from_slice(u32::as_bytes(b));
    let out_h = client.empty(a.len() * core::mem::size_of::<u32>());

    let block_size = 64u32;
    let num_blocks = (n + block_size - 1) / block_size;

    unsafe {
        add_then_mul::launch_unchecked::<F, Runtime>(
            &client, CubeCount::Static(num_blocks, 1, 1), CubeDim::new_1d(block_size),
            ArrayArg::from_raw_parts::<u32>(&a_h, a.len(), 1),
            ArrayArg::from_raw_parts::<u32>(&b_h, b.len(), 1),
            ArrayArg::from_raw_parts::<u32>(&out_h, a.len(), 1),
            n,
        ).expect("launch failed");
    }

    let bytes = client.read_one(out_h);
    u32::from_bytes(&bytes).to_vec()
}

fn host_addsq_ext4<P: r0_field::BinomialExt4Parameters>(a: &[u32], b: &[u32], n: usize) -> Vec<u32> {
    let mut out = vec![0u32; n * 4];
    for i in 0..n {
        let av = Ext4::<P>::from_raw([a[i], a[n + i], a[2 * n + i], a[3 * n + i]]);
        let bv = Ext4::<P>::from_raw([b[i], b[n + i], b[2 * n + i], b[3 * n + i]]);
        let raw = ((av + bv) * bv).raw();
        out[i] = raw[0]; out[n + i] = raw[1]; out[2 * n + i] = raw[2]; out[3 * n + i] = raw[3];
    }
    out
}

fn host_addsq_ext5<P: r0_field::BinomialExt5Parameters>(a: &[u32], b: &[u32], n: usize) -> Vec<u32> {
    let mut out = vec![0u32; n * 5];
    for i in 0..n {
        let av = Ext5::<P>::from_raw([a[i], a[n + i], a[2 * n + i], a[3 * n + i], a[4 * n + i]]);
        let bv = Ext5::<P>::from_raw([b[i], b[n + i], b[2 * n + i], b[3 * n + i], b[4 * n + i]]);
        let raw = ((av + bv) * bv).raw();
        out[i] = raw[0]; out[n + i] = raw[1]; out[2 * n + i] = raw[2]; out[3 * n + i] = raw[3]; out[4 * n + i] = raw[4];
    }
    out
}

fn make_input_ext<P: MontyParameters>(n: usize, degree: usize, seed: u32) -> Vec<u32> {
    let mut buf = Vec::with_capacity(n * degree);
    for c in 0..degree {
        for i in 0..n {
            let canonical = (i as u32).wrapping_mul(0x9E37_79B1).wrapping_add(seed).wrapping_add(c as u32 * 0xC0FF_EE17);
            buf.push(MontyField::<P>::from_canonical(canonical).raw());
        }
    }
    buf
}

const N: u32 = 64;

#[test]
fn ext4_kb4() {
    let n = N as usize;
    let a = make_input_ext::<<KoalaBear4Parameters as r0_field::BinomialExt4Parameters>::Base>(n, 4, 0x3333_3333);
    let b = make_input_ext::<<KoalaBear4Parameters as r0_field::BinomialExt4Parameters>::Base>(n, 4, 0x4444_4444);
    let actual = run::<Ext4<KoalaBear4Parameters>>(&a, &b, N);
    let expected = host_addsq_ext4::<KoalaBear4Parameters>(&a, &b, n);
    assert_eq!(actual, expected, "Ext4<KB4>");
}

#[test]
fn ext5_bb5() {
    let n = N as usize;
    let a = make_input_ext::<<BabyBear5Parameters as r0_field::BinomialExt5Parameters>::Base>(n, 5, 0x5555_5555);
    let b = make_input_ext::<<BabyBear5Parameters as r0_field::BinomialExt5Parameters>::Base>(n, 5, 0x6666_6666);
    let actual = run::<Ext5<BabyBear5Parameters>>(&a, &b, N);
    let expected = host_addsq_ext5::<BabyBear5Parameters>(&a, &b, n);
    assert_eq!(actual, expected, "Ext5<BB5>");
}
