//! End-to-end tests for `PolyDivExec::div_by_x_minus_z`. Cube path
//! checked against a serial host Horner reference for:
//!   - KB^4: full sweep across log_n sizes (canonical extension field)
//!   - BB base: spot-check (degree-1 path)
//!   - BB^5: spot-check (padded 16-lane Repr path)

use cubecl::prelude::*;

use r0_cube::{Device, Runtime};
use r0_field::{
    BabyBear5, BabyBearParameters, BaseElem, Ext4, Ext5, KoalaBear4, KoalaBearParameters,
    MontyField, MontyParameters,
};
use r0_polynomial::host_ref::{div_by_x_minus_z_serial, HostField};
use r0_polynomial::{PairScanLayout, PolyDivExec};

// ---------------------------------------------------------------------------
// FieldHarness: per-F bridge between the test core and the concrete types.
// ---------------------------------------------------------------------------

trait FieldHarness {
    type Cube: PairScanLayout;
    type Host: HostField + Copy;
    const DEGREE: usize;
    fn host_from_seed(seed: u64) -> Self::Host;
    fn host_to_raw(host: Self::Host) -> Vec<u32>;
    fn host_from_raw(raw: &[u32]) -> Self::Host;
}

struct HBaseBb;
impl FieldHarness for HBaseBb {
    type Cube = BaseElem<BabyBearParameters>;
    type Host = MontyField<BabyBearParameters>;
    const DEGREE: usize = 1;
    fn host_from_seed(seed: u64) -> Self::Host {
        MontyField::<BabyBearParameters>::from_canonical((seed % BabyBearParameters::PRIME as u64) as u32)
    }
    fn host_to_raw(host: Self::Host) -> Vec<u32> { vec![host.raw()] }
    fn host_from_raw(raw: &[u32]) -> Self::Host { MontyField::<BabyBearParameters>::from_raw(raw[0]) }
}

struct HExt4Kb;
impl FieldHarness for HExt4Kb {
    type Cube = Ext4<r0_field::KoalaBear4Parameters>;
    type Host = KoalaBear4;
    const DEGREE: usize = 4;
    fn host_from_seed(seed: u64) -> Self::Host {
        let p = KoalaBearParameters::PRIME as u64;
        KoalaBear4::from_canonical([
            ((seed.wrapping_mul(0x9E3779B97F4A7C15) ^ 0x1) % p) as u32,
            ((seed.wrapping_mul(0x9E3779B97F4A7C15) ^ 0x2) % p) as u32,
            ((seed.wrapping_mul(0x9E3779B97F4A7C15) ^ 0x3) % p) as u32,
            ((seed.wrapping_mul(0x9E3779B97F4A7C15) ^ 0x4) % p) as u32,
        ])
    }
    fn host_to_raw(host: Self::Host) -> Vec<u32> { host.raw().to_vec() }
    fn host_from_raw(raw: &[u32]) -> Self::Host { KoalaBear4::from_raw([raw[0], raw[1], raw[2], raw[3]]) }
}

struct HExt5Bb;
impl FieldHarness for HExt5Bb {
    type Cube = Ext5<r0_field::BabyBear5Parameters>;
    type Host = BabyBear5;
    const DEGREE: usize = 5;
    fn host_from_seed(seed: u64) -> Self::Host {
        let p = BabyBearParameters::PRIME as u64;
        BabyBear5::from_canonical([
            ((seed.wrapping_mul(0x9E3779B97F4A7C15) ^ 0x1) % p) as u32,
            ((seed.wrapping_mul(0x9E3779B97F4A7C15) ^ 0x2) % p) as u32,
            ((seed.wrapping_mul(0x9E3779B97F4A7C15) ^ 0x3) % p) as u32,
            ((seed.wrapping_mul(0x9E3779B97F4A7C15) ^ 0x4) % p) as u32,
            ((seed.wrapping_mul(0x9E3779B97F4A7C15) ^ 0x5) % p) as u32,
        ])
    }
    fn host_to_raw(host: Self::Host) -> Vec<u32> { host.raw().to_vec() }
    fn host_from_raw(raw: &[u32]) -> Self::Host { BabyBear5::from_raw([raw[0], raw[1], raw[2], raw[3], raw[4]]) }
}

// ---------------------------------------------------------------------------
// Layout helpers.
// ---------------------------------------------------------------------------

fn pack_polys<H: FieldHarness>(host: &[Vec<H::Host>], n: usize) -> Vec<u32> {
    let batch = host.len();
    let d = H::DEGREE;
    let mut buf = vec![0u32; batch * n * d];
    for (b, row) in host.iter().enumerate() {
        let base = b * n * d;
        for (i, &elem) in row.iter().enumerate() {
            let raw = H::host_to_raw(elem);
            for c in 0..d { buf[base + c * n + i] = raw[c]; }
        }
    }
    buf
}

fn unpack_polys<H: FieldHarness>(buf: &[u32], batch: usize, n: usize) -> Vec<Vec<H::Host>> {
    let d = H::DEGREE;
    let mut out = Vec::with_capacity(batch);
    let mut limbs = vec![0u32; d];
    for b in 0..batch {
        let base = b * n * d;
        let mut row = Vec::with_capacity(n);
        for i in 0..n {
            for c in 0..d { limbs[c] = buf[base + c * n + i]; }
            row.push(H::host_from_raw(&limbs));
        }
        out.push(row);
    }
    out
}

fn pack_zs<H: FieldHarness>(zs: &[H::Host]) -> Vec<u32> {
    let batch = zs.len();
    let d = H::DEGREE;
    let mut buf = vec![0u32; batch * d];
    for (b, &z) in zs.iter().enumerate() {
        let raw = H::host_to_raw(z);
        for c in 0..d { buf[c * batch + b] = raw[c]; }
    }
    buf
}

// ---------------------------------------------------------------------------
// Generic test core: cube-vs-serial-host oracle.
// ---------------------------------------------------------------------------

fn cube_vs_host<H: FieldHarness>(
    device: &Device<Runtime>, log_n: u32, batch: usize, seed: u64, label: &str,
) {
    let n = 1usize << log_n;

    let mut host_polys: Vec<Vec<H::Host>> = Vec::with_capacity(batch);
    for b in 0..batch {
        let mut row = Vec::with_capacity(n);
        for i in 0..n {
            row.push(H::host_from_seed(seed.wrapping_add(0x1000 * b as u64 + i as u64)));
        }
        host_polys.push(row);
    }
    let host_zs: Vec<H::Host> = (0..batch)
        .map(|b| H::host_from_seed(seed.wrapping_add(0xFFFF_0000 + b as u64)))
        .collect();

    let mut expected = host_polys.clone();
    for (row, &z) in expected.iter_mut().zip(host_zs.iter()) {
        div_by_x_minus_z_serial(row.as_mut_slice(), z);
    }

    let exec = PolyDivExec::<H::Cube, Runtime>::new(device, log_n, batch);
    let client = exec.client();

    let buf_bytes = pack_polys::<H>(&host_polys, n);
    let zs_bytes = pack_zs::<H>(&host_zs);
    let buf_h = client.create_from_slice(u32::as_bytes(&buf_bytes));
    let zs_h = client.create_from_slice(u32::as_bytes(&zs_bytes));

    exec.div_by_x_minus_z(&buf_h, &zs_h, log_n, batch);

    let out_u32: Vec<u32> = u32::from_bytes(&client.read_one(buf_h)).to_vec();
    let actual = unpack_polys::<H>(&out_u32, batch, n);

    for b in 0..batch {
        for i in 0..n {
            assert_eq!(H::host_to_raw(actual[b][i]), H::host_to_raw(expected[b][i]),
                "{label}: mismatch at batch={b} pos={i} (log_n={log_n})");
        }
    }
}

// ---------------------------------------------------------------------------
// KB^4 full sweep.
// ---------------------------------------------------------------------------

#[test]
fn divide_koala_bear_4() {
    let device = Device::<Runtime>::acquire();
    let log_wg = device.client().properties().hardware.max_units_per_cube.trailing_zeros();

    let log_ns: Vec<u32> = vec![
        1, 4,
        log_wg.min(10),
        (log_wg + 2).min(12),
        (2 * log_wg).min(20),
        (2 * log_wg + 1).min(21),
    ];
    for &log_n in &log_ns {
        cube_vs_host::<HExt4Kb>(&device, log_n, 2, 0xC0FFEE_DEAD_BEEFu64, "KB^4");
    }
}

#[test]
fn divide_baby_bear_spot() {
    let device = Device::<Runtime>::acquire();
    cube_vs_host::<HBaseBb>(&device, 1, 2, 0xBEEF_CAFE_1234u64, "BB log_n=1");
    cube_vs_host::<HBaseBb>(&device, 12, 2, 0xBEEF_CAFE_1234u64, "BB log_n=12");
}

#[test]
fn divide_baby_bear_5_spot() {
    let device = Device::<Runtime>::acquire();
    cube_vs_host::<HExt5Bb>(&device, 10, 2, 0xDEAD_BEEF_5555u64, "BB^5");
}
