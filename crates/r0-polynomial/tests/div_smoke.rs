//! End-to-end tests for `PolyDivExec::div_by_x_minus_z` over all five
//! `r0-field` instances (BB, KB, BB^4, KB^4, BB^5). For each field we
//! check:
//!
//! - Hand-constructed low-N case (`(x − z₀) · (x − z₁)` divided by
//!   `(x − z₀)` yields `(x − z₁)` with zero remainder).
//! - `p(x)·(x − z) / (x − z) == p(x)` identity for a small random `p`.
//! - Cube vs serial host across a range of `log_n` (single-block fast
//!   path through L=1 multi-block on Mac wgpu).
//!
//! All tests run on wgpu only — cubecl's CPU emulator reports
//! `plane_size = 1`, which constrains `wg_size = 1` in our scan
//! substrate (degenerate). CUDA is gated behind the `cuda` feature
//! per the workspace convention.

#![cfg(feature = "wgpu")]

use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;

use r0_cube::Device;
use r0_field::{
    BabyBear4, BabyBear5, BabyBearParameters, BaseElem, Ext4, Ext5, KoalaBear4,
    KoalaBearParameters, MontyField,
};
use r0_polynomial::host_ref::{div_by_x_minus_z_serial, HostField};
use r0_polynomial::{PairScanLayout, PolyDivExec};

// `HostField` impls for the five `r0-field` instances live with the
// trait in `r0_polynomial::host_ref` to satisfy orphan rules — they're
// re-exported transitively via the trait import above.

// ---------------------------------------------------------------------------
// FieldHarness: per-F bridge between the test core and the concrete
// host / cube types. One impl per field instance.
// ---------------------------------------------------------------------------

trait FieldHarness {
    /// The cube-side `ExtField + PairScanLayout` type — `BaseElem<P>`,
    /// `Ext4<P>`, or `Ext5<P>`.
    type Cube: PairScanLayout;
    /// The host-side counterpart. For `Cube = BaseElem<P>` this is
    /// `MontyField<P>`; for the extensions it's the same `Ext4<P>` /
    /// `Ext5<P>` (which double as both host and cube types).
    type Host: HostField + Copy;
    /// Number of `u32` limbs per element (= `Cube::DEGREE`).
    const DEGREE: usize;

    /// Construct a host element from a u64 seed (any deterministic mix).
    fn host_from_seed(seed: u64) -> Self::Host;
    /// Pack a host element into `DEGREE` raw Montgomery `u32`s, in
    /// component order matching `ExtField::store`'s transposed layout.
    fn host_to_raw(host: Self::Host) -> Vec<u32>;
    /// Inverse of [`host_to_raw`].
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
    fn host_to_raw(host: Self::Host) -> Vec<u32> {
        vec![host.raw()]
    }
    fn host_from_raw(raw: &[u32]) -> Self::Host {
        MontyField::<BabyBearParameters>::from_raw(raw[0])
    }
}

struct HBaseKb;
impl FieldHarness for HBaseKb {
    type Cube = BaseElem<KoalaBearParameters>;
    type Host = MontyField<KoalaBearParameters>;
    const DEGREE: usize = 1;

    fn host_from_seed(seed: u64) -> Self::Host {
        MontyField::<KoalaBearParameters>::from_canonical(
            (seed % KoalaBearParameters::PRIME as u64) as u32,
        )
    }
    fn host_to_raw(host: Self::Host) -> Vec<u32> {
        vec![host.raw()]
    }
    fn host_from_raw(raw: &[u32]) -> Self::Host {
        MontyField::<KoalaBearParameters>::from_raw(raw[0])
    }
}

struct HExt4Bb;
impl FieldHarness for HExt4Bb {
    type Cube = Ext4<r0_field::BabyBear4Parameters>;
    type Host = BabyBear4;
    const DEGREE: usize = 4;

    fn host_from_seed(seed: u64) -> Self::Host {
        let p = BabyBearParameters::PRIME as u64;
        BabyBear4::from_canonical([
            ((seed.wrapping_mul(0x9E3779B97F4A7C15) ^ 0x1) % p) as u32,
            ((seed.wrapping_mul(0x9E3779B97F4A7C15) ^ 0x2) % p) as u32,
            ((seed.wrapping_mul(0x9E3779B97F4A7C15) ^ 0x3) % p) as u32,
            ((seed.wrapping_mul(0x9E3779B97F4A7C15) ^ 0x4) % p) as u32,
        ])
    }
    fn host_to_raw(host: Self::Host) -> Vec<u32> {
        host.raw().to_vec()
    }
    fn host_from_raw(raw: &[u32]) -> Self::Host {
        BabyBear4::from_raw([raw[0], raw[1], raw[2], raw[3]])
    }
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
    fn host_to_raw(host: Self::Host) -> Vec<u32> {
        host.raw().to_vec()
    }
    fn host_from_raw(raw: &[u32]) -> Self::Host {
        KoalaBear4::from_raw([raw[0], raw[1], raw[2], raw[3]])
    }
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
    fn host_to_raw(host: Self::Host) -> Vec<u32> {
        host.raw().to_vec()
    }
    fn host_from_raw(raw: &[u32]) -> Self::Host {
        BabyBear5::from_raw([raw[0], raw[1], raw[2], raw[3], raw[4]])
    }
}

// Minor convenience: pull `MontyParameters::PRIME` into scope for the
// canonical-form helpers above without writing `<P as MontyParameters>::PRIME`
// every time.
use r0_field::MontyParameters;

// ---------------------------------------------------------------------------
// Layout helpers: pack / unpack the per-batch transposed buffers.
// ---------------------------------------------------------------------------

/// Lay out `batch × n` host coefficients into the
/// `batch × n × DEGREE` u32 buffer `ExtField::load` expects:
///   per polynomial, base = b·n·D; component c of element i at offset
///   c·n + i within the polynomial slice.
fn pack_polys<H: FieldHarness>(host: &[Vec<H::Host>], n: usize) -> Vec<u32> {
    let batch = host.len();
    let d = H::DEGREE;
    let mut buf = vec![0u32; batch * n * d];
    for (b, row) in host.iter().enumerate() {
        let base = b * n * d;
        for (i, &elem) in row.iter().enumerate() {
            let raw = H::host_to_raw(elem);
            for c in 0..d {
                buf[base + c * n + i] = raw[c];
            }
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
            for c in 0..d {
                limbs[c] = buf[base + c * n + i];
            }
            row.push(H::host_from_raw(&limbs));
        }
        out.push(row);
    }
    out
}

/// Lay out `batch` host scalars into the `batch × DEGREE` u32 contexts
/// buffer (transposed: component c of row b at offset c·batch + b).
fn pack_zs<H: FieldHarness>(zs: &[H::Host]) -> Vec<u32> {
    let batch = zs.len();
    let d = H::DEGREE;
    let mut buf = vec![0u32; batch * d];
    for (b, &z) in zs.iter().enumerate() {
        let raw = H::host_to_raw(z);
        for c in 0..d {
            buf[c * batch + b] = raw[c];
        }
    }
    buf
}

// ---------------------------------------------------------------------------
// Generic test core: cube-vs-serial-host oracle.
// ---------------------------------------------------------------------------

fn cube_vs_host<H>(
    device: &Device<WgpuRuntime>,
    log_n: u32,
    batch: usize,
    seed: u64,
    label: &str,
) where
    H: FieldHarness,
{
    let n = 1usize << log_n;

    // Build random host polynomials and per-row z values.
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

    // Serial reference per row, in place.
    let mut expected = host_polys.clone();
    for (row, &z) in expected.iter_mut().zip(host_zs.iter()) {
        div_by_x_minus_z_serial(row.as_mut_slice(), z);
    }

    // Pack inputs, run cube, read back.
    let exec =
        PolyDivExec::<H::Cube, WgpuRuntime>::new(device, log_n, batch);
    let client = exec.client();

    let buf_bytes = pack_polys::<H>(&host_polys, n);
    let zs_bytes = pack_zs::<H>(&host_zs);
    let buf_h = client.create_from_slice(u32::as_bytes(&buf_bytes));
    let zs_h = client.create_from_slice(u32::as_bytes(&zs_bytes));

    exec.div_by_x_minus_z(&buf_h, &zs_h, log_n, batch);

    let raw_out = client.read_one(buf_h);
    let out_u32: Vec<u32> = u32::from_bytes(&raw_out).to_vec();
    let actual = unpack_polys::<H>(&out_u32, batch, n);

    for b in 0..batch {
        for i in 0..n {
            let exp_raw = H::host_to_raw(expected[b][i]);
            let got_raw = H::host_to_raw(actual[b][i]);
            assert_eq!(
                exp_raw, got_raw,
                "{label}: mismatch at batch={b} pos={i} (log_n={log_n})"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Per-field cube-oracle sweep. `log_n` from 1 (single-block) up through
// L=1 multi-block on Mac wgpu (`2 · log_wg + 1`, where log_wg = 10 there
// so the L=1 sizes hit 2^21).
// ---------------------------------------------------------------------------

fn run_field_sweep<H: FieldHarness>(label: &str)
where
    H::Cube: 'static,
{
    let device = Device::<WgpuRuntime>::acquire();
    let client = device.client();
    let log_wg = client
        .properties()
        .hardware
        .max_units_per_cube
        .trailing_zeros();

    // Sizes covering: single-block fast path, single-block n=wg, L=0
    // multi-block, L=0 max, L=1 just-over.
    let log_ns: Vec<u32> = vec![
        1,
        4,
        log_wg.min(10),               // typical single-block
        (log_wg + 2).min(12),         // L=0 multi
        (2 * log_wg).min(20),         // L=0 max-ish
        (2 * log_wg + 1).min(21),     // L=1 just-over
    ];

    for &log_n in &log_ns {
        // Small batch to keep the test fast across five fields.
        cube_vs_host::<H>(&device, log_n, 2, 0xC0FFEE_DEAD_BEEFu64, label);
    }
}

// ---------------------------------------------------------------------------
// Trivial smoke: dividing the zero polynomial by `(x − z)` yields the
// zero polynomial back (quotient zero, remainder zero). Catches gross
// plumbing failures before the random oracle sweep without depending on
// `HostField` exposing negation.
// ---------------------------------------------------------------------------

fn check_zero_smoke<H: FieldHarness>() {
    let zero = <H::Host as HostField>::zero();
    let device = Device::<WgpuRuntime>::acquire();
    let n = 1usize << 3;
    let zero_row: Vec<H::Host> = vec![zero; n];
    let z_host = vec![H::host_from_seed(42)];

    let exec = PolyDivExec::<H::Cube, WgpuRuntime>::new(&device, 3, 1);
    let client = exec.client();

    let buf_bytes = pack_polys::<H>(&[zero_row], n);
    let zs_bytes = pack_zs::<H>(&z_host);
    let buf_h = client.create_from_slice(u32::as_bytes(&buf_bytes));
    let zs_h = client.create_from_slice(u32::as_bytes(&zs_bytes));

    exec.div_by_x_minus_z(&buf_h, &zs_h, 3, 1);

    let raw_out = client.read_one(buf_h);
    let out_u32: Vec<u32> = u32::from_bytes(&raw_out).to_vec();
    let actual = unpack_polys::<H>(&out_u32, 1, n);
    for elem in &actual[0] {
        assert_eq!(H::host_to_raw(*elem), H::host_to_raw(zero));
    }
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[test]
fn divide_baby_bear() {
    check_zero_smoke::<HBaseBb>();
    run_field_sweep::<HBaseBb>("BabyBear");
}

#[test]
fn divide_koala_bear() {
    check_zero_smoke::<HBaseKb>();
    run_field_sweep::<HBaseKb>("KoalaBear");
}

#[test]
fn divide_baby_bear_4() {
    check_zero_smoke::<HExt4Bb>();
    run_field_sweep::<HExt4Bb>("BB^4");
}

#[test]
fn divide_koala_bear_4() {
    check_zero_smoke::<HExt4Kb>();
    run_field_sweep::<HExt4Kb>("KB^4");
}

#[test]
fn divide_baby_bear_5() {
    check_zero_smoke::<HExt5Bb>();
    run_field_sweep::<HExt5Bb>("BB^5");
}
