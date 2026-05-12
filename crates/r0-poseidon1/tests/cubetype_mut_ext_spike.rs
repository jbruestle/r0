//! Sub-spike: does `&mut <CubeType-struct>` work when fields are themselves
//! CubeType structs (Ext4)? Original cubetype_mut_spike used u32 fields.
//! ConstraintAccumulator has Ext4 fields and the macro is choking somewhere.

use cubecl::prelude::*;
use r0_cube::{Device, Runtime};
use r0_field::{ext4_add, ext4_from_raws, ext4_mul, Ext4, KoalaBear4Parameters};

#[derive(CubeType, Copy, Clone)]
pub struct ExtAccum {
    pub a: Ext4<KoalaBear4Parameters>,
    pub b: Ext4<KoalaBear4Parameters>,
}

/// Value-in / value-out variant — does cubecl let us return a CubeType struct?
#[cube]
fn map_ext_accum(s: ExtAccum, multiplier: Ext4<KoalaBear4Parameters>) -> ExtAccum {
    ExtAccum {
        a: ext4_add::<KoalaBear4Parameters>(s.a, multiplier),
        b: ext4_mul::<KoalaBear4Parameters>(s.b, multiplier),
    }
}

/// Multiple chained calls via shadowing (no `let mut`).
#[cube]
fn map_ext_accum_chain_shadow(s: ExtAccum, m: Ext4<KoalaBear4Parameters>) -> ExtAccum {
    let s = map_ext_accum(s, m);
    let s = map_ext_accum(s, m);
    let s = map_ext_accum(s, m);
    s
}

/// Comptime-recursive helper — the cleanest way to "loop" while threading
/// a CubeType state when reassignment doesn't work. cubecl resolves the
/// recursion at IR build time as long as the comptime condition reduces.
#[cube]
fn map_ext_accum_recursive(
    s: ExtAccum,
    m: Ext4<KoalaBear4Parameters>,
    #[comptime] i: u32,
    #[comptime] count: u32,
) -> ExtAccum {
    if comptime!(i >= count) {
        s
    } else {
        let s = map_ext_accum(s, m);
        map_ext_accum_recursive(s, m, comptime!(i + 1u32), count)
    }
}

#[cube(launch_unchecked)]
fn spike_kernel(out: &mut Array<u32>) {
    if ABSOLUTE_POS == 0usize {
        let zero = ext4_from_raws::<KoalaBear4Parameters>(0u32, 0u32, 0u32, 0u32);
        let one = ext4_from_raws::<KoalaBear4Parameters>(1u32, 0u32, 0u32, 0u32);
        let s = ExtAccum { a: zero, b: one };
        let mult = ext4_from_raws::<KoalaBear4Parameters>(7u32, 11u32, 13u32, 17u32);
        let s = map_ext_accum(s, mult);
        out[0usize] = s.a.c0;
        out[1usize] = s.a.c1;
        out[2usize] = s.a.c2;
        out[3usize] = s.a.c3;
        out[4usize] = s.b.c0;
        out[5usize] = s.b.c1;
        out[6usize] = s.b.c2;
        out[7usize] = s.b.c3;
    }
}

#[test]
fn ext_accum_mut_compiles_and_runs() {
    let device = Device::<Runtime>::acquire();
    let client = device.client();

    let out_h = client.empty(8 * core::mem::size_of::<u32>());

    unsafe {
        spike_kernel::launch_unchecked::<Runtime>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts::<u32>(&out_h, 8, 1),
        )
        .expect("kernel launch");
    }

    let bytes = client.read_one(out_h);
    let _actual: Vec<u32> = u32::from_bytes(&bytes).to_vec();
    // Just ensure the macro accepted the pattern; correctness of values
    // isn't the point of this spike.
}
