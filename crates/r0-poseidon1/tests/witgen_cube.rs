//! Cube `poseidon1_kb16_permute_with_witness` end-to-end. For a batch of
//! independent inputs, verifies that:
//!   1. The 16 output values per row match `host_permute`.
//!   2. The 148 written S-box values per row match `host_permute_with_trace`.

use cubecl::prelude::*;

use r0_cube::{Device, Runtime};
use r0_field::KoalaBear;
use r0_poseidon1::{
    host_permute_with_trace, poseidon1_kb16_permute_with_witness, N_WITNESS_SBOXES,
};

const N_THREADS: u32 = 16;

/// Each thread reads its 16 inputs from `input` (contiguous-per-thread,
/// `[t*16, t*16 + 16)`), runs the witgen permutation, writes 16 outputs
/// to `output` (same layout), and writes 148 S-box values to `witness`
/// in transposed layout: column `c` at `(witness_col_base + c) * stride + t`.
#[cube(launch_unchecked)]
fn perm_witness_kernel(
    input: &Array<u32>,
    output: &mut Array<u32>,
    witness: &mut Array<u32>,
    witness_col_base: u32,
    stride: u32,
    #[comptime] n_threads: u32,
) {
    let tid = ABSOLUTE_POS;
    if tid < comptime!(n_threads as usize) {
        let mut state = Array::<u32>::new(comptime!(16usize));
        #[unroll]
        for i in 0u32..16u32 {
            state[comptime!(i as usize)] = input[tid * 16usize + comptime!(i as usize)];
        }

        poseidon1_kb16_permute_with_witness(
            &mut state,
            witness,
            witness_col_base,
            tid as u32,
            stride,
        );

        #[unroll]
        for i in 0u32..16u32 {
            output[tid * 16usize + comptime!(i as usize)] = state[comptime!(i as usize)];
        }
    }
}

#[test]
fn witgen_outputs_and_sboxes_match_host() {
    let device = Device::<Runtime>::acquire();
    let client = device.client();

    // Build N_THREADS independent inputs.
    let n = N_THREADS as usize;
    let mut in_raws: Vec<u32> = Vec::with_capacity(n * 16);
    let mut in_states: Vec<[KoalaBear; 16]> = Vec::with_capacity(n);
    for t in 0..n as u32 {
        let s: [KoalaBear; 16] = core::array::from_fn(|i| {
            let v = t.wrapping_mul(0x9E37_79B1).wrapping_add((i as u32).wrapping_mul(0xC2B2_AE3D));
            KoalaBear::from_canonical(v)
        });
        for k in 0..16 {
            in_raws.push(s[k].raw());
        }
        in_states.push(s);
    }

    // Witness layout: stride = N_THREADS, witness_col_base = 0.
    // 148 columns × N_THREADS rows.
    let stride = N_THREADS;
    let witness_col_base = 0u32;
    let witness_len = N_WITNESS_SBOXES * stride as usize;

    let in_h = client.create_from_slice(u32::as_bytes(&in_raws));
    let out_h = client.empty(in_raws.len() * core::mem::size_of::<u32>());
    let wit_h = client.empty(witness_len * core::mem::size_of::<u32>());

    unsafe {
        perm_witness_kernel::launch_unchecked::<Runtime>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(N_THREADS),
            ArrayArg::from_raw_parts::<u32>(&in_h, in_raws.len(), 1),
            ArrayArg::from_raw_parts::<u32>(&out_h, in_raws.len(), 1),
            ArrayArg::from_raw_parts::<u32>(&wit_h, witness_len, 1),
            ScalarArg::new(witness_col_base),
            ScalarArg::new(stride),
            N_THREADS,
        )
        .expect("perm_witness_kernel launch failed");
    }

    let out_raws: Vec<u32> = u32::from_bytes(&client.read_one(out_h)).to_vec();
    let wit_raws: Vec<u32> = u32::from_bytes(&client.read_one(wit_h)).to_vec();

    for t in 0..n {
        let mut expected_state = in_states[t];
        let expected_trace = host_permute_with_trace(&mut expected_state);

        // Compare 16 outputs.
        for i in 0..16 {
            let actual = KoalaBear::from_raw(out_raws[t * 16 + i]).to_canonical();
            let expected = expected_state[i].to_canonical();
            assert_eq!(actual, expected, "thread {t}: output slot {i} mismatch");
        }

        // Compare 148 S-box values.
        for c in 0..N_WITNESS_SBOXES {
            let addr = c * stride as usize + t;
            let actual = KoalaBear::from_raw(wit_raws[addr]).to_canonical();
            let expected = expected_trace[c].to_canonical();
            assert_eq!(actual, expected, "thread {t}: witness col {c} mismatch");
        }
    }
}
