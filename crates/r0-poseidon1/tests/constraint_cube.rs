//! Cube `poseidon1_kb16_constraint` end-to-end. Three checks:
//!   1. **Valid witness → acc = 0**: Witness produced by witgen path
//!      should satisfy all 148 S-box constraints; accumulator should be
//!      exactly zero in KB^4 regardless of `alpha`.
//!   2. **Flipped bit → acc ≠ 0**: Mutating one limb of one witness column
//!      should yield a non-zero accumulator.
//!   3. **Cube ≡ host shadow**: For an arbitrary witness, the cube path
//!      and the host shadow (`host_constraint_kb_witness`) must agree on
//!      both `acc` and `alpha_pow`.

use cubecl::prelude::*;

use r0_cube::{Device, Runtime};
use r0_field::{ext4_from_raws, Ext4, KoalaBear, KoalaBear4Parameters};
use r0_poseidon1::{
    host_constraint_kb_witness, host_permute_with_trace, poseidon1_kb16_constraint,
    ConstraintAccumulator, N_WITNESS_SBOXES,
};

const N_THREADS: u32 = 4;

#[cube(launch_unchecked)]
fn constraint_kernel(
    input_states: &Array<u32>,
    witness: &Array<u32>,
    output: &mut Array<u32>,
    alpha_c0: u32,
    alpha_c1: u32,
    alpha_c2: u32,
    alpha_c3: u32,
    alpha_pow_c0: u32,
    alpha_pow_c1: u32,
    alpha_pow_c2: u32,
    alpha_pow_c3: u32,
    stride: u32,
    #[comptime] n_threads: u32,
) {
    let tid = ABSOLUTE_POS;
    if tid < comptime!(n_threads as usize) {
        let mut input_state = Array::<u32>::new(comptime!(16usize));
        #[unroll]
        for i in 0u32..16u32 {
            input_state[comptime!(i as usize)] =
                input_states[tid * 16usize + comptime!(i as usize)];
        }

        let alpha = ext4_from_raws::<KoalaBear4Parameters>(
            alpha_c0, alpha_c1, alpha_c2, alpha_c3,
        );
        let alpha_pow = ext4_from_raws::<KoalaBear4Parameters>(
            alpha_pow_c0, alpha_pow_c1, alpha_pow_c2, alpha_pow_c3,
        );
        let zero = ext4_from_raws::<KoalaBear4Parameters>(0u32, 0u32, 0u32, 0u32);

        let cs = ConstraintAccumulator {
            alpha: alpha,
            acc: zero,
            alpha_pow: alpha_pow,
        };

        let cs = poseidon1_kb16_constraint(
            &input_state,
            witness,
            0u32,
            tid as u32,
            stride,
            cs,
        );

        // Output layout per thread: 8 u32s — acc (4) then alpha_pow (4).
        output[tid * 8usize + 0usize] = cs.acc.c0;
        output[tid * 8usize + 1usize] = cs.acc.c1;
        output[tid * 8usize + 2usize] = cs.acc.c2;
        output[tid * 8usize + 3usize] = cs.acc.c3;
        output[tid * 8usize + 4usize] = cs.alpha_pow.c0;
        output[tid * 8usize + 5usize] = cs.alpha_pow.c1;
        output[tid * 8usize + 6usize] = cs.alpha_pow.c2;
        output[tid * 8usize + 7usize] = cs.alpha_pow.c3;
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build N_THREADS independent (input, witness) pairs by running the host
/// serial trace generator on deterministic seeds.
fn build_inputs_and_witnesses() -> (
    Vec<[KoalaBear; 16]>,
    Vec<[KoalaBear; N_WITNESS_SBOXES]>,
) {
    let mut ins = Vec::with_capacity(N_THREADS as usize);
    let mut wits = Vec::with_capacity(N_THREADS as usize);
    for t in 0..N_THREADS as u32 {
        let s: [KoalaBear; 16] = core::array::from_fn(|i| {
            let v = t.wrapping_mul(0x9E37_79B1)
                .wrapping_add((i as u32).wrapping_mul(0xC2B2_AE3D));
            KoalaBear::from_canonical(v)
        });
        let mut state = s;
        let trace = host_permute_with_trace(&mut state);
        ins.push(s);
        wits.push(trace);
    }
    (ins, wits)
}

/// Pack `N_THREADS` length-16 inputs into a contiguous-per-thread buffer.
fn pack_inputs(ins: &[[KoalaBear; 16]]) -> Vec<u32> {
    let mut buf = Vec::with_capacity(ins.len() * 16);
    for state in ins {
        for k in 0..16 {
            buf.push(state[k].raw());
        }
    }
    buf
}

/// Pack `N_THREADS` length-148 witnesses into the transposed layout the
/// cube kernel expects: column `c` of row `t` at `c * stride + t`.
fn pack_witnesses(wits: &[[KoalaBear; N_WITNESS_SBOXES]], stride: u32) -> Vec<u32> {
    let stride = stride as usize;
    let mut buf = vec![0u32; N_WITNESS_SBOXES * stride];
    for (t, trace) in wits.iter().enumerate() {
        for c in 0..N_WITNESS_SBOXES {
            buf[c * stride + t] = trace[c].raw();
        }
    }
    buf
}

fn run_cube(
    client: &cubecl::prelude::ComputeClient<Runtime>,
    inputs: &[u32],
    witness: &[u32],
    stride: u32,
    alpha: [u32; 4],
    alpha_pow_init: [u32; 4],
) -> Vec<u32> {
    let n = N_THREADS as usize;

    let in_h = client.create_from_slice(u32::as_bytes(inputs));
    let wit_h = client.create_from_slice(u32::as_bytes(witness));
    let out_h = client.empty(n * 8 * core::mem::size_of::<u32>());

    unsafe {
        constraint_kernel::launch_unchecked::<Runtime>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(N_THREADS),
            ArrayArg::from_raw_parts::<u32>(&in_h, inputs.len(), 1),
            ArrayArg::from_raw_parts::<u32>(&wit_h, witness.len(), 1),
            ArrayArg::from_raw_parts::<u32>(&out_h, n * 8, 1),
            ScalarArg::new(alpha[0]),
            ScalarArg::new(alpha[1]),
            ScalarArg::new(alpha[2]),
            ScalarArg::new(alpha[3]),
            ScalarArg::new(alpha_pow_init[0]),
            ScalarArg::new(alpha_pow_init[1]),
            ScalarArg::new(alpha_pow_init[2]),
            ScalarArg::new(alpha_pow_init[3]),
            ScalarArg::new(stride),
            N_THREADS,
        )
        .expect("constraint_kernel launch failed");
    }

    u32::from_bytes(&client.read_one(out_h)).to_vec()
}

/// Random KB^4 alpha (canonical limbs).
fn alpha_canonical() -> [u32; 4] {
    [0x12345678, 0x9ABCDEF0, 0x0FEDCBA9, 0x87654321]
}

/// Convert canonical limbs to raw Montgomery limbs for kernel input.
fn ext4_canon_to_raw(canon: [u32; 4]) -> [u32; 4] {
    let f = Ext4::<KoalaBear4Parameters>::from_canonical(canon);
    f.raw()
}

fn ext4_one_raw() -> [u32; 4] {
    Ext4::<KoalaBear4Parameters>::from_canonical([1, 0, 0, 0]).raw()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn valid_witness_yields_zero_accumulator() {
    let device = Device::<Runtime>::acquire();
    let client = device.client();

    let (ins, wits) = build_inputs_and_witnesses();
    let inputs_buf = pack_inputs(&ins);
    let witness_buf = pack_witnesses(&wits, N_THREADS);

    let alpha = ext4_canon_to_raw(alpha_canonical());
    let alpha_pow_init = ext4_one_raw();

    let out = run_cube(client, &inputs_buf, &witness_buf, N_THREADS, alpha, alpha_pow_init);

    for t in 0..N_THREADS as usize {
        let acc_raw = [
            out[t * 8 + 0],
            out[t * 8 + 1],
            out[t * 8 + 2],
            out[t * 8 + 3],
        ];
        let acc = Ext4::<KoalaBear4Parameters>::from_raw(acc_raw);
        let canon = acc.to_canonical();
        assert_eq!(
            canon, [0, 0, 0, 0],
            "thread {t}: valid witness produced non-zero acc {canon:?}"
        );
    }
}

#[test]
fn flipped_witness_yields_nonzero_accumulator() {
    let device = Device::<Runtime>::acquire();
    let client = device.client();

    let (ins, mut wits) = build_inputs_and_witnesses();
    // Flip column 42 (a partial-round sbox value) for thread 0.
    {
        let raw = wits[0][42].raw();
        let flipped = raw ^ 1; // Toggle low bit; still < p.
        wits[0][42] = KoalaBear::from_raw(flipped);
    }
    let inputs_buf = pack_inputs(&ins);
    let witness_buf = pack_witnesses(&wits, N_THREADS);

    let alpha = ext4_canon_to_raw(alpha_canonical());
    let alpha_pow_init = ext4_one_raw();

    let out = run_cube(client, &inputs_buf, &witness_buf, N_THREADS, alpha, alpha_pow_init);

    // Thread 0 (flipped) should have non-zero acc.
    let acc0 = Ext4::<KoalaBear4Parameters>::from_raw([out[0], out[1], out[2], out[3]]);
    assert_ne!(
        acc0.to_canonical(),
        [0, 0, 0, 0],
        "thread 0 (flipped witness) produced zero acc — flip not detected"
    );

    // Other threads should still be zero.
    for t in 1..N_THREADS as usize {
        let acc = Ext4::<KoalaBear4Parameters>::from_raw([
            out[t * 8 + 0],
            out[t * 8 + 1],
            out[t * 8 + 2],
            out[t * 8 + 3],
        ]);
        assert_eq!(
            acc.to_canonical(),
            [0, 0, 0, 0],
            "thread {t} (unmodified witness) should have zero acc"
        );
    }
}

#[test]
fn cube_matches_host_shadow() {
    let device = Device::<Runtime>::acquire();
    let client = device.client();

    let (ins, mut wits) = build_inputs_and_witnesses();
    // Corrupt witnesses partially so we get non-trivial accumulator values.
    for t in 0..N_THREADS as usize {
        let c = 16 + (t * 7) % (N_WITNESS_SBOXES - 16);
        let raw = wits[t][c].raw();
        wits[t][c] = KoalaBear::from_raw(raw ^ ((t as u32 + 1) * 3));
    }

    let inputs_buf = pack_inputs(&ins);
    let witness_buf = pack_witnesses(&wits, N_THREADS);

    let alpha_canon = alpha_canonical();
    let alpha = ext4_canon_to_raw(alpha_canon);
    let alpha_pow_init_canon = [0xCAFE_BABE, 0xDEAD_BEEF, 0x1234_5678, 0xABCD_EF01];
    let alpha_pow_init = ext4_canon_to_raw(alpha_pow_init_canon);

    let out = run_cube(client, &inputs_buf, &witness_buf, N_THREADS, alpha, alpha_pow_init);

    let alpha_ext = Ext4::<KoalaBear4Parameters>::from_canonical(alpha_canon);
    let alpha_pow_ext = Ext4::<KoalaBear4Parameters>::from_canonical(alpha_pow_init_canon);
    let zero_ext = Ext4::<KoalaBear4Parameters>::ZERO;

    for t in 0..N_THREADS as usize {
        let cs_init = ConstraintAccumulator {
            alpha: alpha_ext,
            acc: zero_ext,
            alpha_pow: alpha_pow_ext,
        };
        let cs_host = host_constraint_kb_witness(&ins[t], &wits[t], cs_init);

        let acc_cube = Ext4::<KoalaBear4Parameters>::from_raw([
            out[t * 8 + 0],
            out[t * 8 + 1],
            out[t * 8 + 2],
            out[t * 8 + 3],
        ]);
        let alpha_pow_cube = Ext4::<KoalaBear4Parameters>::from_raw([
            out[t * 8 + 4],
            out[t * 8 + 5],
            out[t * 8 + 6],
            out[t * 8 + 7],
        ]);

        assert_eq!(
            acc_cube.to_canonical(),
            cs_host.acc.to_canonical(),
            "thread {t}: acc cube != host"
        );
        assert_eq!(
            alpha_pow_cube.to_canonical(),
            cs_host.alpha_pow.to_canonical(),
            "thread {t}: alpha_pow cube != host"
        );
    }
}
