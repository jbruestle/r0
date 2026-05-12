//! Per-row Poseidon1 constraint contribution: cube subroutine + host
//! verifier.
//!
//! Same 28-round structure as the permutation, but instead of computing
//! S-box outputs from scratch the engine reads the witness for each
//! S-box, accumulates `α^k · (predicted - witness)` into `acc`, and
//! continues the rolling state from the witness value (so a valid
//! witness produces zero contribution while an invalid witness yields a
//! non-zero accumulator).
//!
//! For chaining into a larger constraint kernel, [`ConstraintAccumulator`]
//! threads `(alpha, acc, alpha_pow)` through; the caller seeds `alpha_pow`
//! and reads the advanced value back out from the returned struct for
//! the next subroutine in the chain.
//!
//! # Deviations from DESIGN.md (cubecl 0.9 limits)
//!
//! 1. **Value-in / value-out instead of `&mut ConstraintAccumulator`.**
//!    cubecl 0.9 supports field assignment on `&mut <CubeType>` only when
//!    the field type is itself a `CubePrimitive` (u32 etc.). For
//!    CubeType-typed fields like `Ext4` it errors with a confusing
//!    `From<…Expand>` trait bound. The
//!    `cubetype_mut_ext_spike` test in this crate documents the failure;
//!    the workaround used here is to take `cstate` by value and return
//!    the updated struct, with caller chain `cstate = poseidon1_kb16_constraint(…, cstate)`.
//!
//! 2. **Comptime-recursive helpers replace `for` loops over a CubeType
//!    accumulator.** `let mut cs = …; cs = helper(cs)` reassignment of a
//!    CubeType-with-CubeType-fields struct hits the same macro
//!    limitation. The clean replacement is a comptime-recursive
//!    `#[cube] fn` (`if comptime!(i >= N) { acc } else { recurse with i+1 }`)
//!    — cubecl resolves the recursion at IR build time, generating an
//!    inlined chain of N calls.

use cubecl::prelude::*;

use r0_field::{
    ext4_add, ext4_from_raws, ext4_mul, monty_add, monty_mul, monty_sub, Ext4, KoalaBear,
    KoalaBear4Parameters, KoalaBearParameters, MontyField,
};

use crate::host_ref::{
    mds_naive, round_constants_lifted, N_INITIAL_FULL, N_PARTIAL, N_ROUNDS, N_WITNESS_SBOXES,
};
use crate::mds::{lambda_over_16_lifted, ws_lifted};
use crate::partial::{tables, HISTORY_LEN};

// ---------------------------------------------------------------------------
// Host-side const generators (called at IR build time via comptime!()).
// ---------------------------------------------------------------------------

fn rc_montgomery_flat() -> Vec<u32> {
    let rc = round_constants_lifted();
    let mut out = Vec::with_capacity(N_ROUNDS * 16);
    for r in 0..N_ROUNDS {
        for i in 0..16 {
            out.push(rc[r][i].raw());
        }
    }
    out
}

fn lambda_montgomery() -> Vec<u32> {
    lambda_over_16_lifted().iter().map(|x| x.raw()).collect()
}

fn ws_montgomery() -> Vec<u32> {
    ws_lifted().iter().map(|x| x.raw()).collect()
}

fn partial_weights_flat_montgomery() -> Vec<u32> {
    let t = tables();
    let mut out = Vec::with_capacity(N_PARTIAL * HISTORY_LEN);
    for r in 0..N_PARTIAL {
        for k in 0..HISTORY_LEN {
            out.push(t.partial_weights[r][k].raw());
        }
    }
    out
}

fn partial_offsets_montgomery() -> Vec<u32> {
    tables().partial_offsets.iter().map(|x| x.raw()).collect()
}

fn terminal_weights_flat_montgomery() -> Vec<u32> {
    let t = tables();
    let mut out = Vec::with_capacity(16 * HISTORY_LEN);
    for i in 0..16 {
        for k in 0..HISTORY_LEN {
            out.push(t.terminal_weights[i][k].raw());
        }
    }
    out
}

fn terminal_offsets_montgomery() -> Vec<u32> {
    tables().terminal_offsets.iter().map(|x| x.raw()).collect()
}

// ---------------------------------------------------------------------------
// ConstraintAccumulator
// ---------------------------------------------------------------------------

/// Per-row constraint state, threaded through a chain of constraint
/// subroutines. `alpha` is the read-only mixing parameter. `acc` is the
/// running `Σ_i α^i · diff_i`. `alpha_pow` is the current α-power; the
/// subroutine reads it on entry and the returned struct holds the
/// advanced value (= input · α^148 for one full Poseidon constraint call).
#[derive(CubeType, Copy, Clone)]
pub struct ConstraintAccumulator {
    pub alpha: Ext4<KoalaBear4Parameters>,
    pub acc: Ext4<KoalaBear4Parameters>,
    pub alpha_pow: Ext4<KoalaBear4Parameters>,
}

// ---------------------------------------------------------------------------
// Butterfly + MDS helpers (mirror of permute.rs's; private here).
// ---------------------------------------------------------------------------

#[cube]
fn bt(v: &mut Array<u32>, #[comptime] lo: u32, #[comptime] hi: u32) {
    let a = v[comptime!(lo as usize)];
    let b = v[comptime!(hi as usize)];
    v[comptime!(lo as usize)] = monty_add::<KoalaBearParameters>(a, b);
    v[comptime!(hi as usize)] = monty_sub::<KoalaBearParameters>(a, b);
}

#[cube]
fn dit(v: &mut Array<u32>, #[comptime] lo: u32, #[comptime] hi: u32, t: u32) {
    let a = v[comptime!(lo as usize)];
    let b = v[comptime!(hi as usize)];
    let tb = monty_mul::<KoalaBearParameters>(b, t);
    v[comptime!(lo as usize)] = monty_add::<KoalaBearParameters>(a, tb);
    v[comptime!(hi as usize)] = monty_sub::<KoalaBearParameters>(a, tb);
}

#[cube]
fn neg_dif(v: &mut Array<u32>, #[comptime] lo: u32, #[comptime] hi: u32, t: u32) {
    let a = v[comptime!(lo as usize)];
    let b = v[comptime!(hi as usize)];
    v[comptime!(lo as usize)] = monty_add::<KoalaBearParameters>(a, b);
    let bma = monty_sub::<KoalaBearParameters>(b, a);
    v[comptime!(hi as usize)] = monty_mul::<KoalaBearParameters>(bma, t);
}

#[cube]
fn mds_fft_16(state: &mut Array<u32>, lambda: &Array<u32>, w: &Array<u32>) {
    bt(state, 0u32, 8u32);
    neg_dif(state, 1u32, 9u32, w[7usize]);
    neg_dif(state, 2u32, 10u32, w[6usize]);
    neg_dif(state, 3u32, 11u32, w[5usize]);
    neg_dif(state, 4u32, 12u32, w[4usize]);
    neg_dif(state, 5u32, 13u32, w[3usize]);
    neg_dif(state, 6u32, 14u32, w[2usize]);
    neg_dif(state, 7u32, 15u32, w[1usize]);
    bt(state, 0u32, 4u32);
    neg_dif(state, 1u32, 5u32, w[6usize]);
    neg_dif(state, 2u32, 6u32, w[4usize]);
    neg_dif(state, 3u32, 7u32, w[2usize]);
    bt(state, 8u32, 12u32);
    neg_dif(state, 9u32, 13u32, w[6usize]);
    neg_dif(state, 10u32, 14u32, w[4usize]);
    neg_dif(state, 11u32, 15u32, w[2usize]);
    bt(state, 0u32, 2u32);
    neg_dif(state, 1u32, 3u32, w[4usize]);
    bt(state, 4u32, 6u32);
    neg_dif(state, 5u32, 7u32, w[4usize]);
    bt(state, 8u32, 10u32);
    neg_dif(state, 9u32, 11u32, w[4usize]);
    bt(state, 12u32, 14u32);
    neg_dif(state, 13u32, 15u32, w[4usize]);
    bt(state, 0u32, 1u32);
    bt(state, 2u32, 3u32);
    bt(state, 4u32, 5u32);
    bt(state, 6u32, 7u32);
    bt(state, 8u32, 9u32);
    bt(state, 10u32, 11u32);
    bt(state, 12u32, 13u32);
    bt(state, 14u32, 15u32);

    #[unroll]
    for i in 0u32..16u32 {
        let s = state[comptime!(i as usize)];
        let l = lambda[comptime!(i as usize)];
        state[comptime!(i as usize)] = monty_mul::<KoalaBearParameters>(s, l);
    }

    bt(state, 0u32, 1u32);
    bt(state, 2u32, 3u32);
    bt(state, 4u32, 5u32);
    bt(state, 6u32, 7u32);
    bt(state, 8u32, 9u32);
    bt(state, 10u32, 11u32);
    bt(state, 12u32, 13u32);
    bt(state, 14u32, 15u32);
    bt(state, 0u32, 2u32);
    dit(state, 1u32, 3u32, w[4usize]);
    bt(state, 4u32, 6u32);
    dit(state, 5u32, 7u32, w[4usize]);
    bt(state, 8u32, 10u32);
    dit(state, 9u32, 11u32, w[4usize]);
    bt(state, 12u32, 14u32);
    dit(state, 13u32, 15u32, w[4usize]);
    bt(state, 0u32, 4u32);
    dit(state, 1u32, 5u32, w[2usize]);
    dit(state, 2u32, 6u32, w[4usize]);
    dit(state, 3u32, 7u32, w[6usize]);
    bt(state, 8u32, 12u32);
    dit(state, 9u32, 13u32, w[2usize]);
    dit(state, 10u32, 14u32, w[4usize]);
    dit(state, 11u32, 15u32, w[6usize]);
    bt(state, 0u32, 8u32);
    dit(state, 1u32, 9u32, w[1usize]);
    dit(state, 2u32, 10u32, w[2usize]);
    dit(state, 3u32, 11u32, w[3usize]);
    dit(state, 4u32, 12u32, w[4usize]);
    dit(state, 5u32, 13u32, w[5usize]);
    dit(state, 6u32, 14u32, w[6usize]);
    dit(state, 7u32, 15u32, w[7usize]);
}

// ---------------------------------------------------------------------------
// Per-S-box constraint check + accumulator advance.
// ---------------------------------------------------------------------------

/// Mix one S-box constraint into `cstate`: predicted = `pre^3` (KB),
/// diff = predicted - witness_val (KB), `acc += alpha_pow · lift(diff)`,
/// `alpha_pow *= alpha`. Returns the updated accumulator. The caller
/// uses `witness_val` separately to advance the rolling state.
#[cube]
fn mix_sbox(pre: u32, witness_val: u32, cstate: ConstraintAccumulator) -> ConstraintAccumulator {
    let pre_sq = monty_mul::<KoalaBearParameters>(pre, pre);
    let predicted = monty_mul::<KoalaBearParameters>(pre_sq, pre);
    let diff_kb = monty_sub::<KoalaBearParameters>(predicted, witness_val);
    let diff_ext4 = ext4_from_raws::<KoalaBear4Parameters>(diff_kb, 0u32, 0u32, 0u32);
    let term = ext4_mul::<KoalaBear4Parameters>(cstate.alpha_pow, diff_ext4);
    ConstraintAccumulator {
        alpha: cstate.alpha,
        acc: ext4_add::<KoalaBear4Parameters>(cstate.acc, term),
        alpha_pow: ext4_mul::<KoalaBear4Parameters>(cstate.alpha_pow, cstate.alpha),
    }
}

/// Comptime-recursive chain: 16 S-box constraint mixes for one full
/// round (after AddRC). Each step reads a witness column, mixes the
/// constraint into `cs`, and writes the witness value into the rolling
/// state.
#[cube]
fn full_round_chain(
    state: &mut Array<u32>,
    rc: &Array<u32>,
    #[comptime] round_idx: u32,
    witness: &Array<u32>,
    witness_col_base: u32,
    row: u32,
    stride: u32,
    #[comptime] sbox_col_base: u32,
    cs: ConstraintAccumulator,
    #[comptime] i: u32,
) -> ConstraintAccumulator {
    if comptime!(i >= 16u32) {
        cs
    } else {
        let s = state[comptime!(i as usize)];
        let c = rc[comptime!((round_idx * 16 + i) as usize)];
        let pre = monty_add::<KoalaBearParameters>(s, c);
        let col = witness_col_base + comptime!(sbox_col_base + i);
        let wit = witness[(col * stride + row) as usize];
        state[comptime!(i as usize)] = wit;
        let cs = mix_sbox(pre, wit, cs);
        full_round_chain(
            state, rc, round_idx, witness, witness_col_base, row, stride,
            sbox_col_base, cs, comptime!(i + 1u32),
        )
    }
}

/// One full round of constraint mixing + state evolution.
#[cube]
fn full_round_constraint(
    state: &mut Array<u32>,
    rc: &Array<u32>,
    #[comptime] round_idx: u32,
    lambda: &Array<u32>,
    w: &Array<u32>,
    witness: &Array<u32>,
    witness_col_base: u32,
    row: u32,
    stride: u32,
    #[comptime] sbox_col_base: u32,
    cs: ConstraintAccumulator,
) -> ConstraintAccumulator {
    let cs = full_round_chain(
        state, rc, round_idx, witness, witness_col_base, row, stride,
        sbox_col_base, cs, 0u32,
    );
    mds_fft_16(state, lambda, w);
    cs
}

/// Comptime-recursive chain: 20 partial rounds, each computing the
/// pre-S-box value via dot-product over `history`, mixing the
/// constraint, and appending the witness value to history.
#[cube]
fn partial_chain(
    history: &mut Array<u32>,
    pw: &Array<u32>,
    po: &Array<u32>,
    witness: &Array<u32>,
    witness_col_base: u32,
    row: u32,
    stride: u32,
    cs: ConstraintAccumulator,
    #[comptime] r: u32,
) -> ConstraintAccumulator {
    if comptime!(r >= 20u32) {
        cs
    } else {
        // Compute pre-S-box: po[r] + Σ_{k=0..16+r} pw[r·36+k] · history[k].
        // u32 accumulator with `let mut` reassignment is fine — only CubeType-
        // with-CubeType-fields trips the macro.
        let mut acc_pre = po[comptime!(r as usize)];
        #[unroll]
        for k in 0u32..(comptime!(16u32 + r)) {
            let weight = pw[comptime!((r * (HISTORY_LEN as u32) + k) as usize)];
            let h = history[comptime!(k as usize)];
            acc_pre = monty_add::<KoalaBearParameters>(
                acc_pre,
                monty_mul::<KoalaBearParameters>(weight, h),
            );
        }
        let col = witness_col_base + comptime!(64u32 + r);
        let wit = witness[(col * stride + row) as usize];
        history[comptime!((16u32 + r) as usize)] = wit;
        let cs = mix_sbox(acc_pre, wit, cs);
        partial_chain(
            history, pw, po, witness, witness_col_base, row, stride, cs,
            comptime!(r + 1u32),
        )
    }
}

// ---------------------------------------------------------------------------
// Public cube subroutine.
// ---------------------------------------------------------------------------

/// Per-row Poseidon1 constraint contribution. Mixes all 148 S-box
/// constraints into `cstate` against the witness. Caller seeds
/// `cstate.alpha_pow` with the desired starting α-power and reads the
/// advanced value (= `alpha_pow_start · α^148`) from the returned
/// struct's `alpha_pow` for the next subroutine in the chain.
///
/// `input_state` is the 16 KB inputs to the permutation. The 148 S-box
/// columns are read from `witness` at `(witness_col_base + c) * stride + row`
/// for `c ∈ [0, 148)`, in round-major layout (cols 0..63 = initial full,
/// 64..83 = partial, 84..147 = terminal full).
#[cube]
pub fn poseidon1_kb16_constraint(
    input_state: &Array<u32>,
    witness: &Array<u32>,
    witness_col_base: u32,
    row: u32,
    stride: u32,
    cstate: ConstraintAccumulator,
) -> ConstraintAccumulator {
    let rc = Array::<u32>::from_data(comptime!(rc_montgomery_flat()));
    let lambda = Array::<u32>::from_data(comptime!(lambda_montgomery()));
    let w = Array::<u32>::from_data(comptime!(ws_montgomery()));
    let pw = Array::<u32>::from_data(comptime!(partial_weights_flat_montgomery()));
    let po = Array::<u32>::from_data(comptime!(partial_offsets_montgomery()));
    let tw = Array::<u32>::from_data(comptime!(terminal_weights_flat_montgomery()));
    let to = Array::<u32>::from_data(comptime!(terminal_offsets_montgomery()));

    let mut state = Array::<u32>::new(comptime!(16usize));
    #[unroll]
    for i in 0u32..16u32 {
        state[comptime!(i as usize)] = input_state[comptime!(i as usize)];
    }

    // ---- 4 initial full rounds ----
    let cs = cstate;
    let cs = full_round_constraint(&mut state, &rc, 0u32, &lambda, &w, witness, witness_col_base, row, stride, 0u32, cs);
    let cs = full_round_constraint(&mut state, &rc, 1u32, &lambda, &w, witness, witness_col_base, row, stride, 16u32, cs);
    let cs = full_round_constraint(&mut state, &rc, 2u32, &lambda, &w, witness, witness_col_base, row, stride, 32u32, cs);
    let cs = full_round_constraint(&mut state, &rc, 3u32, &lambda, &w, witness, witness_col_base, row, stride, 48u32, cs);

    // ---- 20 partial rounds via history-of-16+r ----
    let mut history = Array::<u32>::new(comptime!(HISTORY_LEN));
    #[unroll]
    for k in 0u32..16u32 {
        history[comptime!(k as usize)] = state[comptime!(k as usize)];
    }
    let cs = partial_chain(
        &mut history, &pw, &po, witness, witness_col_base, row, stride, cs, 0u32,
    );

    // ---- Reconstruct state for terminal full rounds ----
    #[unroll]
    for i in 0u32..16u32 {
        let mut s = to[comptime!(i as usize)];
        #[unroll]
        for k in 0u32..(comptime!(HISTORY_LEN as u32)) {
            let weight = tw[comptime!((i * (HISTORY_LEN as u32) + k) as usize)];
            let h = history[comptime!(k as usize)];
            s = monty_add::<KoalaBearParameters>(s, monty_mul::<KoalaBearParameters>(weight, h));
        }
        state[comptime!(i as usize)] = s;
    }

    // ---- 4 terminal full rounds ----
    let cs = full_round_constraint(&mut state, &rc, 24u32, &lambda, &w, witness, witness_col_base, row, stride, 84u32, cs);
    let cs = full_round_constraint(&mut state, &rc, 25u32, &lambda, &w, witness, witness_col_base, row, stride, 100u32, cs);
    let cs = full_round_constraint(&mut state, &rc, 26u32, &lambda, &w, witness, witness_col_base, row, stride, 116u32, cs);
    let cs = full_round_constraint(&mut state, &rc, 27u32, &lambda, &w, witness, witness_col_base, row, stride, 132u32, cs);

    cs
}

// ---------------------------------------------------------------------------
// Host shadow of the cube path. Same algorithm — KB rolling state, KB
// witness, KB^4 accumulator with diff lifted to KB^4 — used as the test
// oracle for the cube version.
// ---------------------------------------------------------------------------

/// Host-side mirror of [`poseidon1_kb16_constraint`]. Same algorithm with
/// KB state and KB witness, lifting each diff to KB^4 before mixing.
/// Used to cross-check the cube path on host inputs.
pub fn host_constraint_kb_witness(
    input_state: &[KoalaBear; 16],
    witness: &[KoalaBear; N_WITNESS_SBOXES],
    cstate: ConstraintAccumulator,
) -> ConstraintAccumulator {
    let rc = round_constants_lifted();
    let t = tables();

    let mix = |pre: KoalaBear, wit: KoalaBear, cs: ConstraintAccumulator| -> ConstraintAccumulator {
        let predicted = pre * pre * pre;
        let diff_kb = predicted - wit;
        let diff_ext4 =
            Ext4::<KoalaBear4Parameters>::from_raw([diff_kb.raw(), 0u32, 0u32, 0u32]);
        let term = cs.alpha_pow * diff_ext4;
        ConstraintAccumulator {
            alpha: cs.alpha,
            acc: cs.acc + term,
            alpha_pow: cs.alpha_pow * cs.alpha,
        }
    };

    let mut state = *input_state;
    let mut cs = cstate;

    // ---- 4 initial full rounds (cols 0..63) ----
    for r in 0..N_INITIAL_FULL {
        for i in 0..16 {
            let pre = state[i] + rc[r][i];
            let wit = witness[r * 16 + i];
            cs = mix(pre, wit, cs);
            state[i] = wit;
        }
        mds_naive(&mut state);
    }

    // ---- 20 partial rounds via history (cols 64..83) ----
    let mut history = [MontyField::ZERO; HISTORY_LEN];
    for k in 0..16 {
        history[k] = state[k];
    }
    for r in 0..N_PARTIAL {
        let mut pre = t.partial_offsets[r];
        for k in 0..(16 + r) {
            pre = pre + t.partial_weights[r][k] * history[k];
        }
        let wit = witness[64 + r];
        cs = mix(pre, wit, cs);
        history[16 + r] = wit;
    }

    // ---- Reconstruct state for terminal full rounds ----
    for i in 0..16 {
        let mut s = t.terminal_offsets[i];
        for k in 0..HISTORY_LEN {
            s = s + t.terminal_weights[i][k] * history[k];
        }
        state[i] = s;
    }

    // ---- 4 terminal full rounds (cols 84..147) ----
    for r_idx in 0..N_INITIAL_FULL {
        let r = N_INITIAL_FULL + N_PARTIAL + r_idx;
        for i in 0..16 {
            let pre = state[i] + rc[r][i];
            let wit = witness[84 + r_idx * 16 + i];
            cs = mix(pre, wit, cs);
            state[i] = wit;
        }
        mds_naive(&mut state);
    }

    cs
}
