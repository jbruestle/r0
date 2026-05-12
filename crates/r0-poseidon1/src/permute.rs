//! Compute-mode Poseidon1 KB16 permutation, as a `#[cube] fn` callable
//! from any cubecl kernel.
//!
//! State is a per-thread `Array::<u32>::new(16)` of raw KB Montgomery
//! u32s. Caller allocates / fills / reads. The 28-round walk uses:
//!
//! - **FFT MDS** for full rounds (4 + 4) — `dif_ifft → λ⊙ → dit_fft`
//!   with constants baked at IR build time via `Array::from_data`.
//! - **History-of-16+r dot product** for the 20 partial rounds — pre-S-box
//!   inputs reconstructed from the 16-element state at the end of the
//!   initial full rounds plus the 20 prior partial-S-box outputs.
//!
//! All constant tables (round constants, FFT lambdas, twiddles, partial
//! weights/offsets, terminal weights/offsets) flow through cubecl's
//! `from_data` mechanism: they emit as WGSL module-scope `const arrays_N
//! = array(...);`, with constant-indexed reads folding to literals
//! downstream.

use cubecl::prelude::*;

use r0_field::{monty_add, monty_mul, monty_sub, KoalaBearParameters};

use crate::host_ref::{round_constants_lifted, N_PARTIAL, N_ROUNDS};
use crate::mds::{lambda_over_16_lifted, ws_lifted};
use crate::partial::{tables, HISTORY_LEN};

// ---------------------------------------------------------------------------
// Host-side constant generators. Each runs once at IR build time; the
// returned Vec is consumed by `Array::<u32>::from_data` and baked as a
// WGSL module-scope `const`.
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
// Butterfly primitives. `#[cube] fn`s with comptime lo/hi indices, taking
// the working-set Array<u32> and the runtime twiddle value (loaded from
// the const twiddle array at the call site).
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

// ---------------------------------------------------------------------------
// FFT MDS — applies `state ← C · state` in place via convolution theorem.
// Calls the butterflies in the same order as leanMultisig's
// `dif_ifft_16_mut` / `dit_fft_16_mut`.
// ---------------------------------------------------------------------------

#[cube]
fn mds_fft_16(state: &mut Array<u32>, lambda: &Array<u32>, w: &Array<u32>) {
    // ---- DIF inverse FFT ----
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

    // ---- Eigenvalue multiply (absorbs 1/16) ----
    #[unroll]
    for i in 0u32..16u32 {
        let s = state[comptime!(i as usize)];
        let l = lambda[comptime!(i as usize)];
        state[comptime!(i as usize)] = monty_mul::<KoalaBearParameters>(s, l);
    }

    // ---- DIT forward FFT ----
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
// Full round: AddRC + cube each slot + FFT MDS. `r` is comptime so the
// AddRC indexes resolve to literal const-array reads.
// ---------------------------------------------------------------------------

#[cube]
fn full_round(
    state: &mut Array<u32>,
    rc: &Array<u32>,
    #[comptime] r: u32,
    lambda: &Array<u32>,
    w: &Array<u32>,
) {
    #[unroll]
    for i in 0u32..16u32 {
        let s = state[comptime!(i as usize)];
        let c = rc[comptime!((r * 16 + i) as usize)];
        state[comptime!(i as usize)] = monty_add::<KoalaBearParameters>(s, c);
    }
    #[unroll]
    for i in 0u32..16u32 {
        let s = state[comptime!(i as usize)];
        let s2 = monty_mul::<KoalaBearParameters>(s, s);
        state[comptime!(i as usize)] = monty_mul::<KoalaBearParameters>(s2, s);
    }
    mds_fft_16(state, lambda, w);
}

/// Full round + per-slot S-box witness write. `sbox_col_base` is the
/// witness column for slot 0 of this round (caller-supplied so the
/// initial vs terminal phase column offset stays comptime).
#[cube]
fn full_round_with_witness(
    state: &mut Array<u32>,
    rc: &Array<u32>,
    #[comptime] r: u32,
    lambda: &Array<u32>,
    w: &Array<u32>,
    witness: &mut Array<u32>,
    witness_col_base: u32,
    row: u32,
    stride: u32,
    #[comptime] sbox_col_base: u32,
) {
    #[unroll]
    for i in 0u32..16u32 {
        let s = state[comptime!(i as usize)];
        let c = rc[comptime!((r * 16 + i) as usize)];
        state[comptime!(i as usize)] = monty_add::<KoalaBearParameters>(s, c);
    }
    #[unroll]
    for i in 0u32..16u32 {
        let s = state[comptime!(i as usize)];
        let s2 = monty_mul::<KoalaBearParameters>(s, s);
        let cubed = monty_mul::<KoalaBearParameters>(s2, s);
        state[comptime!(i as usize)] = cubed;
        // Write the S-box output at column (sbox_col_base + i).
        let col = witness_col_base + comptime!(sbox_col_base + i);
        witness[(col * stride + row) as usize] = cubed;
    }
    mds_fft_16(state, lambda, w);
}

// ---------------------------------------------------------------------------
// The permutation. In place over a 16-slot Array<u32> that the caller
// allocates and fills with raw KB Montgomery u32s.
// ---------------------------------------------------------------------------

/// Apply the width-16 KoalaBear Poseidon1 permutation in place.
///
/// `state` is a 16-slot `Array<u32>` (allocate via
/// `Array::<u32>::new(16)`) of raw Montgomery KB values. Round constants,
/// FFT lambdas and twiddles, and the partial-round weight/offset tables
/// are all baked into the shader as compile-time constants.
#[cube]
pub fn poseidon1_kb16_permute(state: &mut Array<u32>) {
    let rc = Array::<u32>::from_data(comptime!(rc_montgomery_flat()));
    let lambda = Array::<u32>::from_data(comptime!(lambda_montgomery()));
    let w = Array::<u32>::from_data(comptime!(ws_montgomery()));
    let pw = Array::<u32>::from_data(comptime!(partial_weights_flat_montgomery()));
    let po = Array::<u32>::from_data(comptime!(partial_offsets_montgomery()));
    let tw = Array::<u32>::from_data(comptime!(terminal_weights_flat_montgomery()));
    let to = Array::<u32>::from_data(comptime!(terminal_offsets_montgomery()));

    // ---- 4 initial full rounds ----
    full_round(state, &rc, 0u32, &lambda, &w);
    full_round(state, &rc, 1u32, &lambda, &w);
    full_round(state, &rc, 2u32, &lambda, &w);
    full_round(state, &rc, 3u32, &lambda, &w);

    // ---- 20 partial rounds via history-of-16+r ----
    let mut history = Array::<u32>::new(comptime!(HISTORY_LEN));
    #[unroll]
    for k in 0u32..16u32 {
        history[comptime!(k as usize)] = state[comptime!(k as usize)];
    }
    // Tail of history (slots 16..36) is uninitialized. Each iteration
    // writes its own slot before any later iteration reads it.
    #[unroll]
    for r in 0u32..20u32 {
        // pre_sbox = partial_offsets[r] + Σ_{k=0..16+r} pw[r*36+k] * history[k].
        let mut acc = po[comptime!(r as usize)];
        #[unroll]
        for k in 0u32..(comptime!(16u32 + r)) {
            let weight = pw[comptime!((r * (HISTORY_LEN as u32) + k) as usize)];
            let h = history[comptime!(k as usize)];
            acc = monty_add::<KoalaBearParameters>(acc, monty_mul::<KoalaBearParameters>(weight, h));
        }
        // Cube and store as the new partial-S-box output.
        let sq = monty_mul::<KoalaBearParameters>(acc, acc);
        history[comptime!((16u32 + r) as usize)] = monty_mul::<KoalaBearParameters>(sq, acc);
    }

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
    full_round(state, &rc, 24u32, &lambda, &w);
    full_round(state, &rc, 25u32, &lambda, &w);
    full_round(state, &rc, 26u32, &lambda, &w);
    full_round(state, &rc, 27u32, &lambda, &w);
}

/// As [`poseidon1_kb16_permute`] but additionally writes all 148 S-box
/// outputs to the witness buffer in round-major layout. Column `c` is
/// stored at `witness[(witness_col_base + c) * stride + row]`.
///
/// Layout (148 columns total):
///
/// | Range | Cols | Contents |
/// |---|---|---|
/// | `[0, 64)` | 64 | 4 initial full rounds × 16 slots |
/// | `[64, 84)` | 20 | 20 partial rounds × slot 0 |
/// | `[84, 148)` | 64 | 4 terminal full rounds × 16 slots |
#[cube]
pub fn poseidon1_kb16_permute_with_witness(
    state: &mut Array<u32>,
    witness: &mut Array<u32>,
    witness_col_base: u32,
    row: u32,
    stride: u32,
) {
    let rc = Array::<u32>::from_data(comptime!(rc_montgomery_flat()));
    let lambda = Array::<u32>::from_data(comptime!(lambda_montgomery()));
    let w = Array::<u32>::from_data(comptime!(ws_montgomery()));
    let pw = Array::<u32>::from_data(comptime!(partial_weights_flat_montgomery()));
    let po = Array::<u32>::from_data(comptime!(partial_offsets_montgomery()));
    let tw = Array::<u32>::from_data(comptime!(terminal_weights_flat_montgomery()));
    let to = Array::<u32>::from_data(comptime!(terminal_offsets_montgomery()));

    // ---- 4 initial full rounds: cols 0..64 ----
    full_round_with_witness(state, &rc, 0u32, &lambda, &w, witness, witness_col_base, row, stride, 0u32);
    full_round_with_witness(state, &rc, 1u32, &lambda, &w, witness, witness_col_base, row, stride, 16u32);
    full_round_with_witness(state, &rc, 2u32, &lambda, &w, witness, witness_col_base, row, stride, 32u32);
    full_round_with_witness(state, &rc, 3u32, &lambda, &w, witness, witness_col_base, row, stride, 48u32);

    // ---- 20 partial rounds via history-of-16+r: cols 64..84 ----
    let mut history = Array::<u32>::new(comptime!(HISTORY_LEN));
    #[unroll]
    for k in 0u32..16u32 {
        history[comptime!(k as usize)] = state[comptime!(k as usize)];
    }
    #[unroll]
    for r in 0u32..20u32 {
        let mut acc = po[comptime!(r as usize)];
        #[unroll]
        for k in 0u32..(comptime!(16u32 + r)) {
            let weight = pw[comptime!((r * (HISTORY_LEN as u32) + k) as usize)];
            let h = history[comptime!(k as usize)];
            acc = monty_add::<KoalaBearParameters>(acc, monty_mul::<KoalaBearParameters>(weight, h));
        }
        let sq = monty_mul::<KoalaBearParameters>(acc, acc);
        let cubed = monty_mul::<KoalaBearParameters>(sq, acc);
        history[comptime!((16u32 + r) as usize)] = cubed;
        // Write partial-round S-box output at column (64 + r).
        let col = witness_col_base + comptime!(64u32 + r);
        witness[(col * stride + row) as usize] = cubed;
    }

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

    // ---- 4 terminal full rounds: cols 84..148 ----
    full_round_with_witness(state, &rc, 24u32, &lambda, &w, witness, witness_col_base, row, stride, 84u32);
    full_round_with_witness(state, &rc, 25u32, &lambda, &w, witness, witness_col_base, row, stride, 100u32);
    full_round_with_witness(state, &rc, 26u32, &lambda, &w, witness, witness_col_base, row, stride, 116u32);
    full_round_with_witness(state, &rc, 27u32, &lambda, &w, witness, witness_col_base, row, stride, 132u32);
}
