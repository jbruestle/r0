//! Partial-round weight precomputation (host-side).
//!
//! Per the design, every partial-round S-box input is a linear
//! combination of:
//!
//! - `s_full_end[16]` — state at the end of the 4 initial full rounds.
//! - `partial_sbox_out[r]` — cubed values from prior partial rounds 0..r-1.
//!
//! Stack these into a 36-element `basis` and represent the state at any
//! point during the partial-round phase as
//!
//! ```text
//! state[i] = Σ_k coeffs[i][k] · basis[k] + offsets[i]
//! ```
//!
//! At the start of partial round r:
//! - `partial_weights[r] = coeffs[0][..]` (length 36; entries at index ≥ 16+r are zero)
//! - `partial_offsets[r] = offsets[0] + RC[N_INITIAL_FULL + r][0]` (folds in the AddRC of this round)
//!
//! After the 20 partial rounds (state going into the 1st terminal full round):
//! - `terminal_weights[i] = coeffs[i][..]` (length 36)
//! - `terminal_offsets[i] = offsets[i]`
//!
//! Per-round update sequence: AddRC (offsets += RC), SBox-slot-0 (coeffs[0] := e_{16+r}; offsets[0] := 0),
//! MDS (coeffs ← MDS · coeffs columnwise; offsets ← MDS · offsets).
//!
//! This module derives those tables once via `OnceLock`, and ships a
//! host-side `permute_via_history` that uses them — the cross-check
//! against the naive `host_permute` is what convinces us the derivation
//! is correct.

use std::sync::OnceLock;

use r0_field::{KoalaBear, MontyField};

use crate::host_ref::{
    mds_col_lifted, round_constants_lifted, N_INITIAL_FULL, N_PARTIAL, N_ROUNDS,
};

pub(crate) const HISTORY_LEN: usize = 16 + N_PARTIAL; // 36

/// Precomputed partial-round structure.
#[derive(Debug)]
pub(crate) struct PartialTables {
    /// `partial_weights[r][k]` for r ∈ 0..20, k ∈ 0..36. Entries at k ≥ 16+r are zero.
    pub partial_weights: [[KoalaBear; HISTORY_LEN]; N_PARTIAL],
    /// `partial_offsets[r]` for r ∈ 0..20.
    pub partial_offsets: [KoalaBear; N_PARTIAL],
    /// `terminal_weights[i][k]` for i ∈ 0..16, k ∈ 0..36 (state recovery after partials).
    pub terminal_weights: [[KoalaBear; HISTORY_LEN]; 16],
    /// `terminal_offsets[i]` for i ∈ 0..16.
    pub terminal_offsets: [KoalaBear; 16],
}

/// Derive the partial-round tables from the round constants and MDS column.
///
/// Symbolically evolves the (coeffs, offsets) representation of the state
/// through the 20 partial rounds. Cost: ~200k base-field multiplies; one
/// `OnceLock` init at first use.
fn derive() -> PartialTables {
    let rc = round_constants_lifted();
    let mds = mds_col_lifted();

    let mds_at = |i: usize, j: usize| -> KoalaBear {
        // mds[i][j] = MDS_CIRC_COL[(16 + i - j) % 16]
        mds[(16 + i - j) % 16]
    };

    // Initial state: state[i] = basis[i] (i.e. s_full_end[i]).
    let mut coeffs = [[MontyField::ZERO; HISTORY_LEN]; 16];
    let mut offsets = [MontyField::ZERO; 16];
    let one = KoalaBear::from_canonical(1);
    for i in 0..16 {
        coeffs[i][i] = one;
    }

    let mut partial_weights = [[MontyField::ZERO; HISTORY_LEN]; N_PARTIAL];
    let mut partial_offsets = [MontyField::ZERO; N_PARTIAL];

    for r in 0..N_PARTIAL {
        let round_index = N_INITIAL_FULL + r;

        // Snapshot weights[r] BEFORE this round's AddRC affects offset[0].
        // pre_sbox = state[0] + RC[round_index][0]
        //          = Σ_k coeffs[0][k] · basis[k] + (offsets[0] + RC[round_index][0])
        for k in 0..HISTORY_LEN {
            partial_weights[r][k] = coeffs[0][k];
        }
        partial_offsets[r] = offsets[0] + rc[round_index][0];

        // Apply this round's transformations to the (coeffs, offsets) representation.

        // 1. AddRC: offsets += RC. Coeffs unchanged.
        for i in 0..16 {
            offsets[i] = offsets[i] + rc[round_index][i];
        }

        // 2. SBox slot 0: state[0] becomes basis[16+r], i.e. coeffs[0] = e_{16+r}, offsets[0] = 0.
        for k in 0..HISTORY_LEN {
            coeffs[0][k] = MontyField::ZERO;
        }
        coeffs[0][16 + r] = one;
        offsets[0] = MontyField::ZERO;

        // 3. MDS: coeffs ← MDS · coeffs (columnwise), offsets ← MDS · offsets.
        let coeffs_in = coeffs;
        let offsets_in = offsets;
        for i in 0..16 {
            for k in 0..HISTORY_LEN {
                let mut acc = MontyField::ZERO;
                for j in 0..16 {
                    acc = acc + mds_at(i, j) * coeffs_in[j][k];
                }
                coeffs[i][k] = acc;
            }
            let mut acc = MontyField::ZERO;
            for j in 0..16 {
                acc = acc + mds_at(i, j) * offsets_in[j];
            }
            offsets[i] = acc;
        }
    }

    // After 20 partial rounds: state for terminal full rounds.
    let terminal_weights = coeffs;
    let terminal_offsets = offsets;

    PartialTables {
        partial_weights,
        partial_offsets,
        terminal_weights,
        terminal_offsets,
    }
}

pub(crate) fn tables() -> &'static PartialTables {
    static TABLES: OnceLock<PartialTables> = OnceLock::new();
    TABLES.get_or_init(derive)
}

/// Host implementation of the permutation using the history-of-16+r
/// formulation for the partial rounds. Cross-checks against
/// [`crate::host_permute`] in tests; matching means the precomputation is
/// correct and the cube path can mirror this structure.
pub fn host_permute_via_history(state: &mut [KoalaBear; 16]) {
    let rc = round_constants_lifted();
    let mds = mds_col_lifted();
    let tables = tables();

    let mds_at = |i: usize, j: usize| mds[(16 + i - j) % 16];

    // 4 initial full rounds: same as naive.
    for r in 0..N_INITIAL_FULL {
        for i in 0..16 {
            state[i] = state[i] + rc[r][i];
        }
        for i in 0..16 {
            let s = state[i];
            state[i] = s * s * s;
        }
        // MDS via naive matvec (matches host_permute).
        let input = *state;
        for i in 0..16 {
            let mut acc = MontyField::ZERO;
            for j in 0..16 {
                acc = acc + mds_at(i, j) * input[j];
            }
            state[i] = acc;
        }
    }

    // 20 partial rounds via history dot product.
    let mut history = [MontyField::ZERO; HISTORY_LEN];
    for k in 0..16 {
        history[k] = state[k];
    }
    for r in 0..N_PARTIAL {
        let mut pre_sbox = tables.partial_offsets[r];
        for k in 0..(16 + r) {
            pre_sbox = pre_sbox + tables.partial_weights[r][k] * history[k];
        }
        history[16 + r] = pre_sbox * pre_sbox * pre_sbox;
    }

    // Reconstruct state for terminal full rounds.
    for i in 0..16 {
        let mut acc = tables.terminal_offsets[i];
        for k in 0..HISTORY_LEN {
            acc = acc + tables.terminal_weights[i][k] * history[k];
        }
        state[i] = acc;
    }

    // 4 terminal full rounds: same as naive.
    for r in N_INITIAL_FULL + N_PARTIAL..N_ROUNDS {
        for i in 0..16 {
            state[i] = state[i] + rc[r][i];
        }
        for i in 0..16 {
            let s = state[i];
            state[i] = s * s * s;
        }
        let input = *state;
        for i in 0..16 {
            let mut acc = MontyField::ZERO;
            for j in 0..16 {
                acc = acc + mds_at(i, j) * input[j];
            }
            state[i] = acc;
        }
    }
}

