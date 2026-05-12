//! Poseidon1 permutation over KoalaBear, width 16.
//!
//! Three call modes share a common 28-round structure (4 initial full +
//! 20 partial + 4 terminal full): pure permutation, permutation +
//! per-S-box witness write, and per-row constraint contribution into a
//! fiat-shamir-mixed accumulator. Bit-for-bit compatible with Plonky3's
//! `default_koalabear_poseidon1_16`. See the crate README for the full
//! design and the rustdoc on each item below for usage.

mod host_ref;
pub use host_ref::{
    host_permute, host_permute_with_trace, mds_naive, MDS_CIRC_COL_CANONICAL, N_WITNESS_SBOXES,
    ROUND_CONSTANTS_CANONICAL,
};

mod partial;
pub use partial::host_permute_via_history;

mod mds;
pub use mds::{dif_ifft_16, dit_fft_16, host_mds_fft};

mod permute;
pub use permute::{poseidon1_kb16_permute, poseidon1_kb16_permute_with_witness};

mod constraint;
pub use constraint::{
    host_constraint_kb_witness, poseidon1_kb16_constraint, ConstraintAccumulator,
};
