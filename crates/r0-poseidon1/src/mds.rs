//! Host-side FFT MDS for KB16, plus precomputed twiddle/lambda tables
//! that the cube path will pick up.
//!
//! Convolution-theorem evaluation of the circulant 16-MDS:
//!
//! ```text
//! C · x = DIT_FFT( λ ⊙ DIF_IFFT(x) )
//! ```
//!
//! where λ are the eigenvalues of `C` with the inverse-FFT scaling
//! (`1/16`) absorbed: `λ = DIF_IFFT(MDS_CIRC_COL) · 16⁻¹`.
//!
//! Per multiply: ~17 + 16 + 17 = 50 monty_muls (vs ~256 naive).
//!
//! Butterflies and twiddle ordering ported from leanMultisig's
//! `dif_ifft_16_mut` / `dit_fft_16_mut`. Same MDS column → same lambda
//! values, verified against [`crate::host_ref::mds_naive`] in tests.

use std::sync::OnceLock;

use r0_field::{KoalaBear, MontyField};

use crate::host_ref::mds_col_lifted;

/// Powers of the primitive 16th root of unity in KoalaBear, canonical form.
/// `W_CANONICAL[k] = ω^k` for `k ∈ 0..8`. (We never need ω^k for k ≥ 8 in
/// the butterfly chain — those would be -ω^{k-8}, which the chain encodes
/// as a sign flip via `bt`/`neg_dif`.)
///
/// Cross-checks against `r0_field::KoalaBearParameters::TWO_ADIC_GENERATORS[4]`
/// (the primitive 16th root). Values match leanMultisig's `W1..W7` constants.
pub(crate) const W_CANONICAL: [u32; 8] = [
    1,           // ω^0
    0x08dbd69c,  // ω^1
    0x6832fe4a,  // ω^2
    0x27ae21e2,  // ω^3
    0x7e010002,  // ω^4
    0x3a89a025,  // ω^5
    0x174e3650,  // ω^6
    0x27dfce22,  // ω^7
];

/// `16⁻¹ mod p_KB`. Verified: `16 · 1997537281 mod 0x7f000001 = 1`.
pub(crate) const INV_16_CANONICAL: u32 = 1997537281;

/// Twiddle powers ω^0..ω^7, in Montgomery form.
pub(crate) fn ws_lifted() -> &'static [KoalaBear; 8] {
    static W: OnceLock<[KoalaBear; 8]> = OnceLock::new();
    W.get_or_init(|| {
        let mut out = [MontyField::ZERO; 8];
        for k in 0..8 {
            out[k] = KoalaBear::from_canonical(W_CANONICAL[k]);
        }
        out
    })
}

/// `16⁻¹` in Montgomery form.
pub(crate) fn inv_16_lifted() -> KoalaBear {
    static INV: OnceLock<KoalaBear> = OnceLock::new();
    *INV.get_or_init(|| KoalaBear::from_canonical(INV_16_CANONICAL))
}

// --------------------------------------------------------------------------
// Butterflies — same operations as leanMultisig (host-side, on KoalaBear).
// --------------------------------------------------------------------------

#[inline(always)]
fn bt(v: &mut [KoalaBear; 16], lo: usize, hi: usize) {
    let a = v[lo];
    let b = v[hi];
    v[lo] = a + b;
    v[hi] = a - b;
}

#[inline(always)]
fn dit(v: &mut [KoalaBear; 16], lo: usize, hi: usize, t: KoalaBear) {
    let a = v[lo];
    let tb = v[hi] * t;
    v[lo] = a + tb;
    v[hi] = a - tb;
}

#[inline(always)]
fn neg_dif(v: &mut [KoalaBear; 16], lo: usize, hi: usize, t: KoalaBear) {
    let a = v[lo];
    let b = v[hi];
    v[lo] = a + b;
    v[hi] = (b - a) * t;
}

/// Decimation-in-frequency inverse FFT of length 16, in place.
/// Up-to-scaling: result is `16 · IFFT(input)`. The `16` factor gets
/// absorbed into the lambda eigenvalues so it doesn't show up at the call
/// site.
pub fn dif_ifft_16(v: &mut [KoalaBear; 16]) {
    let w = ws_lifted();
    bt(v, 0, 8);
    neg_dif(v, 1, 9, w[7]);
    neg_dif(v, 2, 10, w[6]);
    neg_dif(v, 3, 11, w[5]);
    neg_dif(v, 4, 12, w[4]);
    neg_dif(v, 5, 13, w[3]);
    neg_dif(v, 6, 14, w[2]);
    neg_dif(v, 7, 15, w[1]);
    bt(v, 0, 4);
    neg_dif(v, 1, 5, w[6]);
    neg_dif(v, 2, 6, w[4]);
    neg_dif(v, 3, 7, w[2]);
    bt(v, 8, 12);
    neg_dif(v, 9, 13, w[6]);
    neg_dif(v, 10, 14, w[4]);
    neg_dif(v, 11, 15, w[2]);
    bt(v, 0, 2);
    neg_dif(v, 1, 3, w[4]);
    bt(v, 4, 6);
    neg_dif(v, 5, 7, w[4]);
    bt(v, 8, 10);
    neg_dif(v, 9, 11, w[4]);
    bt(v, 12, 14);
    neg_dif(v, 13, 15, w[4]);
    bt(v, 0, 1);
    bt(v, 2, 3);
    bt(v, 4, 5);
    bt(v, 6, 7);
    bt(v, 8, 9);
    bt(v, 10, 11);
    bt(v, 12, 13);
    bt(v, 14, 15);
}

/// Decimation-in-time forward FFT of length 16, in place.
pub fn dit_fft_16(v: &mut [KoalaBear; 16]) {
    let w = ws_lifted();
    bt(v, 0, 1);
    bt(v, 2, 3);
    bt(v, 4, 5);
    bt(v, 6, 7);
    bt(v, 8, 9);
    bt(v, 10, 11);
    bt(v, 12, 13);
    bt(v, 14, 15);
    bt(v, 0, 2);
    dit(v, 1, 3, w[4]);
    bt(v, 4, 6);
    dit(v, 5, 7, w[4]);
    bt(v, 8, 10);
    dit(v, 9, 11, w[4]);
    bt(v, 12, 14);
    dit(v, 13, 15, w[4]);
    bt(v, 0, 4);
    dit(v, 1, 5, w[2]);
    dit(v, 2, 6, w[4]);
    dit(v, 3, 7, w[6]);
    bt(v, 8, 12);
    dit(v, 9, 13, w[2]);
    dit(v, 10, 14, w[4]);
    dit(v, 11, 15, w[6]);
    bt(v, 0, 8);
    dit(v, 1, 9, w[1]);
    dit(v, 2, 10, w[2]);
    dit(v, 3, 11, w[3]);
    dit(v, 4, 12, w[4]);
    dit(v, 5, 13, w[5]);
    dit(v, 6, 14, w[6]);
    dit(v, 7, 15, w[7]);
}

/// FFT eigenvalues `λ_i / 16`, where `λ` are the eigenvalues of the
/// circulant MDS matrix. The `/16` absorbs the inverse-FFT scaling so
/// `mds_fft_16` doesn't need a separate normalization step.
pub(crate) fn lambda_over_16_lifted() -> &'static [KoalaBear; 16] {
    static LAMBDA: OnceLock<[KoalaBear; 16]> = OnceLock::new();
    LAMBDA.get_or_init(|| {
        let mut col = *mds_col_lifted();
        dif_ifft_16(&mut col);
        let inv = inv_16_lifted();
        col.map(|l| l * inv)
    })
}

/// Apply the circulant MDS matrix in place via the FFT path:
/// `state ← C · state = DIT_FFT( (λ/16) ⊙ DIF_IFFT(state) )`.
pub fn host_mds_fft(state: &mut [KoalaBear; 16]) {
    dif_ifft_16(state);
    let lambda = lambda_over_16_lifted();
    for i in 0..16 {
        state[i] = state[i] * lambda[i];
    }
    dit_fft_16(state);
}
