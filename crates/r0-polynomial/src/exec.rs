//! `PolyDivExec<F, R>` — thin wrapper around [`ScanExec`] for the
//! [`DivByXMinusZ`] recipe.

use core::marker::PhantomData;

use cubecl::prelude::*;
use cubecl::server::Handle;

use r0_cube::{Device, ScanExec};

use crate::div_by_x_minus_z::DivByXMinusZ;
use crate::pair_scan::PairScanLayout;

/// Device-resident polynomial-division executor.
///
/// Bakes `(device, log_n_max, max_batch, F)` at construction; the per-
/// call [`div_by_x_minus_z`](Self::div_by_x_minus_z) picks spine depth
/// based on `log_n` like [`ScanExec::run`].
pub struct PolyDivExec<F: PairScanLayout, R: Runtime> {
    inner: ScanExec<R, DivByXMinusZ<F>>,
    _f: PhantomData<F>,
}

impl<F: PairScanLayout, R: Runtime> PolyDivExec<F, R> {
    /// Construct an executor for division by `(x − z)` over `F`-coefficient
    /// polynomials of length up to `2^log_n_max`, with up to `max_batch`
    /// rows per call.
    pub fn new(device: &Device<R>, log_n_max: u32, max_batch: usize) -> Self {
        Self {
            inner: ScanExec::new(device, log_n_max, max_batch),
            _f: PhantomData,
        }
    }

    /// Underlying cubecl client, for buffer allocation and read-back.
    pub fn client(&self) -> &ComputeClient<R> {
        self.inner.client()
    }

    /// In-place division by `(x − z)` for `batch` polynomials of length
    /// `2^log_n`.
    ///
    /// Buffer expectations:
    /// - `buf`: `batch · 2^log_n · F::DEGREE` u32s, transposed layout
    ///   (per [`ExtField::load`](r0_field::ExtField::load) — component
    ///   `c` of element `i` at offset `c · n + i` within each
    ///   polynomial slice).
    /// - `zs`: `batch · F::DEGREE` u32s, also transposed (component `c`
    ///   of row `b` at offset `c · batch + b`).
    ///
    /// Output convention `rotate=true`: in each polynomial slot `[0..n−1]`
    /// holds the quotient (lowest degree first) and slot `[n−1]` holds
    /// the remainder `r = p(z)`.
    pub fn div_by_x_minus_z(&self, buf: &Handle, zs: &Handle, log_n: u32, batch: usize) {
        // Output aliases input — `ScanExec::run` documents this is allowed.
        self.inner.run(zs, buf, buf, log_n, batch);
    }
}
