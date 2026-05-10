//! Process-shared exclusive lock around a cubecl device, plus the
//! shared resources executors need to talk to it.
//!
//! cubecl backends like `WgpuRuntime` share a single GPU across the whole
//! machine; `cargo test` runs each integration-test binary as its own
//! process in parallel, and concurrent kernel launches against the same
//! GPU can fail under load. Intra-process locks (`parking_lot::Mutex`,
//! `serial_test`) don't help across binaries because each is its own
//! process with its own memory.
//!
//! [`Device<R>`] solves this by holding an OS-level exclusive file lock
//! keyed by the cubecl runtime's type name. Construct one per scope that
//! needs the device (typically per `#[test]`); when the `Device` drops,
//! the lock releases. Other processes / threads waiting for the same
//! runtime will then proceed.
//!
//! The lock is per-runtime, so wgpu and CPU tests do not block each
//! other — concurrency is reduced only where it must be.
//!
//! On `wasm32` (browser builds) the file lock is a no-op: the browser is
//! single-threaded and there are no concurrent processes to coordinate
//! with.
//!
//! # Shared resources
//!
//! Beyond the lock, [`Device<R>`] also owns a [`ComputeClient<R>`] and a
//! single scratch [`Handle`] sized at acquire time. Executors borrow
//! [`Device::client`] and [`Device::scratch`] (cheap reference-counted
//! clones) instead of allocating their own — one buffer per device,
//! shared across `NttExec`, future `ScanExec`, etc., serialized by the
//! same lock that protects the device itself.

use cubecl::prelude::*;
use cubecl::server::Handle;

/// Default scratch size used by [`Device::acquire`] / [`Device::acquire_for`]:
/// 64 MiB. Internal — callers that need a different budget go through
/// the `acquire_with_scratch[_for]` constructors with an explicit value.
const DEFAULT_SCRATCH_BYTES: usize = 64 * 1024 * 1024;

/// Process-wide exclusive guard around a cubecl device, with a shared
/// scratch buffer.
///
/// Acquire one with [`Device::acquire`] (default device, default 64 MiB
/// scratch), [`Device::acquire_for`] (specific device, default scratch),
/// [`Device::acquire_with_scratch`] / [`Device::acquire_with_scratch_for`]
/// (explicit scratch budget). The OS file lock is released automatically
/// when the `Device` drops; while held, other callers in this process or
/// any other process that try to acquire the same runtime's lock will
/// block.
///
/// Pass `&Device<R>` to executor constructors (e.g.
/// `NttExec::new(&device)`); the executor pulls the client and scratch
/// handle out via [`client`](Self::client) and [`scratch`](Self::scratch).
/// The scratch buffer is fixed-size for the `Device`'s lifetime —
/// executors share it under the file lock and must each fit within
/// [`scratch_bytes`](Self::scratch_bytes). Size up front for the largest
/// executor that will share this device.
pub struct Device<R: Runtime> {
    inner: R::Device,
    client: ComputeClient<R>,
    scratch: Handle,
    scratch_bytes: usize,
    #[cfg(not(target_arch = "wasm32"))]
    _lock_file: std::fs::File,
}

impl<R: Runtime> Device<R> {
    /// Acquire exclusive access to the default device for runtime `R`
    /// with the default scratch budget (64 MiB). Blocks until any other
    /// holder releases.
    pub fn acquire() -> Self
    where
        R::Device: Default,
    {
        Self::acquire_with_scratch_for(R::Device::default(), DEFAULT_SCRATCH_BYTES)
    }

    /// Acquire exclusive access to a specific device with the default
    /// scratch budget. See [`Device::acquire`] for locking semantics.
    pub fn acquire_for(device: R::Device) -> Self {
        Self::acquire_with_scratch_for(device, DEFAULT_SCRATCH_BYTES)
    }

    /// Acquire exclusive access to the default device with an explicit
    /// scratch budget (in bytes). Pass `0` to skip scratch allocation —
    /// any executor that calls [`scratch`](Self::scratch) on a 0-byte
    /// device will be working with an empty handle.
    pub fn acquire_with_scratch(scratch_bytes: usize) -> Self
    where
        R::Device: Default,
    {
        Self::acquire_with_scratch_for(R::Device::default(), scratch_bytes)
    }

    /// Acquire exclusive access to a specific device with an explicit
    /// scratch budget. The lock is per-runtime, not per-device-instance,
    /// so two `Device` constructors against the same runtime serialize
    /// regardless of which underlying device they wrap.
    pub fn acquire_with_scratch_for(device: R::Device, scratch_bytes: usize) -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        let _lock_file = acquire_lock_file::<R>();

        let client = R::client(&device);
        let scratch = client.empty(scratch_bytes);

        Self {
            inner: device,
            client,
            scratch,
            scratch_bytes,
            #[cfg(not(target_arch = "wasm32"))]
            _lock_file,
        }
    }

    /// The underlying cubecl device handle. Mostly for compatibility
    /// with code that wants to talk to cubecl directly; new executors
    /// should reach for [`client`](Self::client) instead.
    pub fn inner(&self) -> &R::Device {
        &self.inner
    }

    /// The cubecl compute client for this device. Use to allocate
    /// non-scratch buffers (e.g. precomputed twiddle tables) and to
    /// dispatch reads / syncs. Cloning the returned reference is cheap
    /// (Arc-based internally).
    pub fn client(&self) -> &ComputeClient<R> {
        &self.client
    }

    /// Shared scratch buffer for this device. Executors should clone the
    /// returned [`Handle`] (cheap, reference-counted) rather than allocate
    /// their own scratch — one buffer per device, shared under the file
    /// lock. Size is fixed at acquire time; query
    /// [`scratch_bytes`](Self::scratch_bytes) for the budget.
    pub fn scratch(&self) -> &Handle {
        &self.scratch
    }

    /// Size of the shared scratch buffer in bytes (the value passed to
    /// [`acquire_with_scratch`](Self::acquire_with_scratch) /
    /// [`acquire_with_scratch_for`](Self::acquire_with_scratch_for), or
    /// the default 64 MiB for the no-arg constructors).
    pub fn scratch_bytes(&self) -> usize {
        self.scratch_bytes
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn acquire_lock_file<R: Runtime>() -> std::fs::File {
    use fs2::FileExt;
    use std::fs::OpenOptions;

    // Sanitize the runtime type name to a filesystem-friendly key.
    // `core::any::type_name::<WgpuRuntime>()` looks like
    // "cubecl_wgpu::runtime::WgpuRuntime"; we keep it stable across
    // a build but it's just a key — collisions only matter if two
    // runtimes intentionally want to share a lock.
    let key = core::any::type_name::<R>();
    let safe_key: String = key
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    let path = std::env::temp_dir().join(format!("r0-cubecl-{safe_key}.lock"));

    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(&path)
        .unwrap_or_else(|e| panic!("failed to open device lock at {}: {e}", path.display()));
    file.lock_exclusive()
        .unwrap_or_else(|e| panic!("failed to acquire device lock for {key}: {e}"));
    file
}
