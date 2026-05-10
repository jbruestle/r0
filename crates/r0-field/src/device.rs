//! Process-shared exclusive lock around a cubecl device.
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
//! On `wasm32` (browser builds) the lock is a no-op: the browser is
//! single-threaded and there are no concurrent processes to coordinate
//! with.

use cubecl::prelude::*;

/// Process-wide exclusive guard around a cubecl device.
///
/// Acquire one with [`Device::acquire`] (default device) or
/// [`Device::acquire_for`] (specific device). The lock is released
/// automatically when the `Device` drops. While held, other callers in
/// this process or any other process that try to acquire the same
/// runtime's lock will block.
///
/// Pass `&Device<R>` to executor constructors (e.g.
/// `NttExec::new(&device, ...)`); the executor reads
/// [`inner`](Self::inner) to talk to cubecl and otherwise ignores the
/// wrapper.
pub struct Device<R: Runtime> {
    inner: R::Device,
    #[cfg(not(target_arch = "wasm32"))]
    _lock_file: std::fs::File,
}

impl<R: Runtime> Device<R> {
    /// Acquire exclusive access to the default device for runtime `R`,
    /// blocking until any other holder releases.
    pub fn acquire() -> Self
    where
        R::Device: Default,
    {
        Self::acquire_for(R::Device::default())
    }

    /// Acquire exclusive access to a specific device. Blocks until any
    /// other holder of the same runtime's lock releases. The provided
    /// device is wrapped as-is — the lock is per-runtime, not
    /// per-device-instance, so two `Device` constructors against the
    /// same runtime serialize regardless of which underlying device they
    /// wrap.
    pub fn acquire_for(device: R::Device) -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        {
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
                .unwrap_or_else(|e| {
                    panic!("failed to open device lock at {}: {e}", path.display())
                });
            file.lock_exclusive()
                .unwrap_or_else(|e| panic!("failed to acquire device lock for {key}: {e}"));

            Self {
                inner: device,
                _lock_file: file,
            }
        }

        #[cfg(target_arch = "wasm32")]
        {
            Self { inner: device }
        }
    }

    /// The underlying cubecl device. Pass to `R::client(...)` etc.
    pub fn inner(&self) -> &R::Device {
        &self.inner
    }
}
