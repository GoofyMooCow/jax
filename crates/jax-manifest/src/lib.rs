//! `jax.toml` / `jax.lock` handling, with no third-party dependencies.

pub mod lock;
pub mod manifest;
pub mod toml;
pub mod version;

pub use lock::{LockedPackage, Lockfile};
pub use manifest::{
    find_manifest, scaffold, Build, Capsule, CapsuleLevel, DepSource, Dependency, Kind, Manifest,
    ManifestError, Optimize, Profile, Project, LOCK_FILE, MANIFEST_FILE, TARGETS,
};
pub use version::Version;
