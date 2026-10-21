//! Toolchain management: where toolchains live, how they are downloaded,
//! verified, unpacked and found. Zero third-party dependencies.
//!
//! Layout under the Jax home (`JAX_HOME`, default `~/.jax`):
//!
//! ```text
//! toolchains/<version>/   complete, immutable toolchain directories
//! store/                  content-addressed package cache (shared, read-only)
//! cache/                  build cache
//! tmp/  locks/            scratch space and lock files
//! config.toml             user preferences (never affects build output)
//! ```

pub mod fetch;
pub mod home;
pub mod install;
pub mod sha256;
pub mod tar;

pub use home::{exe_name, host_target, Config, Home, Toolchains, DEFAULT_DIST_URL};
pub use install::{ensure_installed, latest_version, manifest_text, Dist};
