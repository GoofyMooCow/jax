//! The Jax home directory, the installed-toolchain store and user config.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use jax_manifest::toml::{self, Value};
use jax_manifest::Version;

/// The official distribution (static files on GitHub Pages for now).
pub const DEFAULT_DIST_URL: &str = "https://goofymoocow.github.io/jax/dist";

/// The platform Jax is running on, as a Jax target name.
pub fn host_target() -> &'static str {
    let os = if cfg!(target_os = "windows") {
        "windows"
    } else if cfg!(target_os = "macos") {
        "macos"
    } else {
        "linux"
    };
    let arch = if cfg!(target_arch = "aarch64") { "arm64" } else { "x64" };
    match (os, arch) {
        ("windows", "x64") => "windows-x64",
        ("windows", _) => "windows-arm64",
        ("macos", "x64") => "macos-x64",
        ("macos", _) => "macos-arm64",
        (_, "x64") => "linux-x64",
        _ => "linux-arm64",
    }
}

/// `name` with the platform's executable suffix.
pub fn exe_name(base: &str) -> String {
    if cfg!(windows) {
        format!("{base}.exe")
    } else {
        base.to_string()
    }
}

#[derive(Clone, Debug)]
pub struct Home {
    pub root: PathBuf,
}

impl Home {
    /// `JAX_HOME`, else `~/.jax` (`%LOCALAPPDATA%/Jax` on Windows).
    pub fn discover() -> Result<Home, String> {
        if let Some(h) = env::var_os("JAX_HOME") {
            if !h.is_empty() {
                return Ok(Home { root: PathBuf::from(h) });
            }
        }
        if cfg!(windows) {
            if let Some(d) = env::var_os("LOCALAPPDATA") {
                return Ok(Home {
                    root: PathBuf::from(d).join("Jax"),
                });
            }
        }
        let home = env::var_os("HOME")
            .or_else(|| env::var_os("USERPROFILE"))
            .ok_or("cannot determine the home directory; set JAX_HOME")?;
        Ok(Home {
            root: PathBuf::from(home).join(".jax"),
        })
    }

    pub fn at(root: impl Into<PathBuf>) -> Home {
        Home { root: root.into() }
    }

    pub fn toolchains_dir(&self) -> PathBuf {
        self.root.join("toolchains")
    }
    pub fn tmp_dir(&self) -> PathBuf {
        self.root.join("tmp")
    }
    pub fn locks_dir(&self) -> PathBuf {
        self.root.join("locks")
    }
    pub fn store_dir(&self) -> PathBuf {
        self.root.join("store")
    }
    pub fn cache_dir(&self) -> PathBuf {
        self.root.join("cache")
    }
    pub fn bin_dir(&self) -> PathBuf {
        self.root.join("bin")
    }
    pub fn config_path(&self) -> PathBuf {
        self.root.join("config.toml")
    }
    pub fn toolchains(&self) -> Toolchains<'_> {
        Toolchains { home: self }
    }
}

/// Read-only view of the installed toolchains.
pub struct Toolchains<'h> {
    home: &'h Home,
}

impl<'h> Toolchains<'h> {
    pub fn dir(&self, v: &Version) -> PathBuf {
        self.home.toolchains_dir().join(v.to_string())
    }

    pub fn driver_path(&self, v: &Version) -> PathBuf {
        self.dir(v).join("bin").join(exe_name("jax-driver"))
    }

    pub fn is_installed(&self, v: &Version) -> bool {
        self.dir(v).join("toolchain.toml").is_file() && self.driver_path(v).is_file()
    }

    /// Installed versions, oldest first.
    pub fn list(&self) -> Vec<Version> {
        let mut out = Vec::new();
        if let Ok(rd) = fs::read_dir(self.home.toolchains_dir()) {
            for e in rd.flatten() {
                if let Some(name) = e.file_name().to_str() {
                    if let Ok(v) = Version::parse(name) {
                        if self.is_installed(&v) {
                            out.push(v);
                        }
                    }
                }
            }
        }
        out.sort();
        out
    }

    pub fn remove(&self, v: &Version) -> Result<(), String> {
        let dir = self.dir(v);
        if !dir.exists() {
            return Err(format!("toolchain {v} is not installed"));
        }
        fs::remove_dir_all(&dir).map_err(|e| format!("cannot remove {}: {e}", dir.display()))
    }

    /// The directory holding the runtime stub for `target` inside toolchain `v`.
    pub fn runtime_path(&self, v: &Version, target: &str) -> PathBuf {
        self.dir(v).join("runtimes").join(target).join(exe_name("jax-rt"))
    }
}

/// User preferences from `config.toml`. They never influence build output.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Config {
    pub dist_url: Option<String>,
    pub default_toolchain: Option<Version>,
}

impl Config {
    pub fn load(home: &Home) -> Config {
        Config::load_from(&home.config_path())
    }

    pub fn load_from(path: &Path) -> Config {
        let Ok(text) = fs::read_to_string(path) else { return Config::default() };
        let Ok(t) = toml::parse(&text) else { return Config::default() };
        Config {
            dist_url: match t.get("dist_url") {
                Some(Value::Str(s)) => Some(s.clone()),
                _ => None,
            },
            default_toolchain: match t.get("default_toolchain") {
                Some(Value::Str(s)) => Version::parse(s).ok(),
                _ => None,
            },
        }
    }

    pub fn save(&self, home: &Home) -> Result<(), String> {
        let mut out = String::from("# Jax user configuration\n");
        if let Some(u) = &self.dist_url {
            out.push_str(&format!("dist_url = {}\n", toml::quote(u)));
        }
        if let Some(v) = &self.default_toolchain {
            out.push_str(&format!("default_toolchain = {}\n", toml::quote(&v.to_string())));
        }
        fs::create_dir_all(&home.root).map_err(|e| e.to_string())?;
        fs::write(home.config_path(), out).map_err(|e| format!("cannot write config: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_roundtrip() {
        let dir = std::env::temp_dir().join(format!("jax-home-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let home = Home::at(&dir);
        assert_eq!(Config::load(&home), Config::default());
        let c = Config {
            dist_url: Some("file:///x".into()),
            default_toolchain: Some(Version::new(0, 1, 0)),
        };
        c.save(&home).unwrap();
        assert_eq!(Config::load(&home), c);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn host_is_a_known_target() {
        assert!(jax_manifest::TARGETS.contains(&host_target()));
    }
}
