//! Automatic toolchain installation: download, verify, unpack, atomically publish.

use std::fs;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, SystemTime};

use jax_manifest::toml::{self, Value};
use jax_manifest::Version;

use crate::fetch::{fetch_bytes, fetch_text};
use crate::home::{exe_name, host_target, Config, Home};
use crate::sha256::hex_digest;
use crate::tar;

/// Where toolchains are downloaded from.
#[derive(Clone, Debug)]
pub struct Dist {
    pub base: String,
    pub offline: bool,
}

impl Dist {
    /// `JAX_DIST_URL`, else `config.toml`, else the default; `JAX_OFFLINE=1` forbids downloads.
    pub fn resolve(home: &Home) -> Dist {
        let cfg = Config::load(home);
        let base = std::env::var("JAX_DIST_URL")
            .ok()
            .filter(|s| !s.is_empty())
            .or(cfg.dist_url)
            .unwrap_or_else(|| crate::home::DEFAULT_DIST_URL.to_string());
        let offline = matches!(std::env::var("JAX_OFFLINE").as_deref(), Ok("1") | Ok("true"));
        Dist {
            base: base.trim_end_matches('/').to_string(),
            offline,
        }
    }

    fn url(&self, path: &str) -> String {
        format!("{}/{}", self.base, path)
    }
}

/// Text of a release manifest (`<dist>/toolchains/<version>/manifest.toml`).
/// `artifacts` is `(target, file name, sha256)`.
pub fn manifest_text(version: &Version, artifacts: &[(String, String, String)]) -> String {
    let mut out = format!("version = {}\n", toml::quote(&version.to_string()));
    for (target, file, sha) in artifacts {
        out.push_str(&format!("\n[artifacts.{}]\n", toml::key(target)));
        out.push_str(&format!("file = {}\n", toml::quote(file)));
        out.push_str(&format!("sha256 = {}\n", toml::quote(sha)));
    }
    out
}

/// The newest version the distribution advertises (`<dist>/latest.txt`).
pub fn latest_version(dist: &Dist) -> Result<Version, String> {
    if dist.offline {
        return Err("cannot look up the latest toolchain while JAX_OFFLINE is set".into());
    }
    let text = fetch_text(&dist.url("latest.txt"))?;
    Version::parse(text.trim())
}

struct FileLock {
    path: PathBuf,
}

impl FileLock {
    fn acquire(path: PathBuf) -> Result<FileLock, String> {
        if let Some(p) = path.parent() {
            fs::create_dir_all(p).map_err(|e| e.to_string())?;
        }
        for _ in 0..600 {
            match fs::OpenOptions::new().write(true).create_new(true).open(&path) {
                Ok(_) => return Ok(FileLock { path }),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    // Steal locks left behind by a crashed process.
                    if let Ok(meta) = fs::metadata(&path) {
                        if let Ok(age) = SystemTime::now().duration_since(meta.modified().unwrap_or(SystemTime::now())) {
                            if age > Duration::from_secs(900) {
                                let _ = fs::remove_file(&path);
                                continue;
                            }
                        }
                    }
                    thread::sleep(Duration::from_millis(200));
                }
                Err(e) => return Err(format!("cannot create lock {}: {e}", path.display())),
            }
        }
        Err(format!("timed out waiting for {}", path.display()))
    }
}

impl Drop for FileLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

/// Make sure toolchain `version` is installed, downloading it if necessary.
/// Returns the path of its driver executable. `progress` receives status lines.
pub fn ensure_installed(
    home: &Home,
    dist: &Dist,
    version: &Version,
    progress: &mut dyn FnMut(&str),
) -> Result<PathBuf, String> {
    let tcs = home.toolchains();
    if tcs.is_installed(version) {
        return Ok(tcs.driver_path(version));
    }
    if dist.offline {
        return Err(format!(
            "Jax {version} is required but not installed, and JAX_OFFLINE is set.\n  Install it first with: jax toolchain install {version}"
        ));
    }
    let _lock = FileLock::acquire(home.locks_dir().join(format!("toolchain-{version}.lock")))?;
    if tcs.is_installed(version) {
        // Another process finished the job while we waited.
        return Ok(tcs.driver_path(version));
    }
    let target = host_target();
    progress(&format!("Jax {version} is not installed; downloading it for {target}..."));

    let manifest_url = dist.url(&format!("toolchains/{version}/manifest.toml"));
    let manifest_src = fetch_text(&manifest_url).map_err(|e| {
        format!("cannot find Jax {version} at {}: {e}", dist.base)
    })?;
    let manifest = toml::parse(&manifest_src).map_err(|e| format!("bad release manifest: {e}"))?;
    let artifacts = match manifest.get("artifacts") {
        Some(Value::Table(t)) => t,
        _ => return Err("release manifest has no [artifacts]".into()),
    };
    let artifact = match artifacts.get(target) {
        Some(Value::Table(t)) => t,
        _ => {
            let have: Vec<&str> = artifacts.keys().map(|s| s.as_str()).collect();
            return Err(format!(
                "Jax {version} is not available for {target}; it is published for: {}",
                have.join(", ")
            ));
        }
    };
    let (file, sha) = match (artifact.get("file"), artifact.get("sha256")) {
        (Some(Value::Str(f)), Some(Value::Str(s))) => (f.clone(), s.to_lowercase()),
        _ => return Err("release manifest entry needs `file` and `sha256`".into()),
    };
    if file.contains('/') || file.contains('\\') || file.contains("..") {
        return Err(format!("suspicious artifact file name in release manifest: {file}"));
    }

    let archive = fetch_bytes(&dist.url(&format!("toolchains/{version}/{file}")))?;
    let actual = hex_digest(&archive);
    if actual != sha {
        return Err(format!(
            "checksum mismatch for {file}\n  expected {sha}\n  actual   {actual}\nThe download was corrupted or tampered with; nothing was installed."
        ));
    }
    progress(&format!("verified {file} (sha256 {}...)", &sha[..12]));

    let tmp_root = home.tmp_dir();
    fs::create_dir_all(&tmp_root).map_err(|e| e.to_string())?;
    let staging = tmp_root.join(format!("{version}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&staging);
    tar::unpack(&archive, &staging)?;
    let driver = staging.join("bin").join(exe_name("jax-driver"));
    if !driver.is_file() {
        let _ = fs::remove_dir_all(&staging);
        return Err("the downloaded toolchain does not contain bin/jax-driver".into());
    }
    let info = format!(
        "version = {}\ntarget = {}\narchive_sha256 = {}\n",
        toml::quote(&version.to_string()),
        toml::quote(target),
        toml::quote(&actual)
    );
    fs::write(staging.join("toolchain.toml"), info).map_err(|e| e.to_string())?;

    let final_dir = tcs.dir(version);
    fs::create_dir_all(home.toolchains_dir()).map_err(|e| e.to_string())?;
    if let Err(e) = fs::rename(&staging, &final_dir) {
        let _ = fs::remove_dir_all(&staging);
        if !tcs.is_installed(version) {
            return Err(format!("cannot install toolchain into {}: {e}", final_dir.display()));
        }
    }
    progress(&format!("installed Jax {version} into {}", final_dir.display()));
    Ok(tcs.driver_path(version))
}

/// Used by tests and `xtask`: path of the release directory for a version.
pub fn release_dir(dist_root: &Path, version: &Version) -> PathBuf {
    dist_root.join("toolchains").join(version.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tar::{write_tar, Entry};

    fn make_dist(root: &Path, version: &Version, tamper: bool) {
        let tar = write_tar(&[
            Entry {
                path: format!("bin/{}", exe_name("jax-driver")),
                data: b"fake driver".to_vec(),
                mode: 0o755,
            },
            Entry {
                path: "runtimes/readme".into(),
                data: b"hi".to_vec(),
                mode: 0o644,
            },
        ])
        .unwrap();
        let sha = hex_digest(&tar);
        let dir = release_dir(root, version);
        fs::create_dir_all(&dir).unwrap();
        let file = format!("jax-toolchain-{version}-{}.tar", host_target());
        fs::write(dir.join(&file), if tamper { b"garbage".to_vec() } else { tar }).unwrap();
        fs::write(
            dir.join("manifest.toml"),
            manifest_text(version, &[(host_target().to_string(), file, sha)]),
        )
        .unwrap();
        fs::write(root.join("latest.txt"), format!("{version}\n")).unwrap();
    }

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("jax-inst-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn installs_once_and_reuses() {
        let root = tmp("ok");
        let v = Version::new(9, 9, 9);
        make_dist(&root.join("dist"), &v, false);
        let home = Home::at(root.join("home"));
        let dist = Dist {
            base: format!("file://{}", root.join("dist").display()),
            offline: false,
        };
        let mut msgs = Vec::new();
        let driver = ensure_installed(&home, &dist, &v, &mut |m| msgs.push(m.to_string())).unwrap();
        assert!(driver.is_file());
        assert!(msgs.iter().any(|m| m.contains("downloading")));
        assert_eq!(home.toolchains().list(), vec![v.clone()]);
        // Second call is a no-op and prints nothing.
        let mut msgs2 = Vec::new();
        ensure_installed(&home, &dist, &v, &mut |m| msgs2.push(m.to_string())).unwrap();
        assert!(msgs2.is_empty());
        assert_eq!(latest_version(&dist).unwrap(), v);
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn rejects_bad_checksum() {
        let root = tmp("bad");
        let v = Version::new(9, 9, 8);
        make_dist(&root.join("dist"), &v, true);
        let home = Home::at(root.join("home"));
        let dist = Dist {
            base: format!("file://{}", root.join("dist").display()),
            offline: false,
        };
        let err = ensure_installed(&home, &dist, &v, &mut |_| {}).unwrap_err();
        assert!(err.contains("checksum mismatch"), "{err}");
        assert!(home.toolchains().list().is_empty());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn offline_mode_refuses() {
        let root = tmp("off");
        let home = Home::at(root.join("home"));
        let dist = Dist {
            base: "file:///nowhere".into(),
            offline: true,
        };
        let err = ensure_installed(&home, &dist, &Version::new(1, 2, 3), &mut |_| {}).unwrap_err();
        assert!(err.contains("JAX_OFFLINE"), "{err}");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn missing_release_is_reported() {
        let root = tmp("miss");
        fs::create_dir_all(root.join("dist")).unwrap();
        let home = Home::at(root.join("home"));
        let dist = Dist {
            base: format!("file://{}", root.join("dist").display()),
            offline: false,
        };
        let err = ensure_installed(&home, &dist, &Version::new(1, 2, 3), &mut |_| {}).unwrap_err();
        assert!(err.contains("cannot find Jax 1.2.3"), "{err}");
        let _ = fs::remove_dir_all(&root);
    }
}
