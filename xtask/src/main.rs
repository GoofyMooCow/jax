//! Repository automation.
//!
//! * `cargo xtask dist [--out DIR]`  build this machine's release artifacts
//! * `cargo xtask merge --out DIR IN...`  combine dist directories from several machines
//!
//! A dist directory is what gets published (as static files) at `<site>/dist`:
//!
//! ```text
//! latest.txt
//! toolchains/<version>/manifest.toml                     sha256 per platform
//! toolchains/<version>/jax-toolchain-<version>-<target>.tar
//! bin/<target>/jax[.exe]  and  jax[.exe].sha256          the launcher
//! ```

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{exit, Command};

use jax_manifest::toml::{self, Value};
use jax_manifest::Version;
use jax_toolchain::sha256::hex_digest;
use jax_toolchain::tar::{write_tar, Entry};
use jax_toolchain::{exe_name, host_target, manifest_text};

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(|s| s.as_str()) {
        Some("dist") => dist(&args[1..]),
        Some("merge") => merge(&args[1..]),
        _ => Err("usage: cargo xtask dist [--out <dir>]\n       cargo xtask merge --out <dir> <dist-dir>...".into()),
    };
    if let Err(e) = result {
        eprintln!("xtask: {e}");
        exit(1);
    }
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}

fn out_dir(args: &[String], default: PathBuf) -> Result<PathBuf, String> {
    match args.iter().position(|a| a == "--out") {
        Some(i) => Ok(PathBuf::from(args.get(i + 1).ok_or("`--out` needs a directory")?)),
        None => Ok(default),
    }
}

type Artifacts = BTreeMap<String, (String, String)>; // target -> (file, sha256)

fn read_artifacts(manifest_path: &Path) -> Artifacts {
    let mut out = Artifacts::new();
    if let Ok(text) = fs::read_to_string(manifest_path) {
        if let Ok(t) = toml::parse(&text) {
            if let Some(Value::Table(a)) = t.get("artifacts") {
                for (tg, v) in a {
                    if let Value::Table(e) = v {
                        if let (Some(Value::Str(f)), Some(Value::Str(s))) = (e.get("file"), e.get("sha256")) {
                            out.insert(tg.clone(), (f.clone(), s.clone()));
                        }
                    }
                }
            }
        }
    }
    out
}

fn write_manifest(path: &Path, version: &Version, artifacts: &Artifacts) -> Result<(), String> {
    let list: Vec<(String, String, String)> = artifacts
        .iter()
        .map(|(t, (f, s))| (t.clone(), f.clone(), s.clone()))
        .collect();
    fs::write(path, manifest_text(version, &list)).map_err(|e| e.to_string())
}

fn dist(args: &[String]) -> Result<(), String> {
    let root = workspace_root();
    let out = out_dir(args, root.join("dist"))?;

    println!("xtask: building release binaries...");
    let status = Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".into()))
        .current_dir(&root)
        .args(["build", "--release", "-p", "jax-launcher", "-p", "jax-driver", "-p", "jax-rt"])
        .status()
        .map_err(|e| format!("cannot run cargo: {e}"))?;
    if !status.success() {
        return Err("cargo build failed".into());
    }

    let release = root.join("target").join("release");
    let read = |name: &str| fs::read(release.join(exe_name(name))).map_err(|e| format!("cannot read {name}: {e}"));
    let driver = read("jax-driver")?;
    let rt = read("jax-rt")?;
    let launcher = read("jax")?;
    let license = fs::read(root.join("LICENSE")).unwrap_or_default();

    let target = host_target();
    let version = Version::parse(VERSION)?;
    let tar = write_tar(&[
        Entry {
            path: format!("bin/{}", exe_name("jax-driver")),
            data: driver,
            mode: 0o755,
        },
        Entry {
            path: format!("runtimes/{target}/{}", exe_name("jax-rt")),
            data: rt,
            mode: 0o755,
        },
        Entry {
            path: "LICENSE".into(),
            data: license,
            mode: 0o644,
        },
    ])?;
    let sha = hex_digest(&tar);

    let rel_dir = out.join("toolchains").join(VERSION);
    fs::create_dir_all(&rel_dir).map_err(|e| e.to_string())?;
    let file = format!("jax-toolchain-{VERSION}-{target}.tar");
    fs::write(rel_dir.join(&file), &tar).map_err(|e| e.to_string())?;

    let manifest_path = rel_dir.join("manifest.toml");
    let mut artifacts = read_artifacts(&manifest_path);
    artifacts.insert(target.to_string(), (file.clone(), sha.clone()));
    write_manifest(&manifest_path, &version, &artifacts)?;
    fs::write(out.join("latest.txt"), format!("{VERSION}\n")).map_err(|e| e.to_string())?;

    let bin_dir = out.join("bin").join(target);
    fs::create_dir_all(&bin_dir).map_err(|e| e.to_string())?;
    let name = exe_name("jax");
    let launcher_path = bin_dir.join(&name);
    fs::write(&launcher_path, &launcher).map_err(|e| e.to_string())?;
    fs::write(bin_dir.join(format!("{name}.sha256")), format!("{}  {name}\n", hex_digest(&launcher)))
        .map_err(|e| e.to_string())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(&launcher_path, fs::Permissions::from_mode(0o755));
    }

    println!("xtask: wrote {}", out.display());
    println!("       toolchain {file}  sha256 {sha}");
    println!("       launcher  {}", launcher_path.display());
    Ok(())
}

fn collect(dir: &Path, base: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    for e in fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))? {
        let p = e.map_err(|e| e.to_string())?.path();
        if p.is_dir() {
            collect(&p, base, out)?;
        } else {
            out.push(p.strip_prefix(base).unwrap().to_path_buf());
        }
    }
    Ok(())
}

/// Combine dist directories built on different machines into one.
fn merge(args: &[String]) -> Result<(), String> {
    let out = out_dir(args, PathBuf::from("dist-merged"))?;
    let mut inputs: Vec<PathBuf> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "--out" {
            i += 2;
            continue;
        }
        inputs.push(PathBuf::from(&args[i]));
        i += 1;
    }
    if inputs.is_empty() {
        return Err("no input directories given".into());
    }
    let mut manifests: BTreeMap<String, Artifacts> = BTreeMap::new();
    let mut newest: Option<Version> = None;
    for input in &inputs {
        let mut files = Vec::new();
        collect(input, input, &mut files)?;
        for rel in files {
            let src = input.join(&rel);
            let is_manifest = rel.file_name().map(|n| n == "manifest.toml").unwrap_or(false);
            if is_manifest {
                let version = rel
                    .parent()
                    .and_then(|p| p.file_name())
                    .map(|s| s.to_string_lossy().to_string())
                    .ok_or("manifest outside a version directory")?;
                let entry = manifests.entry(version).or_default();
                for (t, v) in read_artifacts(&src) {
                    entry.insert(t, v);
                }
                continue;
            }
            if rel.file_name().map(|n| n == "latest.txt").unwrap_or(false) {
                if let Ok(v) = Version::parse(fs::read_to_string(&src).unwrap_or_default().trim()) {
                    if newest.as_ref().map(|n| v > *n).unwrap_or(true) {
                        newest = Some(v);
                    }
                }
                continue;
            }
            let dst = out.join(&rel);
            fs::create_dir_all(dst.parent().unwrap()).map_err(|e| e.to_string())?;
            fs::copy(&src, &dst).map_err(|e| format!("copy {}: {e}", src.display()))?;
        }
    }
    for (version, artifacts) in &manifests {
        let v = Version::parse(version)?;
        let dir = out.join("toolchains").join(version);
        fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
        write_manifest(&dir.join("manifest.toml"), &v, artifacts)?;
        println!("xtask: {version}: {}", artifacts.keys().cloned().collect::<Vec<_>>().join(", "));
    }
    if let Some(v) = newest {
        fs::write(out.join("latest.txt"), format!("{v}\n")).map_err(|e| e.to_string())?;
    }
    println!("xtask: merged {} director{} into {}", inputs.len(), if inputs.len() == 1 { "y" } else { "ies" }, out.display());
    Ok(())
}
