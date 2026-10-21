//! The `jax` launcher.
//!
//! A tiny, dependency-free program whose only job is to find the exact
//! toolchain a project pins (`jax = "x.y.z"` in `jax.toml`), download it if
//! it is missing, and hand control to it. All real behavior lives in the
//! toolchain, so a project always behaves like the Jax version it declares.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{exit, Command};

use jax_manifest::{find_manifest, Manifest, Version};
use jax_toolchain::install::Dist;
use jax_toolchain::{ensure_installed, host_target, latest_version, Config, Home};

const LAUNCHER_VERSION: &str = env!("CARGO_PKG_VERSION");

fn main() {
    match run() {
        Ok(code) => exit(code),
        Err(e) => {
            eprintln!("error: {e}");
            exit(1);
        }
    }
}

fn say(msg: &str) {
    eprintln!("jax: {msg}");
}

#[derive(Debug)]
enum Why {
    Flag,
    Env,
    Project(PathBuf),
    FileHeader(PathBuf),
    ConfigDefault,
    NewestInstalled,
    NewestAvailable,
}

fn run() -> Result<i32, String> {
    let mut args: Vec<String> = env::args().skip(1).collect();
    let home = Home::discover()?;

    // `jax +0.1.0 build` selects a toolchain explicitly.
    let mut flag: Option<Version> = None;
    if let Some(first) = args.first() {
        if let Some(v) = first.strip_prefix('+') {
            flag = Some(Version::parse(v).map_err(|e| format!("bad toolchain selector `{first}`: {e}"))?);
            args.remove(0);
        }
    }

    match args.first().map(|s| s.as_str()) {
        Some("--launcher-version") => {
            println!("jax launcher {LAUNCHER_VERSION} ({})", host_target());
            return Ok(0);
        }
        Some("toolchain") => return toolchain_command(&home, &args[1..]),
        _ => {}
    }

    let dist = Dist::resolve(&home);
    let (version, why) = resolve_version(&home, &dist, &args, flag)?;
    let driver = ensure_installed(&home, &dist, &version, &mut |m| say(m))?;
    if env::var_os("JAX_VERBOSE").is_some() {
        say(&format!("using toolchain {version} ({why:?})"));
    }
    exec_driver(&driver, &args, &home, &version)
}

fn resolve_version(home: &Home, dist: &Dist, args: &[String], flag: Option<Version>) -> Result<(Version, Why), String> {
    if let Some(v) = flag {
        return Ok((v, Why::Flag));
    }
    if let Ok(v) = env::var("JAX_TOOLCHAIN") {
        if !v.is_empty() {
            let v = Version::parse(&v).map_err(|e| format!("JAX_TOOLCHAIN: {e}"))?;
            return Ok((v, Why::Env));
        }
    }
    let cwd = env::current_dir().map_err(|e| e.to_string())?;
    if let Some(path) = find_manifest(&cwd) {
        let m = Manifest::load(&path).map_err(|e| e.to_string())?;
        return Ok((m.project.jax, Why::Project(path)));
    }
    for a in args {
        if a.ends_with(".jax") {
            if let Some(v) = file_header_version(Path::new(a)) {
                return Ok((v, Why::FileHeader(PathBuf::from(a))));
            }
        }
    }
    let cfg = Config::load(home);
    if let Some(v) = cfg.default_toolchain.clone() {
        return Ok((v, Why::ConfigDefault));
    }
    if let Some(v) = home.toolchains().list().pop() {
        return Ok((v, Why::NewestInstalled));
    }
    say("no toolchain installed yet; looking up the latest release...");
    let latest = latest_version(dist)?;
    let mut cfg = cfg;
    cfg.default_toolchain = Some(latest.clone());
    let _ = cfg.save(home);
    Ok((latest, Why::NewestAvailable))
}

/// Single-file programs may pin a version with `//! jax 0.1.0` in their first lines.
fn file_header_version(path: &Path) -> Option<Version> {
    let text = fs::read_to_string(path).ok()?;
    for line in text.lines().take(10) {
        if let Some(rest) = line.trim().strip_prefix("//! jax ") {
            return Version::parse(rest.trim()).ok();
        }
    }
    None
}

fn exec_driver(driver: &Path, args: &[String], home: &Home, version: &Version) -> Result<i32, String> {
    let mut cmd = Command::new(driver);
    cmd.args(args)
        .env("JAX_LAUNCHER", "1")
        .env("JAX_HOME", &home.root)
        .env("JAX_TOOLCHAIN_VERSION", version.to_string());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        let err = cmd.exec();
        Err(format!("cannot run {}: {err}", driver.display()))
    }
    #[cfg(not(unix))]
    {
        let status = cmd
            .status()
            .map_err(|e| format!("cannot run {}: {e}", driver.display()))?;
        Ok(status.code().unwrap_or(1))
    }
}

// ----- `jax toolchain ...` ---------------------------------------------------

fn toolchain_help() {
    println!(
        "jax toolchain: manage installed Jax toolchains

    jax toolchain list                    Installed versions (* = default)
    jax toolchain install <version>       Download and install a version
    jax toolchain remove <version>        Delete an installed version
    jax toolchain default <version>       Version used outside of projects
    jax toolchain which [version]         Path of the driver that would run here
    jax toolchain latest                  Newest version the distribution offers
    jax toolchain prune [--yes]           Remove versions nothing needs
    jax toolchain path                    The Jax home directory

Environment: JAX_HOME, JAX_DIST_URL (http(s):// or file://), JAX_OFFLINE=1, JAX_TOOLCHAIN"
    );
}

fn parse_version_arg(args: &[String], what: &str) -> Result<Version, String> {
    let s = args.first().ok_or_else(|| format!("usage: jax toolchain {what} <version>"))?;
    Version::parse(s)
}

fn toolchain_command(home: &Home, args: &[String]) -> Result<i32, String> {
    let dist = Dist::resolve(home);
    match args.first().map(|s| s.as_str()) {
        Some("list") => {
            let cfg = Config::load(home);
            let installed = home.toolchains().list();
            if installed.is_empty() {
                println!("no toolchains installed (home: {})", home.root.display());
            }
            for v in installed {
                let mark = if cfg.default_toolchain.as_ref() == Some(&v) { "*" } else { " " };
                println!("{mark} {v}");
            }
            Ok(0)
        }
        Some("install") => {
            let v = parse_version_arg(&args[1..], "install")?;
            let driver = ensure_installed(home, &dist, &v, &mut |m| say(m))?;
            println!("Jax {v} is installed ({})", driver.display());
            if args.iter().any(|a| a == "--set-default") {
                let mut cfg = Config::load(home);
                cfg.default_toolchain = Some(v);
                cfg.save(home)?;
            }
            Ok(0)
        }
        Some("remove") => {
            let v = parse_version_arg(&args[1..], "remove")?;
            home.toolchains().remove(&v)?;
            println!("removed Jax {v}");
            Ok(0)
        }
        Some("default") => {
            let v = parse_version_arg(&args[1..], "default")?;
            let mut cfg = Config::load(home);
            cfg.default_toolchain = Some(v.clone());
            cfg.save(home)?;
            println!("default toolchain is now {v}");
            Ok(0)
        }
        Some("which") => {
            let v = match args.get(1) {
                Some(s) => Version::parse(s)?,
                None => resolve_version(home, &dist, &[], None)?.0,
            };
            println!("{}", home.toolchains().driver_path(&v).display());
            Ok(0)
        }
        Some("latest") => {
            println!("{}", latest_version(&dist)?);
            Ok(0)
        }
        Some("path") => {
            println!("{}", home.root.display());
            Ok(0)
        }
        Some("prune") => {
            let yes = args.iter().any(|a| a == "--yes");
            let cfg = Config::load(home);
            let mut keep: Vec<Version> = cfg.default_toolchain.into_iter().collect();
            if let Ok(cwd) = env::current_dir() {
                if let Some(p) = find_manifest(&cwd) {
                    if let Ok(m) = Manifest::load(&p) {
                        keep.push(m.project.jax);
                    }
                }
            }
            if let Some(i) = args.iter().position(|a| a == "--keep") {
                for s in args.get(i + 1).map(|s| s.split(',')).into_iter().flatten() {
                    keep.push(Version::parse(s)?);
                }
            }
            let mut removed = 0;
            for v in home.toolchains().list() {
                if keep.contains(&v) {
                    continue;
                }
                if yes {
                    home.toolchains().remove(&v)?;
                    println!("removed Jax {v}");
                } else {
                    println!("would remove Jax {v}");
                }
                removed += 1;
            }
            if removed == 0 {
                println!("nothing to prune");
            } else if !yes {
                println!("re-run with --yes to delete");
            }
            Ok(0)
        }
        Some("help") | Some("--help") | Some("-h") | None => {
            toolchain_help();
            Ok(0)
        }
        Some(other) => Err(format!("unknown `jax toolchain` command `{other}`; run `jax toolchain help`")),
    }
}
