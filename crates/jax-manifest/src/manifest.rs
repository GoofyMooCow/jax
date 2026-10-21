//! The `jax.toml` project manifest.

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};

use crate::toml::{self, Table, Value};
use crate::version::Version;

pub const MANIFEST_FILE: &str = "jax.toml";
pub const LOCK_FILE: &str = "jax.lock";

/// Platforms Jax can target.
pub const TARGETS: [&str; 6] = [
    "windows-x64",
    "windows-arm64",
    "linux-x64",
    "linux-arm64",
    "macos-x64",
    "macos-arm64",
];

#[derive(Debug, Clone, PartialEq)]
pub struct ManifestError {
    pub path: Option<PathBuf>,
    pub message: String,
}

impl fmt::Display for ManifestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.path {
            Some(p) => write!(f, "{}: {}", p.display(), self.message),
            None => write!(f, "{}", self.message),
        }
    }
}

impl std::error::Error for ManifestError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    App,
    Lib,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Optimize {
    None,
    Speed,
    Size,
    Balanced,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapsuleLevel {
    Lean,
    Vendored,
    Sealed,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DepSource {
    Registry { version: Version, registry: Option<String> },
    Path(String),
    Git { url: String, rev: String },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Dependency {
    pub source: DepSource,
    pub features: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Project {
    pub name: String,
    pub version: Version,
    pub kind: Kind,
    /// The exact toolchain version this project builds with.
    pub jax: Version,
    pub edition: Option<String>,
    pub license: Option<String>,
    pub description: Option<String>,
    pub authors: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Build {
    pub targets: Vec<String>,
    pub vendor: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Profile {
    pub optimize: Optimize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Capsule {
    pub level: CapsuleLevel,
    pub platforms: Vec<String>,
}

impl Default for Capsule {
    fn default() -> Capsule {
        Capsule {
            level: CapsuleLevel::Lean,
            platforms: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Manifest {
    pub project: Project,
    pub dependencies: BTreeMap<String, Dependency>,
    pub dev_dependencies: BTreeMap<String, Dependency>,
    pub tools: BTreeMap<String, Version>,
    pub build: Build,
    pub profiles: BTreeMap<String, Profile>,
    pub tasks: BTreeMap<String, String>,
    pub capsule: Capsule,
    pub registries: BTreeMap<String, String>,
    pub workspace_members: Vec<String>,
}

impl Manifest {
    pub fn parse(src: &str) -> Result<Manifest, ManifestError> {
        let root = toml::parse(src).map_err(|e| ManifestError {
            path: None,
            message: format!("invalid TOML at {e}"),
        })?;
        from_table(&root).map_err(|message| ManifestError { path: None, message })
    }

    pub fn load(path: &Path) -> Result<Manifest, ManifestError> {
        let src = std::fs::read_to_string(path).map_err(|e| ManifestError {
            path: Some(path.to_path_buf()),
            message: format!("cannot read manifest: {e}"),
        })?;
        Manifest::parse(&src).map_err(|mut e| {
            e.path = Some(path.to_path_buf());
            e
        })
    }

    /// The source entry point of this project, relative to its root.
    pub fn entry_point(&self) -> &'static str {
        match self.project.kind {
            Kind::App => "src/main.jax",
            Kind::Lib => "src/lib.jax",
        }
    }
}

/// Walk up from `start` looking for `jax.toml`.
pub fn find_manifest(start: &Path) -> Option<PathBuf> {
    let mut dir = Some(start);
    while let Some(d) = dir {
        let candidate = d.join(MANIFEST_FILE);
        if candidate.is_file() {
            return Some(candidate);
        }
        dir = d.parent();
    }
    None
}

/// The text of a fresh `jax.toml` for `jax new`.
pub fn scaffold(name: &str, kind: Kind, jax: &Version) -> String {
    format!(
        "[project]\nname = {}\nversion = \"0.1.0\"\nkind = \"{}\"\njax = {}   # exact toolchain version; ranges are not allowed\n\n[dependencies]\n",
        toml::quote(name),
        if kind == Kind::App { "app" } else { "lib" },
        toml::quote(&jax.to_string()),
    )
}

// ----- extraction helpers ---------------------------------------------------

type R<T> = Result<T, String>;

fn as_table<'a>(v: &'a Value, path: &str) -> R<&'a Table> {
    match v {
        Value::Table(t) => Ok(t),
        other => Err(format!("`{path}` must be a table, found {}", other.type_name())),
    }
}

fn check_keys(t: &Table, allowed: &[&str], path: &str) -> R<()> {
    for k in t.keys() {
        if !allowed.contains(&k.as_str()) {
            let hint = jax_syntax_suggest(k, allowed);
            let loc = if path.is_empty() { "the manifest".to_string() } else { format!("[{path}]") };
            return Err(match hint {
                Some(h) => format!("unknown key `{k}` in {loc}; did you mean `{h}`?"),
                None => format!("unknown key `{k}` in {loc}; allowed keys: {}", allowed.join(", ")),
            });
        }
    }
    Ok(())
}

fn jax_syntax_suggest(name: &str, candidates: &[&str]) -> Option<String> {
    // Tiny edit-distance suggestion (kept local to avoid depending on jax-syntax).
    fn dist(a: &str, b: &str) -> usize {
        let a: Vec<char> = a.chars().collect();
        let b: Vec<char> = b.chars().collect();
        let mut prev: Vec<usize> = (0..=b.len()).collect();
        for i in 1..=a.len() {
            let mut cur = vec![i];
            for j in 1..=b.len() {
                let c = if a[i - 1] == b[j - 1] { 0 } else { 1 };
                cur.push((prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + c));
            }
            prev = cur;
        }
        prev[b.len()]
    }
    candidates
        .iter()
        .map(|c| (dist(name, c), *c))
        .filter(|(d, _)| *d > 0 && *d <= 2)
        .min_by_key(|(d, _)| *d)
        .map(|(_, c)| c.to_string())
}

fn req_str(t: &Table, key: &str, path: &str) -> R<String> {
    match t.get(key) {
        Some(Value::Str(s)) => Ok(s.clone()),
        Some(other) => Err(format!("`{path}.{key}` must be a string, found {}", other.type_name())),
        None => Err(format!("missing required key `{key}` in [{path}]")),
    }
}

fn opt_str(t: &Table, key: &str, path: &str) -> R<Option<String>> {
    match t.get(key) {
        Some(Value::Str(s)) => Ok(Some(s.clone())),
        Some(other) => Err(format!("`{path}.{key}` must be a string, found {}", other.type_name())),
        None => Ok(None),
    }
}

fn opt_bool(t: &Table, key: &str, path: &str) -> R<Option<bool>> {
    match t.get(key) {
        Some(Value::Bool(b)) => Ok(Some(*b)),
        Some(other) => Err(format!("`{path}.{key}` must be a boolean, found {}", other.type_name())),
        None => Ok(None),
    }
}

fn str_array(t: &Table, key: &str, path: &str) -> R<Vec<String>> {
    match t.get(key) {
        None => Ok(Vec::new()),
        Some(Value::Array(a)) => a
            .iter()
            .map(|v| match v {
                Value::Str(s) => Ok(s.clone()),
                other => Err(format!("`{path}.{key}` must contain only strings, found {}", other.type_name())),
            })
            .collect(),
        Some(other) => Err(format!("`{path}.{key}` must be an array, found {}", other.type_name())),
    }
}

fn parse_version(s: &str, what: &str) -> R<Version> {
    Version::parse(s).map_err(|e| format!("{what}: {e}"))
}

fn validate_name(name: &str) -> R<()> {
    let ok = !name.is_empty()
        && name.len() <= 64
        && name.chars().next().map(|c| c.is_ascii_lowercase()).unwrap_or(false)
        && name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_');
    if ok {
        Ok(())
    } else {
        Err(format!(
            "`project.name` is invalid: `{name}`; use lowercase letters, digits, `-` and `_`, starting with a letter (max 64 characters)"
        ))
    }
}

fn validate_targets(list: &[String], path: &str) -> R<()> {
    for t in list {
        if !TARGETS.contains(&t.as_str()) {
            return Err(format!("`{path}`: unknown target `{t}`; known targets: {}", TARGETS.join(", ")));
        }
    }
    Ok(())
}

fn parse_dependency(name: &str, v: &Value, section: &str) -> R<Dependency> {
    let path = format!("{section}.{name}");
    match v {
        Value::Str(s) => Ok(Dependency {
            source: DepSource::Registry {
                version: parse_version(s, &format!("`{path}`"))?,
                registry: None,
            },
            features: Vec::new(),
        }),
        Value::Table(t) => {
            check_keys(t, &["version", "path", "git", "rev", "registry", "features"], &path)?;
            let features = str_array(t, "features", &path)?;
            let version = opt_str(t, "version", &path)?;
            let local = opt_str(t, "path", &path)?;
            let git = opt_str(t, "git", &path)?;
            let rev = opt_str(t, "rev", &path)?;
            let registry = opt_str(t, "registry", &path)?;
            let sources = [version.is_some(), local.is_some(), git.is_some()]
                .iter()
                .filter(|b| **b)
                .count();
            if sources != 1 {
                return Err(format!(
                    "dependency `{name}` must specify exactly one of `version`, `path` or `git`"
                ));
            }
            let source = if let Some(v) = version {
                DepSource::Registry {
                    version: parse_version(&v, &format!("`{path}.version`"))?,
                    registry,
                }
            } else if let Some(p) = local {
                DepSource::Path(p)
            } else {
                let url = git.unwrap();
                let rev = rev.ok_or_else(|| {
                    format!("git dependency `{name}` needs `rev` (a commit hash); branches and tags are not reproducible")
                })?;
                DepSource::Git { url, rev }
            };
            Ok(Dependency { source, features })
        }
        other => Err(format!("`{path}` must be a version string or a table, found {}", other.type_name())),
    }
}

fn parse_deps(root: &Table, section: &str) -> R<BTreeMap<String, Dependency>> {
    let mut out = BTreeMap::new();
    if let Some(v) = root.get(section) {
        for (name, dv) in as_table(v, section)? {
            out.insert(name.clone(), parse_dependency(name, dv, section)?);
        }
    }
    Ok(out)
}

fn from_table(root: &Table) -> R<Manifest> {
    check_keys(
        root,
        &[
            "project",
            "dependencies",
            "dev-dependencies",
            "tools",
            "build",
            "profile",
            "tasks",
            "capsule",
            "registries",
            "workspace",
        ],
        "",
    )?;

    // [project]
    let pv = root
        .get("project")
        .ok_or_else(|| "missing required [project] section".to_string())?;
    let pt = as_table(pv, "project")?;
    check_keys(
        pt,
        &["name", "version", "kind", "jax", "edition", "license", "description", "authors"],
        "project",
    )?;
    let name = req_str(pt, "name", "project")?;
    validate_name(&name)?;
    let version = parse_version(&req_str(pt, "version", "project")?, "`project.version`")?;
    let kind = match opt_str(pt, "kind", "project")?.as_deref() {
        None | Some("app") => Kind::App,
        Some("lib") => Kind::Lib,
        Some(other) => return Err(format!("`project.kind` must be \"app\" or \"lib\", found \"{other}\"")),
    };
    let jax = match pt.get("jax") {
        None => {
            return Err("missing required key `jax` in [project]; every project pins the exact toolchain it builds with, e.g. jax = \"0.1.0\"".into())
        }
        Some(_) => parse_version(&req_str(pt, "jax", "project")?, "`project.jax`")?,
    };
    let edition = opt_str(pt, "edition", "project")?;
    if let Some(e) = &edition {
        if e.len() != 4 || !e.chars().all(|c| c.is_ascii_digit()) {
            return Err(format!("`project.edition` must be a year like \"2026\", found \"{e}\""));
        }
    }
    let project = Project {
        name,
        version,
        kind,
        jax,
        edition,
        license: opt_str(pt, "license", "project")?,
        description: opt_str(pt, "description", "project")?,
        authors: str_array(pt, "authors", "project")?,
    };

    // [tools]
    let mut tools = BTreeMap::new();
    if let Some(v) = root.get("tools") {
        for (n, tv) in as_table(v, "tools")? {
            match tv {
                Value::Str(s) => {
                    tools.insert(n.clone(), parse_version(s, &format!("`tools.{n}`"))?);
                }
                other => return Err(format!("`tools.{n}` must be a version string, found {}", other.type_name())),
            }
        }
    }

    // [build]
    let mut build = Build::default();
    if let Some(v) = root.get("build") {
        let t = as_table(v, "build")?;
        check_keys(t, &["targets", "vendor"], "build")?;
        build.targets = str_array(t, "targets", "build")?;
        validate_targets(&build.targets, "build.targets")?;
        build.vendor = opt_bool(t, "vendor", "build")?.unwrap_or(false);
    }

    // [profile.*]
    let mut profiles = BTreeMap::new();
    if let Some(v) = root.get("profile") {
        for (n, pv) in as_table(v, "profile")? {
            let path = format!("profile.{n}");
            let t = as_table(pv, &path)?;
            check_keys(t, &["optimize"], &path)?;
            let optimize = match opt_str(t, "optimize", &path)?.as_deref() {
                None | Some("balanced") => Optimize::Balanced,
                Some("speed") => Optimize::Speed,
                Some("size") => Optimize::Size,
                Some("none") => Optimize::None,
                Some(o) => {
                    return Err(format!(
                        "`{path}.optimize` must be \"speed\", \"size\", \"balanced\" or \"none\", found \"{o}\""
                    ))
                }
            };
            profiles.insert(n.clone(), Profile { optimize });
        }
    }

    // [tasks]
    let mut tasks = BTreeMap::new();
    if let Some(v) = root.get("tasks") {
        for (n, tv) in as_table(v, "tasks")? {
            match tv {
                Value::Str(s) => {
                    tasks.insert(n.clone(), s.clone());
                }
                other => return Err(format!("`tasks.{n}` must be a string, found {}", other.type_name())),
            }
        }
    }

    // [capsule]
    let mut capsule = Capsule::default();
    if let Some(v) = root.get("capsule") {
        let t = as_table(v, "capsule")?;
        check_keys(t, &["level", "platforms"], "capsule")?;
        capsule.level = match opt_str(t, "level", "capsule")?.as_deref() {
            None | Some("lean") => CapsuleLevel::Lean,
            Some("vendored") => CapsuleLevel::Vendored,
            Some("sealed") => CapsuleLevel::Sealed,
            Some(o) => {
                return Err(format!(
                    "`capsule.level` must be \"lean\", \"vendored\" or \"sealed\", found \"{o}\""
                ))
            }
        };
        capsule.platforms = str_array(t, "platforms", "capsule")?;
        validate_targets(&capsule.platforms, "capsule.platforms")?;
    }

    // [registries]
    let mut registries = BTreeMap::new();
    if let Some(v) = root.get("registries") {
        for (n, rv) in as_table(v, "registries")? {
            match rv {
                Value::Str(s) => {
                    registries.insert(n.clone(), s.clone());
                }
                other => return Err(format!("`registries.{n}` must be a URL string, found {}", other.type_name())),
            }
        }
    }

    // [workspace]
    let mut workspace_members = Vec::new();
    if let Some(v) = root.get("workspace") {
        let t = as_table(v, "workspace")?;
        check_keys(t, &["members"], "workspace")?;
        workspace_members = str_array(t, "members", "workspace")?;
    }

    Ok(Manifest {
        project,
        dependencies: parse_deps(root, "dependencies")?,
        dev_dependencies: parse_deps(root, "dev-dependencies")?,
        tools,
        build,
        profiles,
        tasks,
        capsule,
        registries,
        workspace_members,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const FULL: &str = r#"
[project]
name = "my-app"
version = "0.3.0"
kind = "app"
jax = "1.4.2"
edition = "2026"
license = "MIT"

[dependencies]
"acme.chart" = "2.1.0"
"web.router" = { version = "1.3.0", features = ["tls"] }
"local.util" = { path = "../util" }
"fork.thing" = { git = "https://example.org/thing", rev = "a1b2c3d" }

[dev-dependencies]
"test.mocks" = "0.2.0"

[tools]
"jax-fuzz" = "0.4.1"

[build]
targets = ["windows-x64", "linux-x64", "macos-arm64"]

[profile.release]
optimize = "speed"

[tasks]
deploy = "jax build"

[capsule]
level = "vendored"
"#;

    #[test]
    fn parses_full_manifest() {
        let m = Manifest::parse(FULL).unwrap();
        assert_eq!(m.project.name, "my-app");
        assert_eq!(m.project.jax.to_string(), "1.4.2");
        assert_eq!(m.dependencies.len(), 4);
        assert_eq!(m.build.targets.len(), 3);
        assert_eq!(m.profiles["release"].optimize, Optimize::Speed);
        assert_eq!(m.capsule.level, CapsuleLevel::Vendored);
        assert_eq!(m.entry_point(), "src/main.jax");
    }

    #[test]
    fn scaffold_roundtrips() {
        let text = scaffold("hello", Kind::App, &Version::new(0, 1, 0));
        let m = Manifest::parse(&text).unwrap();
        assert_eq!(m.project.name, "hello");
        assert_eq!(m.project.jax, Version::new(0, 1, 0));
    }

    #[test]
    fn rejects_version_ranges() {
        let e = Manifest::parse("[project]\nname=\"a\"\nversion=\"1.0.0\"\njax=\"^1.4\"\n").unwrap_err();
        assert!(e.message.contains("not an exact version"), "{e}");
    }

    #[test]
    fn requires_jax_pin() {
        let e = Manifest::parse("[project]\nname=\"a\"\nversion=\"1.0.0\"\n").unwrap_err();
        assert!(e.message.contains("pins the exact toolchain"), "{e}");
    }

    #[test]
    fn typo_suggestion() {
        let e = Manifest::parse("[project]\nname=\"a\"\nversion=\"1.0.0\"\njax=\"1.0.0\"\n[dependancies]\n").unwrap_err();
        assert!(e.message.contains("unknown key"), "{e}");
        let e = Manifest::parse("[project]\nname=\"a\"\nverison=\"1.0.0\"\njax=\"1.0.0\"\n").unwrap_err();
        assert!(e.message.contains("did you mean `version`"), "{e}");
    }

    #[test]
    fn git_needs_rev() {
        let e = Manifest::parse(
            "[project]\nname=\"a\"\nversion=\"1.0.0\"\njax=\"1.0.0\"\n[dependencies]\nx = { git = \"u\" }\n",
        )
        .unwrap_err();
        assert!(e.message.contains("rev"), "{e}");
    }

    #[test]
    fn bad_names_and_targets() {
        assert!(Manifest::parse("[project]\nname=\"Bad Name\"\nversion=\"1.0.0\"\njax=\"1.0.0\"\n").is_err());
        assert!(Manifest::parse(
            "[project]\nname=\"a\"\nversion=\"1.0.0\"\njax=\"1.0.0\"\n[build]\ntargets=[\"beos-x64\"]\n"
        )
        .is_err());
    }
}
