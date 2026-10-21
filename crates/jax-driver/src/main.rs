//! `jax-driver`: the toolchain-side command-line interface.
//!
//! Users normally run it through the `jax` launcher, which picks the exact
//! toolchain version a project pins and executes this binary from it.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::exit;

use jax_bytecode::Program;
use jax_compiler::{render, Options};
use jax_manifest::{find_manifest, scaffold, Kind, Manifest, Version, TARGETS};
use jax_syntax::{Diagnostic, SourceFile};
use jax_toolchain::{exe_name, host_target};

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    exit(run(&args));
}

fn run(args: &[String]) -> i32 {
    let rest = if args.is_empty() { &[][..] } else { &args[1..] };
    match args.first().map(|s| s.as_str()) {
        Some("new") => cmd_new(rest, false),
        Some("init") => cmd_new(rest, true),
        Some("check") => cmd_check(rest),
        Some("build") => cmd_build(rest),
        Some("run") => cmd_run(rest),
        Some("test") => cmd_test(rest),
        Some("dump") => cmd_dump(rest),
        Some("version") | Some("--version") | Some("-V") => {
            println!("jax {VERSION} ({})", host_target());
            0
        }
        Some("help") | Some("--help") | Some("-h") | None => {
            print_help();
            0
        }
        Some(cmd @ ("fmt" | "lint" | "doc" | "bench" | "add" | "remove" | "update" | "vendor" | "publish" | "dist" | "capsule" | "lsp")) => {
            eprintln!("error: `jax {cmd}` is planned but not implemented in Jax {VERSION} yet (see docs/STATUS.md)");
            2
        }
        Some(other) => {
            let known = ["new", "init", "check", "build", "run", "test", "dump", "version", "help"];
            match jax_syntax::suggest(other, known) {
                Some(s) => eprintln!("error: unknown command `{other}`; did you mean `{s}`?"),
                None => eprintln!("error: unknown command `{other}`; run `jax help`"),
            }
            2
        }
    }
}

fn print_help() {
    println!(
        "Jax {VERSION}: a portability-first programming language

USAGE:
    jax <command> [options]

COMMANDS:
    new <name> [--lib]        Create a new project in ./<name>
    init [--lib]              Create a project in the current directory
    check [file.jax]          Type-check without producing output
    build [file.jax]          Compile to a standalone executable
        --target <platform>   Cross-compile (needs that runtime in the toolchain)
        --bundle              Produce a portable .jxb bundle instead
        -o <path>             Output path
    run [file.jax]            Compile and run
    test [file.jax]           Run `test \"...\" {{ }}` blocks
    dump tokens|ast|bytecode <file.jax>
                              Inspect compiler stages
    version                   Print the toolchain version

PLANNED: fmt, lint, doc, bench, add/remove/update, vendor, publish, dist, capsule, lsp

Run these through the `jax` launcher so each project uses the toolchain it pins."
    );
}

// ----- project discovery ---------------------------------------------------

struct Project {
    root: PathBuf,
    manifest: Manifest,
}

fn load_project() -> Result<Option<Project>, String> {
    let cwd = env::current_dir().map_err(|e| e.to_string())?;
    let Some(path) = find_manifest(&cwd) else { return Ok(None) };
    let manifest = Manifest::load(&path).map_err(|e| e.to_string())?;
    let root = path.parent().unwrap().to_path_buf();
    let mine = Version::parse(VERSION).unwrap();
    if manifest.project.jax != mine && env::var_os("JAX_ALLOW_VERSION_MISMATCH").is_none() {
        return Err(format!(
            "this project pins Jax {} but this toolchain is Jax {VERSION}.\n  Run commands through the `jax` launcher, which selects the pinned version automatically\n  (set JAX_ALLOW_VERSION_MISMATCH=1 to override while developing the toolchain).",
            manifest.project.jax
        ));
    }
    Ok(Some(Project { root, manifest }))
}

/// Source file to compile: an explicit `*.jax` argument, else the project entry point.
fn source_path(args: &[String], project: &Option<Project>) -> Result<PathBuf, String> {
    if let Some(p) = args.iter().find(|a| a.ends_with(".jax")) {
        return Ok(PathBuf::from(p));
    }
    match project {
        Some(pr) => Ok(pr.root.join(pr.manifest.entry_point())),
        None => Err("no jax.toml found; run inside a project or pass a .jax file".into()),
    }
}

fn print_diags(file: &SourceFile, diags: &[Diagnostic]) {
    for d in diags {
        eprintln!("{}", render(file, d));
    }
    let errors = diags.iter().filter(|d| d.is_error()).count();
    if errors > 0 {
        eprintln!("error: aborting due to {errors} previous error{}", if errors == 1 { "" } else { "s" });
    }
}

fn read_source(path: &Path) -> Result<SourceFile, String> {
    let src = fs::read_to_string(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let name = path.to_string_lossy().replace('\\', "/");
    Ok(SourceFile::new(name, src))
}

fn options(include_tests: bool, require_main: bool) -> Options {
    Options {
        jax_version: VERSION.to_string(),
        include_tests,
        require_main,
    }
}

fn fail(msg: impl AsRef<str>) -> i32 {
    eprintln!("error: {}", msg.as_ref());
    1
}

// ----- commands --------------------------------------------------------------

fn cmd_check(args: &[String]) -> i32 {
    let project = match load_project() {
        Ok(p) => p,
        Err(e) => return fail(e),
    };
    let path = match source_path(args, &project) {
        Ok(p) => p,
        Err(e) => return fail(e),
    };
    let file = match read_source(&path) {
        Ok(f) => f,
        Err(e) => return fail(e),
    };
    let is_lib = project.as_ref().map(|p| p.manifest.project.kind == Kind::Lib).unwrap_or(false);
    match jax_compiler::check(&file, &options(true, !is_lib)) {
        Ok(warnings) => {
            print_diags(&file, &warnings);
            println!("Checked {}: no errors", file.name);
            0
        }
        Err(diags) => {
            print_diags(&file, &diags);
            1
        }
    }
}

fn compile_for(args: &[String], include_tests: bool) -> Result<(Program, SourceFile, Option<Project>), i32> {
    let project = load_project().map_err(fail)?;
    let path = source_path(args, &project).map_err(fail)?;
    let file = read_source(&path).map_err(fail)?;
    let is_lib = project.as_ref().map(|p| p.manifest.project.kind == Kind::Lib).unwrap_or(false);
    match jax_compiler::compile(&file, &options(include_tests, !include_tests && !is_lib)) {
        Ok(p) => Ok((p, file, project)),
        Err(diags) => {
            print_diags(&file, &diags);
            Err(1)
        }
    }
}

fn cmd_run(args: &[String]) -> i32 {
    // Everything after `--` belongs to the program (not yet forwarded: `main` takes no arguments).
    let own: Vec<String> = args.iter().take_while(|a| a.as_str() != "--").cloned().collect();
    let (program, _file, _project) = match compile_for(&own, false) {
        Ok(x) => x,
        Err(c) => return c,
    };
    let stdout = std::io::stdout();
    let out = std::io::BufWriter::with_capacity(64 * 1024, stdout.lock());
    let mut vm = jax_vm::Vm::new(&program, Box::new(out));
    match vm.run_entry() {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("{e}");
            e.exit_code()
        }
    }
}

fn cmd_test(args: &[String]) -> i32 {
    let (program, file, _project) = match compile_for(args, true) {
        Ok(x) => x,
        Err(c) => return c,
    };
    if program.tests.is_empty() {
        println!("no tests found in {}", file.name);
        return 0;
    }
    let mut failed = 0;
    for (name, idx) in &program.tests {
        let stdout = std::io::stdout();
        let mut vm = jax_vm::Vm::new(&program, Box::new(std::io::BufWriter::new(stdout.lock())));
        match vm.run_function(*idx) {
            Ok(_) => println!("test {name} ... ok"),
            Err(e) => {
                failed += 1;
                println!("test {name} ... FAILED\n    {}", e.to_string().replace('\n', "\n    "));
            }
        }
    }
    let total = program.tests.len();
    println!("\ntest result: {}. {} passed; {failed} failed", if failed == 0 { "ok" } else { "FAILED" }, total - failed);
    if failed == 0 {
        0
    } else {
        1
    }
}

fn runtime_for(target: &str) -> Result<PathBuf, String> {
    let exe = env::current_exe().map_err(|e| e.to_string())?;
    let exe_dir = exe.parent().ok_or("cannot locate the toolchain directory")?;
    let suffix = if target.starts_with("windows") { "jax-rt.exe" } else { "jax-rt" };
    let installed = exe_dir.join("..").join("runtimes").join(target).join(suffix);
    if installed.is_file() {
        return Ok(installed);
    }
    if target == host_target() {
        let sibling = exe_dir.join(exe_name("jax-rt"));
        if sibling.is_file() {
            return Ok(sibling);
        }
    }
    let mut available = Vec::new();
    if let Ok(rd) = fs::read_dir(exe_dir.join("..").join("runtimes")) {
        for e in rd.flatten() {
            available.push(e.file_name().to_string_lossy().to_string());
        }
    }
    available.sort();
    Err(format!(
        "this toolchain has no runtime for {target} (available: {})",
        if available.is_empty() { "none".to_string() } else { available.join(", ") }
    ))
}

fn cmd_build(args: &[String]) -> i32 {
    let mut target = host_target().to_string();
    let mut bundle = false;
    let mut out: Option<PathBuf> = None;
    let mut rest: Vec<String> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--target" => {
                i += 1;
                match args.get(i) {
                    Some(t) => target = t.clone(),
                    None => return fail("`--target` needs a platform name"),
                }
            }
            "--bundle" => bundle = true,
            "-o" => {
                i += 1;
                match args.get(i) {
                    Some(p) => out = Some(PathBuf::from(p)),
                    None => return fail("`-o` needs a path"),
                }
            }
            other => rest.push(other.to_string()),
        }
        i += 1;
    }
    if !TARGETS.contains(&target.as_str()) {
        return fail(format!("unknown target `{target}`; known targets: {}", TARGETS.join(", ")));
    }
    let (program, file, project) = match compile_for(&rest, false) {
        Ok(x) => x,
        Err(c) => return c,
    };
    let stem = match &project {
        Some(p) if !rest.iter().any(|a| a.ends_with(".jax")) => p.manifest.project.name.clone(),
        _ => Path::new(&file.name)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "out".into()),
    };
    let target_exe_name = if target.starts_with("windows") { format!("{stem}.exe") } else { stem.clone() };
    let default_out = match &project {
        Some(p) => p.root.join("build").join(&target).join(if bundle { format!("{stem}.jxb") } else { target_exe_name }),
        None => PathBuf::from(if bundle { format!("{stem}.jxb") } else { target_exe_name }),
    };
    let out_path = out.unwrap_or(default_out);
    if let Some(parent) = out_path.parent() {
        if !parent.as_os_str().is_empty() {
            if let Err(e) = fs::create_dir_all(parent) {
                return fail(format!("cannot create {}: {e}", parent.display()));
            }
        }
    }
    let bytes = if bundle {
        program.encode()
    } else {
        let rt_path = match runtime_for(&target) {
            Ok(p) => p,
            Err(e) => return fail(e),
        };
        let runtime = match fs::read(&rt_path) {
            Ok(b) => b,
            Err(e) => return fail(format!("cannot read runtime {}: {e}", rt_path.display())),
        };
        program.embed_in(&runtime)
    };
    if let Err(e) = fs::write(&out_path, &bytes) {
        return fail(format!("cannot write {}: {e}", out_path.display()));
    }
    #[cfg(unix)]
    if !bundle {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(&out_path, fs::Permissions::from_mode(0o755));
    }
    println!(
        "Built {} ({}, {:.1} KiB)",
        out_path.display(),
        if bundle { "portable bundle".to_string() } else { target },
        bytes.len() as f64 / 1024.0
    );
    0
}

fn cmd_dump(args: &[String]) -> i32 {
    let (Some(what), Some(path)) = (args.first(), args.get(1)) else {
        return fail("usage: jax dump tokens|ast|bytecode <file.jax>");
    };
    let file = match read_source(Path::new(path)) {
        Ok(f) => f,
        Err(e) => return fail(e),
    };
    match what.as_str() {
        "tokens" => {
            let lexed = jax_syntax::lexer::lex(&file.src, 0);
            for t in &lexed.tokens {
                let (line, col) = file.line_col(t.span.lo);
                println!("{line}:{col}\t{:?}", t.kind);
            }
            print_diags(&file, &lexed.diags);
            0
        }
        "ast" => {
            let r = jax_syntax::parse(&file.src);
            println!("{:#?}", r.program);
            print_diags(&file, &r.diags);
            if r.diags.iter().any(|d| d.is_error()) {
                1
            } else {
                0
            }
        }
        "bytecode" => match jax_compiler::compile(&file, &options(true, false)) {
            Ok(p) => {
                print!("{}", p.disassemble());
                0
            }
            Err(d) => {
                print_diags(&file, &d);
                1
            }
        },
        other => fail(format!("unknown dump target `{other}`; use tokens, ast or bytecode")),
    }
}

fn cmd_new(args: &[String], in_place: bool) -> i32 {
    let lib = args.iter().any(|a| a == "--lib");
    let kind = if lib { Kind::Lib } else { Kind::App };
    let names: Vec<&String> = args.iter().filter(|a| !a.starts_with("--")).collect();
    let (dir, name) = if in_place {
        let cwd = match env::current_dir() {
            Ok(c) => c,
            Err(e) => return fail(e.to_string()),
        };
        let name = cwd.file_name().map(|s| s.to_string_lossy().to_lowercase().replace(' ', "-")).unwrap_or_default();
        (cwd, name)
    } else {
        match names.first() {
            Some(n) => (PathBuf::from(n), Path::new(n.as_str()).file_name().unwrap().to_string_lossy().to_string()),
            None => return fail("usage: jax new <name> [--lib]"),
        }
    };
    let version = Version::parse(VERSION).unwrap();
    let manifest_text = scaffold(&name, kind, &version);
    if let Err(e) = Manifest::parse(&manifest_text) {
        return fail(format!("`{name}` is not a valid project name: {}", e.message));
    }
    let manifest_path = dir.join("jax.toml");
    if manifest_path.exists() {
        return fail(format!("{} already exists", manifest_path.display()));
    }
    if !in_place && dir.exists() {
        return fail(format!("directory {} already exists", dir.display()));
    }
    let entry = if lib { "src/lib.jax" } else { "src/main.jax" };
    let entry_src = if lib {
        "/// A function other Jax code can use.\npub fn greet(name: Text) -> Text {\n    \"Hello, {name}!\"\n}\n\ntest \"greet says hello\" {\n    assert(greet(\"Jax\") == \"Hello, Jax!\")\n}\n"
    } else {
        "fn main() {\n    print(\"Hello, Jax!\")\n}\n"
    };
    let result = (|| -> std::io::Result<()> {
        fs::create_dir_all(dir.join("src"))?;
        fs::write(&manifest_path, manifest_text)?;
        fs::write(dir.join(entry), entry_src)?;
        fs::write(dir.join(".gitignore"), "build/\n.jax/\n")?;
        Ok(())
    })();
    match result {
        Ok(()) => {
            println!("Created {} `{name}` in {} (pinned to Jax {VERSION})", if lib { "library" } else { "application" }, dir.display());
            0
        }
        Err(e) => fail(e.to_string()),
    }
}
