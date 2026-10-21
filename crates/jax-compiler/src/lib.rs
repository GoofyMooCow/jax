//! The Jax compiler pipeline: source text -> bytecode program.
//!
//! lex + parse (`jax-syntax`) -> type check (`jax-types`) -> bytecode (`jax-codegen`).

use jax_bytecode::Program;
use jax_syntax::ast;
use jax_syntax::{Diagnostic, SourceFile};

pub use jax_syntax::render;

/// The compiler version, recorded in every compiled program.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Clone, Debug)]
pub struct Options {
    /// Toolchain version string stamped into the output.
    pub jax_version: String,
    /// Compile `test` blocks too.
    pub include_tests: bool,
    /// Require a `main` function (applications).
    pub require_main: bool,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            jax_version: VERSION.to_string(),
            include_tests: false,
            require_main: true,
        }
    }
}

fn has_errors(diags: &[Diagnostic]) -> bool {
    diags.iter().any(|d| d.is_error())
}

/// Parse only.
pub fn parse(file: &SourceFile) -> Result<ast::Program, Vec<Diagnostic>> {
    let r = jax_syntax::parse(&file.src);
    if has_errors(&r.diags) {
        Err(r.diags)
    } else {
        Ok(r.program)
    }
}

/// Parse and type check without generating code (`jax check`). Returns warnings on success.
pub fn check(file: &SourceFile, opts: &Options) -> Result<Vec<Diagnostic>, Vec<Diagnostic>> {
    let program = parse(file)?;
    let out = jax_types::check(&program);
    let mut diags = out.diags;
    if opts.require_main && !out.info.has_main && !has_errors(&diags) {
        diags.push(
            Diagnostic::error(jax_syntax::Span::new(0, 0), "this program has no `main` function")
                .with_help("add `fn main() { ... }`"),
        );
    }
    if has_errors(&diags) {
        Err(diags)
    } else {
        Ok(diags)
    }
}

/// Compile a source file to bytecode.
pub fn compile(file: &SourceFile, opts: &Options) -> Result<Program, Vec<Diagnostic>> {
    let program = parse(file)?;
    let out = jax_types::check(&program);
    let mut diags = out.diags;
    if opts.require_main && !out.info.has_main && !has_errors(&diags) {
        diags.push(
            Diagnostic::error(jax_syntax::Span::new(0, 0), "this program has no `main` function")
                .with_help("add `fn main() { ... }`"),
        );
    }
    if has_errors(&diags) {
        return Err(diags);
    }
    let gen_opts = jax_codegen::GenOptions {
        jax_version: opts.jax_version.clone(),
        include_tests: opts.include_tests,
    };
    jax_codegen::generate(&program, &out.info, &gen_opts)
        .map_err(|e| vec![Diagnostic::error(jax_syntax::Span::new(0, 0), format!("internal compiler error: {e}"))])
}
