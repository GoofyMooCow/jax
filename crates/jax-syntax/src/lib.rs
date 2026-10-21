//! Lexer, parser, AST and diagnostics for the Jax language.

pub mod ast;
pub mod diag;
pub mod lexer;
pub mod parser;
pub mod token;

pub use diag::{render, suggest, Diagnostic, Severity, SourceFile, Span};
pub use parser::{parse, ParseResult};
