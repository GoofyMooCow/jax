//! Token definitions for the Jax lexer.

use crate::diag::Span;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kw {
    Fn,
    Let,
    Var,
    Const,
    If,
    Else,
    Match,
    While,
    Loop,
    For,
    In,
    Return,
    Break,
    Continue,
    Struct,
    Enum,
    Impl,
    Trait,
    Pub,
    Use,
    As,
    True,
    False,
    None,
    SelfValue,
    Async,
    Await,
    Mut,
    Move,
    With,
}

impl Kw {
    pub fn from_str(s: &str) -> Option<Kw> {
        Some(match s {
            "fn" => Kw::Fn,
            "let" => Kw::Let,
            "var" => Kw::Var,
            "const" => Kw::Const,
            "if" => Kw::If,
            "else" => Kw::Else,
            "match" => Kw::Match,
            "while" => Kw::While,
            "loop" => Kw::Loop,
            "for" => Kw::For,
            "in" => Kw::In,
            "return" => Kw::Return,
            "break" => Kw::Break,
            "continue" => Kw::Continue,
            "struct" => Kw::Struct,
            "enum" => Kw::Enum,
            "impl" => Kw::Impl,
            "trait" => Kw::Trait,
            "pub" => Kw::Pub,
            "use" => Kw::Use,
            "as" => Kw::As,
            "true" => Kw::True,
            "false" => Kw::False,
            "none" => Kw::None,
            "self" => Kw::SelfValue,
            "async" => Kw::Async,
            "await" => Kw::Await,
            "mut" => Kw::Mut,
            "move" => Kw::Move,
            "with" => Kw::With,
            _ => return None,
        })
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Kw::Fn => "fn",
            Kw::Let => "let",
            Kw::Var => "var",
            Kw::Const => "const",
            Kw::If => "if",
            Kw::Else => "else",
            Kw::Match => "match",
            Kw::While => "while",
            Kw::Loop => "loop",
            Kw::For => "for",
            Kw::In => "in",
            Kw::Return => "return",
            Kw::Break => "break",
            Kw::Continue => "continue",
            Kw::Struct => "struct",
            Kw::Enum => "enum",
            Kw::Impl => "impl",
            Kw::Trait => "trait",
            Kw::Pub => "pub",
            Kw::Use => "use",
            Kw::As => "as",
            Kw::True => "true",
            Kw::False => "false",
            Kw::None => "none",
            Kw::SelfValue => "self",
            Kw::Async => "async",
            Kw::Await => "await",
            Kw::Mut => "mut",
            Kw::Move => "move",
            Kw::With => "with",
        }
    }
}

/// A piece of a string literal: plain text or an interpolated `{expr}`.
#[derive(Clone, Debug, PartialEq)]
pub enum StrPiece {
    Lit(String),
    /// Source text of the expression and the absolute byte offset where it starts.
    Expr { src: String, offset: u32 },
}

#[derive(Clone, Debug, PartialEq)]
pub enum Tok {
    Ident(String),
    Int(i64),
    Float(f64),
    Str(Vec<StrPiece>),
    Kw(Kw),

    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Eq,
    EqEq,
    NotEq,
    Lt,
    Le,
    Gt,
    Ge,
    AndAnd,
    OrOr,
    Bang,
    Question,
    QuestionQuestion,
    QuestionDot,
    Dot,
    DotDot,
    DotDotEq,
    Comma,
    Colon,
    Semi,
    Arrow,
    FatArrow,
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Pipe,
    At,
    PlusEq,
    MinusEq,
    StarEq,
    SlashEq,
    PercentEq,
    Eof,
}

impl Tok {
    /// How the token is described in "expected X, found Y" messages.
    pub fn describe(&self) -> String {
        match self {
            Tok::Ident(s) => format!("identifier `{s}`"),
            Tok::Int(n) => format!("number `{n}`"),
            Tok::Float(n) => format!("number `{n}`"),
            Tok::Str(_) => "string literal".into(),
            Tok::Kw(k) => format!("keyword `{}`", k.as_str()),
            Tok::Eof => "end of file".into(),
            other => format!("`{}`", other.symbol()),
        }
    }

    pub fn symbol(&self) -> &'static str {
        match self {
            Tok::Plus => "+",
            Tok::Minus => "-",
            Tok::Star => "*",
            Tok::Slash => "/",
            Tok::Percent => "%",
            Tok::Eq => "=",
            Tok::EqEq => "==",
            Tok::NotEq => "!=",
            Tok::Lt => "<",
            Tok::Le => "<=",
            Tok::Gt => ">",
            Tok::Ge => ">=",
            Tok::AndAnd => "&&",
            Tok::OrOr => "||",
            Tok::Bang => "!",
            Tok::Question => "?",
            Tok::QuestionQuestion => "??",
            Tok::QuestionDot => "?.",
            Tok::Dot => ".",
            Tok::DotDot => "..",
            Tok::DotDotEq => "..=",
            Tok::Comma => ",",
            Tok::Colon => ":",
            Tok::Semi => ";",
            Tok::Arrow => "->",
            Tok::FatArrow => "=>",
            Tok::LParen => "(",
            Tok::RParen => ")",
            Tok::LBrace => "{",
            Tok::RBrace => "}",
            Tok::LBracket => "[",
            Tok::RBracket => "]",
            Tok::Pipe => "|",
            Tok::At => "@",
            Tok::PlusEq => "+=",
            Tok::MinusEq => "-=",
            Tok::StarEq => "*=",
            Tok::SlashEq => "/=",
            Tok::PercentEq => "%=",
            _ => "?",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Token {
    pub kind: Tok,
    pub span: Span,
    /// A line break separates this token from the previous one.
    pub nl_before: bool,
    /// Whitespace (or a comment) separates this token from the previous one.
    pub space_before: bool,
}
