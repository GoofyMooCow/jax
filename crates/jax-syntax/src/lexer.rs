//! The Jax lexer.
//!
//! Jax has no semicolons: statements end at a newline. The lexer therefore
//! records, for every token, whether a line break precedes it (`nl_before`);
//! the parser uses that flag to decide where expressions end.

use crate::diag::{Diagnostic, Span};
use crate::token::{Kw, StrPiece, Tok, Token};

pub struct Lexed {
    pub tokens: Vec<Token>,
    pub diags: Vec<Diagnostic>,
}

/// Lex `src`. `base` is added to every span offset, which lets the parser
/// lex the inside of a string interpolation with spans that still point into
/// the original file.
pub fn lex(src: &str, base: u32) -> Lexed {
    let mut lx = Lexer::new(src, base);
    lx.run();
    Lexed {
        tokens: lx.toks,
        diags: lx.diags,
    }
}

struct Lexer {
    chars: Vec<char>,
    offs: Vec<u32>,
    i: usize,
    base: u32,
    toks: Vec<Token>,
    diags: Vec<Diagnostic>,
    nl: bool,
    sp: bool,
}

impl Lexer {
    fn new(src: &str, base: u32) -> Lexer {
        let mut chars = Vec::new();
        let mut offs = Vec::new();
        for (o, c) in src.char_indices() {
            chars.push(c);
            offs.push(o as u32);
        }
        offs.push(src.len() as u32);
        Lexer {
            chars,
            offs,
            i: 0,
            base,
            toks: Vec::new(),
            diags: Vec::new(),
            nl: true,
            sp: true,
        }
    }

    fn peek(&self, n: usize) -> char {
        self.chars.get(self.i + n).copied().unwrap_or('\0')
    }

    fn at_end(&self) -> bool {
        self.i >= self.chars.len()
    }

    fn span(&self, from: usize, to: usize) -> Span {
        Span::new(self.base + self.offs[from], self.base + self.offs[to])
    }

    fn push(&mut self, kind: Tok, from: usize) {
        let span = self.span(from, self.i);
        self.toks.push(Token {
            kind,
            span,
            nl_before: self.nl,
            space_before: self.sp,
        });
        self.nl = false;
        self.sp = false;
    }

    fn error(&mut self, from: usize, to: usize, msg: impl Into<String>) {
        let span = self.span(from, to.max(from + 1).min(self.chars.len()));
        self.diags.push(Diagnostic::error(span, msg).with_code("E0001"));
    }

    fn run(&mut self) {
        loop {
            self.skip_trivia();
            if self.at_end() {
                let from = self.chars.len();
                self.push(Tok::Eof, from);
                return;
            }
            let start = self.i;
            let c = self.peek(0);
            if c.is_alphabetic() || c == '_' {
                self.ident(start);
            } else if c.is_ascii_digit() {
                self.number(start);
            } else if c == '"' {
                self.string(start);
            } else {
                self.operator(start, c);
            }
        }
    }

    fn skip_trivia(&mut self) {
        loop {
            let c = self.peek(0);
            if self.at_end() {
                return;
            }
            if c == '\n' {
                self.nl = true;
                self.sp = true;
                self.i += 1;
            } else if c.is_whitespace() {
                self.sp = true;
                self.i += 1;
            } else if c == '/' && self.peek(1) == '/' {
                while !self.at_end() && self.peek(0) != '\n' {
                    self.i += 1;
                }
                self.sp = true;
            } else if c == '/' && self.peek(1) == '*' {
                let start = self.i;
                let mut depth = 0;
                loop {
                    if self.at_end() {
                        self.error(start, start + 2, "unterminated block comment");
                        return;
                    }
                    if self.peek(0) == '/' && self.peek(1) == '*' {
                        depth += 1;
                        self.i += 2;
                    } else if self.peek(0) == '*' && self.peek(1) == '/' {
                        depth -= 1;
                        self.i += 2;
                        if depth == 0 {
                            break;
                        }
                    } else {
                        if self.peek(0) == '\n' {
                            self.nl = true;
                        }
                        self.i += 1;
                    }
                }
                self.sp = true;
            } else {
                return;
            }
        }
    }

    fn ident(&mut self, start: usize) {
        while self.peek(0).is_alphanumeric() || self.peek(0) == '_' {
            self.i += 1;
        }
        let text: String = self.chars[start..self.i].iter().collect();
        let kind = match Kw::from_str(&text) {
            Some(k) => Tok::Kw(k),
            None => Tok::Ident(text),
        };
        self.push(kind, start);
    }

    fn number(&mut self, start: usize) {
        // Radix prefixes.
        if self.peek(0) == '0' && matches!(self.peek(1), 'x' | 'b' | 'o') {
            let radix = match self.peek(1) {
                'x' => 16,
                'b' => 2,
                _ => 8,
            };
            self.i += 2;
            let digits_start = self.i;
            while self.peek(0).is_digit(radix) || self.peek(0) == '_' {
                self.i += 1;
            }
            let text: String = self.chars[digits_start..self.i]
                .iter()
                .filter(|c| **c != '_')
                .collect();
            match i64::from_str_radix(&text, radix) {
                Ok(n) => self.push(Tok::Int(n), start),
                Err(_) => {
                    self.error(start, self.i, "invalid integer literal");
                    self.push(Tok::Int(0), start);
                }
            }
            return;
        }

        while self.peek(0).is_ascii_digit() || (self.peek(0) == '_' && self.peek(1).is_ascii_digit()) {
            self.i += 1;
        }
        let mut is_float = false;
        // `1.5` is a float; `1..5` and `1.max(2)` are not.
        if self.peek(0) == '.' && self.peek(1).is_ascii_digit() {
            is_float = true;
            self.i += 1;
            while self.peek(0).is_ascii_digit() || (self.peek(0) == '_' && self.peek(1).is_ascii_digit()) {
                self.i += 1;
            }
        }
        if matches!(self.peek(0), 'e' | 'E')
            && (self.peek(1).is_ascii_digit()
                || (matches!(self.peek(1), '+' | '-') && self.peek(2).is_ascii_digit()))
        {
            is_float = true;
            self.i += 2;
            while self.peek(0).is_ascii_digit() {
                self.i += 1;
            }
        }
        let text: String = self.chars[start..self.i].iter().filter(|c| **c != '_').collect();
        if is_float {
            match text.parse::<f64>() {
                Ok(f) => self.push(Tok::Float(f), start),
                Err(_) => {
                    self.error(start, self.i, "invalid float literal");
                    self.push(Tok::Float(0.0), start);
                }
            }
        } else {
            match text.parse::<i64>() {
                Ok(n) => self.push(Tok::Int(n), start),
                Err(_) => {
                    self.error(start, self.i, "integer literal is too large for `Int` (64-bit)");
                    self.push(Tok::Int(0), start);
                }
            }
        }
    }

    fn string(&mut self, start: usize) {
        self.i += 1; // opening quote
        let mut pieces: Vec<StrPiece> = Vec::new();
        let mut lit = String::new();
        loop {
            if self.at_end() {
                self.error(start, start + 1, "unterminated string literal");
                break;
            }
            let c = self.peek(0);
            match c {
                '"' => {
                    self.i += 1;
                    break;
                }
                '\\' => {
                    let esc_start = self.i;
                    self.i += 1;
                    let e = self.peek(0);
                    self.i += 1;
                    match e {
                        'n' => lit.push('\n'),
                        't' => lit.push('\t'),
                        'r' => lit.push('\r'),
                        '0' => lit.push('\0'),
                        '\\' => lit.push('\\'),
                        '"' => lit.push('"'),
                        '{' => lit.push('{'),
                        '}' => lit.push('}'),
                        _ => self.error(esc_start, self.i, format!("unknown escape sequence `\\{e}`")),
                    }
                }
                '{' => {
                    self.i += 1;
                    let expr_start = self.i;
                    let mut depth = 1;
                    while !self.at_end() {
                        match self.peek(0) {
                            '{' => depth += 1,
                            '}' => {
                                depth -= 1;
                                if depth == 0 {
                                    break;
                                }
                            }
                            '"' => {
                                // Skip a nested string literal.
                                self.i += 1;
                                while !self.at_end() && self.peek(0) != '"' {
                                    if self.peek(0) == '\\' {
                                        self.i += 1;
                                    }
                                    self.i += 1;
                                }
                            }
                            _ => {}
                        }
                        self.i += 1;
                    }
                    if self.at_end() {
                        self.error(start, start + 1, "unterminated `{` in string interpolation");
                        break;
                    }
                    let src: String = self.chars[expr_start..self.i].iter().collect();
                    let offset = self.base + self.offs[expr_start];
                    self.i += 1; // closing brace
                    if !lit.is_empty() {
                        pieces.push(StrPiece::Lit(std::mem::take(&mut lit)));
                    }
                    pieces.push(StrPiece::Expr { src, offset });
                }
                _ => {
                    lit.push(c);
                    self.i += 1;
                }
            }
        }
        if !lit.is_empty() || pieces.is_empty() {
            pieces.push(StrPiece::Lit(lit));
        }
        self.push(Tok::Str(pieces), start);
    }

    fn operator(&mut self, start: usize, c: char) {
        let n = self.peek(1);
        let n2 = self.peek(2);
        let (kind, len) = match (c, n, n2) {
            ('.', '.', '=') => (Tok::DotDotEq, 3),
            ('.', '.', _) => (Tok::DotDot, 2),
            ('.', _, _) => (Tok::Dot, 1),
            ('?', '?', _) => (Tok::QuestionQuestion, 2),
            ('?', '.', _) => (Tok::QuestionDot, 2),
            ('?', _, _) => (Tok::Question, 1),
            ('=', '=', _) => (Tok::EqEq, 2),
            ('=', '>', _) => (Tok::FatArrow, 2),
            ('=', _, _) => (Tok::Eq, 1),
            ('!', '=', _) => (Tok::NotEq, 2),
            ('!', _, _) => (Tok::Bang, 1),
            ('<', '=', _) => (Tok::Le, 2),
            ('<', _, _) => (Tok::Lt, 1),
            ('>', '=', _) => (Tok::Ge, 2),
            ('>', _, _) => (Tok::Gt, 1),
            ('&', '&', _) => (Tok::AndAnd, 2),
            ('|', '|', _) => (Tok::OrOr, 2),
            ('|', _, _) => (Tok::Pipe, 1),
            ('+', '=', _) => (Tok::PlusEq, 2),
            ('+', _, _) => (Tok::Plus, 1),
            ('-', '>', _) => (Tok::Arrow, 2),
            ('-', '=', _) => (Tok::MinusEq, 2),
            ('-', _, _) => (Tok::Minus, 1),
            ('*', '=', _) => (Tok::StarEq, 2),
            ('*', _, _) => (Tok::Star, 1),
            ('/', '=', _) => (Tok::SlashEq, 2),
            ('/', _, _) => (Tok::Slash, 1),
            ('%', '=', _) => (Tok::PercentEq, 2),
            ('%', _, _) => (Tok::Percent, 1),
            (',', _, _) => (Tok::Comma, 1),
            (':', _, _) => (Tok::Colon, 1),
            (';', _, _) => (Tok::Semi, 1),
            ('(', _, _) => (Tok::LParen, 1),
            (')', _, _) => (Tok::RParen, 1),
            ('{', _, _) => (Tok::LBrace, 1),
            ('}', _, _) => (Tok::RBrace, 1),
            ('[', _, _) => (Tok::LBracket, 1),
            (']', _, _) => (Tok::RBracket, 1),
            ('@', _, _) => (Tok::At, 1),
            _ => {
                self.i += 1;
                let msg = if c == '&' {
                    "`&` is not an operator in Jax (there are no references to take; use `&&` for logical and)".to_string()
                } else {
                    format!("unexpected character `{c}`")
                };
                self.error(start, self.i, msg);
                self.sp = true;
                return;
            }
        };
        self.i += len;
        self.push(kind, start);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(src: &str) -> Vec<Tok> {
        let l = lex(src, 0);
        assert!(l.diags.is_empty(), "{:?}", l.diags);
        l.tokens.into_iter().map(|t| t.kind).collect()
    }

    #[test]
    fn basic_tokens() {
        let k = kinds("let x = 1 + 2.5");
        assert_eq!(
            k,
            vec![
                Tok::Kw(Kw::Let),
                Tok::Ident("x".into()),
                Tok::Eq,
                Tok::Int(1),
                Tok::Plus,
                Tok::Float(2.5),
                Tok::Eof
            ]
        );
    }

    #[test]
    fn range_is_not_float() {
        let k = kinds("0..10");
        assert_eq!(k, vec![Tok::Int(0), Tok::DotDot, Tok::Int(10), Tok::Eof]);
    }

    #[test]
    fn newline_flag() {
        let l = lex("a\nb c", 0);
        assert!(l.tokens[1].nl_before);
        assert!(!l.tokens[2].nl_before);
    }

    #[test]
    fn interpolation() {
        let k = kinds(r#""hi {name}!""#);
        match &k[0] {
            Tok::Str(p) => {
                assert_eq!(p.len(), 3);
                assert_eq!(p[0], StrPiece::Lit("hi ".into()));
                assert!(matches!(&p[1], StrPiece::Expr { src, .. } if src == "name"));
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn comments_and_operators() {
        let k = kinds("a ?? b // hi\n/* x /* nested */ */ c?.d");
        assert!(k.contains(&Tok::QuestionQuestion));
        assert!(k.contains(&Tok::QuestionDot));
    }

    #[test]
    fn bad_char_reported() {
        let l = lex("a $ b", 0);
        assert_eq!(l.diags.len(), 1);
    }
}
