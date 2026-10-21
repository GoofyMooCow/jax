//! Recursive-descent parser for Jax.
//!
//! Statements end at a newline (or `;`, or the closing `}` of a block).
//! Inside `(...)` and `[...]` newlines are insignificant. Method chains may
//! continue on the next line when it starts with `.` or `?.`.

use std::cell::Cell;
use std::rc::Rc;

use crate::ast::*;
use crate::diag::{Diagnostic, Span};
use crate::lexer::lex;
use crate::token::{Kw, StrPiece, Tok, Token};

type PResult<T> = Result<T, Diagnostic>;

pub struct ParseResult {
    pub program: Program,
    pub diags: Vec<Diagnostic>,
}

/// Lex and parse a whole source file.
pub fn parse(src: &str) -> ParseResult {
    let lexed = lex(src, 0);
    let mut diags = lexed.diags;
    let ids = Rc::new(Cell::new(0));
    let mut p = Parser::new(lexed.tokens, ids);
    let program = p.parse_program();
    diags.extend(p.diags);
    ParseResult { program, diags }
}

struct Parser {
    toks: Vec<Token>,
    pos: usize,
    diags: Vec<Diagnostic>,
    ids: Rc<Cell<u32>>,
    /// Nesting depth of `(`/`[`; newlines are insignificant when > 0.
    depth: u32,
    last_span: Span,
}

impl Parser {
    fn new(toks: Vec<Token>, ids: Rc<Cell<u32>>) -> Parser {
        Parser {
            toks,
            pos: 0,
            diags: Vec::new(),
            ids,
            depth: 0,
            last_span: Span::DUMMY,
        }
    }

    // ----- token helpers -------------------------------------------------

    fn tok(&self) -> &Token {
        &self.toks[self.pos.min(self.toks.len() - 1)]
    }

    fn kind(&self) -> &Tok {
        &self.tok().kind
    }

    fn peek_kind(&self, n: usize) -> &Tok {
        &self.toks[(self.pos + n).min(self.toks.len() - 1)].kind
    }

    fn bump(&mut self) -> Token {
        let t = self.tok().clone();
        self.last_span = t.span;
        if self.pos < self.toks.len() - 1 {
            self.pos += 1;
        }
        t
    }

    fn is(&self, k: &Tok) -> bool {
        self.kind() == k
    }

    fn is_kw(&self, k: Kw) -> bool {
        matches!(self.kind(), Tok::Kw(x) if *x == k)
    }

    fn eat(&mut self, k: &Tok) -> bool {
        if self.is(k) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn eat_kw(&mut self, k: Kw) -> bool {
        if self.is_kw(k) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn at_eof(&self) -> bool {
        matches!(self.kind(), Tok::Eof)
    }

    fn new_id(&mut self) -> NodeId {
        let id = self.ids.get();
        self.ids.set(id + 1);
        id
    }

    fn unexpected<T>(&self, expected: &str) -> PResult<T> {
        let t = self.tok();
        Err(Diagnostic::error(
            t.span,
            format!("expected {expected}, found {}", t.kind.describe()),
        )
        .with_code("E0100"))
    }

    fn expect(&mut self, k: &Tok) -> PResult<Token> {
        if self.is(k) {
            Ok(self.bump())
        } else {
            self.unexpected(&format!("`{}`", k.symbol()))
        }
    }

    fn expect_kw(&mut self, k: Kw) -> PResult<Token> {
        if self.is_kw(k) {
            Ok(self.bump())
        } else {
            self.unexpected(&format!("`{}`", k.as_str()))
        }
    }

    fn expect_ident(&mut self, what: &str) -> PResult<(String, Span)> {
        if let Tok::Ident(name) = self.kind().clone() {
            let t = self.bump();
            Ok((name, t.span))
        } else {
            self.unexpected(what)
        }
    }

    fn mk(&mut self, span: Span, kind: ExprKind) -> Expr {
        Expr {
            id: self.new_id(),
            span,
            kind,
        }
    }

    // ----- items ---------------------------------------------------------

    fn parse_program(&mut self) -> Program {
        let mut items = Vec::new();
        while !self.at_eof() {
            let before = self.pos;
            match self.parse_item() {
                Ok(item) => items.push(item),
                Err(d) => {
                    self.diags.push(d);
                    self.recover(before);
                }
            }
        }
        Program { items }
    }

    fn is_item_start(&self) -> bool {
        match self.kind() {
            Tok::Kw(Kw::Fn | Kw::Struct | Kw::Enum | Kw::Impl | Kw::Trait | Kw::Use | Kw::Const | Kw::Pub | Kw::Async)
            | Tok::At => true,
            Tok::Ident(s) => s == "test" && matches!(self.peek_kind(1), Tok::Str(_)),
            _ => false,
        }
    }

    /// Skip to the next line that starts an item.
    fn recover(&mut self, before: usize) {
        if self.pos == before {
            self.bump();
        }
        while !self.at_eof() {
            if self.tok().nl_before && self.is_item_start() {
                break;
            }
            self.bump();
        }
    }

    fn parse_item(&mut self) -> PResult<Item> {
        let start = self.tok().span;
        let mut attrs = Vec::new();
        while self.is(&Tok::At) {
            attrs.push(self.parse_attr()?);
        }
        let is_pub = self.eat_kw(Kw::Pub);
        let kind = match self.kind().clone() {
            Tok::Kw(Kw::Fn) | Tok::Kw(Kw::Async) => ItemKind::Fn(self.parse_fn()?),
            Tok::Kw(Kw::Struct) => ItemKind::Struct(self.parse_struct()?),
            Tok::Kw(Kw::Enum) => ItemKind::Enum(self.parse_enum()?),
            Tok::Kw(Kw::Impl) => ItemKind::Impl(self.parse_impl()?),
            Tok::Kw(Kw::Trait) => ItemKind::Trait(self.parse_trait()?),
            Tok::Kw(Kw::Use) => ItemKind::Use(self.parse_use()?),
            Tok::Kw(Kw::Const) => ItemKind::Const(self.parse_const()?),
            Tok::Ident(s) if s == "test" && matches!(self.peek_kind(1), Tok::Str(_)) => {
                ItemKind::Test(self.parse_test()?)
            }
            _ => {
                return self.unexpected(
                    "an item (`fn`, `struct`, `enum`, `impl`, `trait`, `use`, `const` or `test`)",
                )
            }
        };
        Ok(Item {
            attrs,
            is_pub,
            span: start.to(self.last_span),
            kind,
        })
    }

    fn parse_attr(&mut self) -> PResult<Attr> {
        let at = self.expect(&Tok::At)?;
        let (name, nspan) = self.expect_ident("an attribute name")?;
        if self.is(&Tok::LParen) {
            let mut depth = 0;
            loop {
                match self.kind() {
                    Tok::LParen => depth += 1,
                    Tok::RParen => {
                        depth -= 1;
                        if depth == 0 {
                            self.bump();
                            break;
                        }
                    }
                    Tok::Eof => return self.unexpected("`)`"),
                    _ => {}
                }
                self.bump();
            }
        }
        Ok(Attr {
            name,
            span: at.span.to(nspan),
        })
    }

    fn parse_generics_opt(&mut self) -> PResult<Vec<GenericParam>> {
        let mut out = Vec::new();
        if !self.eat(&Tok::Lt) {
            return Ok(out);
        }
        loop {
            if self.is(&Tok::Gt) {
                break;
            }
            let (name, span) = self.expect_ident("a type parameter name")?;
            let mut bounds = Vec::new();
            if self.eat(&Tok::Colon) {
                loop {
                    let (b, _) = self.expect_ident("a trait name")?;
                    bounds.push(b);
                    if !self.eat(&Tok::Plus) {
                        break;
                    }
                }
            }
            out.push(GenericParam { name, bounds, span });
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        self.expect(&Tok::Gt)?;
        Ok(out)
    }

    fn parse_fn(&mut self) -> PResult<FnDecl> {
        let is_async = self.eat_kw(Kw::Async);
        self.expect_kw(Kw::Fn)?;
        let (name, name_span) = self.expect_ident("a function name")?;
        let generics = self.parse_generics_opt()?;
        self.expect(&Tok::LParen)?;
        self.depth += 1;
        let mut self_param = None;
        let mut params = Vec::new();
        let mut first = true;
        while !self.is(&Tok::RParen) {
            if first && self.is_kw(Kw::SelfValue) {
                self.bump();
                self_param = Some(SelfKind::Value);
            } else if first && self.is_kw(Kw::Mut) && matches!(self.peek_kind(1), Tok::Kw(Kw::SelfValue)) {
                self.bump();
                self.bump();
                self_param = Some(SelfKind::Mut);
            } else {
                let (pname, span) = self.expect_ident("a parameter name")?;
                self.expect(&Tok::Colon)?;
                let ty = self.parse_type()?;
                let default = if self.eat(&Tok::Eq) {
                    Some(self.parse_expr()?)
                } else {
                    None
                };
                params.push(Param {
                    name: pname,
                    span,
                    ty,
                    default,
                });
            }
            first = false;
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        self.depth -= 1;
        self.expect(&Tok::RParen)?;
        let ret = if self.eat(&Tok::Arrow) {
            Some(self.parse_type()?)
        } else {
            None
        };
        let body = if self.is(&Tok::LBrace) {
            Some(self.parse_block()?)
        } else {
            None
        };
        Ok(FnDecl {
            name,
            name_span,
            is_async,
            generics,
            self_param,
            params,
            ret,
            body,
        })
    }

    fn parse_struct(&mut self) -> PResult<StructDecl> {
        self.expect_kw(Kw::Struct)?;
        let (name, name_span) = self.expect_ident("a struct name")?;
        let generics = self.parse_generics_opt()?;
        self.expect(&Tok::LBrace)?;
        let mut fields = Vec::new();
        loop {
            while self.eat(&Tok::Comma) {}
            if self.is(&Tok::RBrace) || self.at_eof() {
                break;
            }
            let is_pub = self.eat_kw(Kw::Pub);
            let (fname, span) = self.expect_ident("a field name")?;
            self.expect(&Tok::Colon)?;
            let ty = self.parse_type()?;
            fields.push(FieldDecl {
                name: fname,
                span,
                ty,
                is_pub,
            });
        }
        self.expect(&Tok::RBrace)?;
        Ok(StructDecl {
            name,
            name_span,
            generics,
            fields,
        })
    }

    fn parse_enum(&mut self) -> PResult<EnumDecl> {
        self.expect_kw(Kw::Enum)?;
        let (name, name_span) = self.expect_ident("an enum name")?;
        let generics = self.parse_generics_opt()?;
        self.expect(&Tok::LBrace)?;
        let mut variants = Vec::new();
        loop {
            while self.eat(&Tok::Comma) {}
            if self.is(&Tok::RBrace) || self.at_eof() {
                break;
            }
            let (vname, span) = self.expect_ident("a variant name")?;
            let mut fields = Vec::new();
            if self.eat(&Tok::LParen) {
                self.depth += 1;
                while !self.is(&Tok::RParen) {
                    let named = matches!(self.kind(), Tok::Ident(_)) && matches!(self.peek_kind(1), Tok::Colon);
                    if named {
                        let (fname, _) = self.expect_ident("a field name")?;
                        self.expect(&Tok::Colon)?;
                        let ty = self.parse_type()?;
                        fields.push(VariantField {
                            name: Some(fname),
                            ty,
                        });
                    } else {
                        let ty = self.parse_type()?;
                        fields.push(VariantField { name: None, ty });
                    }
                    if !self.eat(&Tok::Comma) {
                        break;
                    }
                }
                self.depth -= 1;
                self.expect(&Tok::RParen)?;
            }
            variants.push(Variant {
                name: vname,
                span,
                fields,
            });
        }
        self.expect(&Tok::RBrace)?;
        Ok(EnumDecl {
            name,
            name_span,
            generics,
            variants,
        })
    }

    fn parse_impl(&mut self) -> PResult<ImplDecl> {
        self.expect_kw(Kw::Impl)?;
        let generics = self.parse_generics_opt()?;
        let first = self.parse_type()?;
        let (trait_ty, self_ty) = if self.eat_kw(Kw::For) {
            let t = self.parse_type()?;
            (Some(first), t)
        } else {
            (None, first)
        };
        let items = self.parse_member_items()?;
        Ok(ImplDecl {
            generics,
            trait_ty,
            self_ty,
            items,
        })
    }

    fn parse_trait(&mut self) -> PResult<TraitDecl> {
        self.expect_kw(Kw::Trait)?;
        let (name, name_span) = self.expect_ident("a trait name")?;
        let generics = self.parse_generics_opt()?;
        let items = self.parse_member_items()?;
        Ok(TraitDecl {
            name,
            name_span,
            generics,
            items,
        })
    }

    fn parse_member_items(&mut self) -> PResult<Vec<Item>> {
        self.expect(&Tok::LBrace)?;
        let saved = self.depth;
        self.depth = 0;
        let mut items = Vec::new();
        while !self.is(&Tok::RBrace) && !self.at_eof() {
            let item = self.parse_item()?;
            if !matches!(item.kind, ItemKind::Fn(_)) {
                return Err(Diagnostic::error(item.span, "only functions are allowed here"));
            }
            items.push(item);
        }
        self.depth = saved;
        self.expect(&Tok::RBrace)?;
        Ok(items)
    }

    fn parse_use(&mut self) -> PResult<UseDecl> {
        self.expect_kw(Kw::Use)?;
        let mut path = vec![self.expect_ident("a module path")?.0];
        while self.eat(&Tok::Dot) {
            path.push(self.expect_ident("a module path segment")?.0);
        }
        let alias = if self.eat_kw(Kw::As) {
            Some(self.expect_ident("an alias")?.0)
        } else {
            None
        };
        Ok(UseDecl { path, alias })
    }

    fn parse_const(&mut self) -> PResult<ConstDecl> {
        self.expect_kw(Kw::Const)?;
        let (name, name_span) = self.expect_ident("a constant name")?;
        let ty = if self.eat(&Tok::Colon) {
            Some(self.parse_type()?)
        } else {
            None
        };
        self.expect(&Tok::Eq)?;
        let value = self.parse_expr()?;
        Ok(ConstDecl {
            name,
            name_span,
            ty,
            value,
        })
    }

    fn parse_test(&mut self) -> PResult<TestDecl> {
        self.bump(); // `test`
        let t = self.bump();
        let name = match t.kind {
            Tok::Str(pieces) => pieces
                .into_iter()
                .map(|p| match p {
                    StrPiece::Lit(s) => s,
                    StrPiece::Expr { src, .. } => format!("{{{src}}}"),
                })
                .collect::<String>(),
            _ => unreachable!(),
        };
        let body = self.parse_block()?;
        Ok(TestDecl { name, body })
    }

    // ----- types ---------------------------------------------------------

    fn parse_type(&mut self) -> PResult<Type> {
        let start = self.tok().span;
        let mut ty = match self.kind().clone() {
            Tok::Ident(name) if name == "dyn" && matches!(self.peek_kind(1), Tok::Ident(_)) => {
                self.bump();
                let (tn, _) = self.expect_ident("a trait name")?;
                Type {
                    span: start.to(self.last_span),
                    kind: TypeKind::Dyn(tn),
                }
            }
            Tok::Ident(mut name) => {
                self.bump();
                while self.is(&Tok::Dot) && matches!(self.peek_kind(1), Tok::Ident(_)) {
                    self.bump();
                    let (seg, _) = self.expect_ident("a type name")?;
                    name.push('.');
                    name.push_str(&seg);
                }
                let mut args = Vec::new();
                if self.eat(&Tok::Lt) {
                    loop {
                        args.push(self.parse_type()?);
                        if !self.eat(&Tok::Comma) {
                            break;
                        }
                    }
                    self.expect(&Tok::Gt)?;
                }
                Type {
                    span: start.to(self.last_span),
                    kind: TypeKind::Named { name, args },
                }
            }
            Tok::LParen => {
                self.bump();
                let mut elems = Vec::new();
                while !self.is(&Tok::RParen) {
                    elems.push(self.parse_type()?);
                    if !self.eat(&Tok::Comma) {
                        break;
                    }
                }
                self.expect(&Tok::RParen)?;
                let span = start.to(self.last_span);
                if elems.is_empty() {
                    Type {
                        span,
                        kind: TypeKind::Named {
                            name: "Unit".into(),
                            args: vec![],
                        },
                    }
                } else {
                    Type {
                        span,
                        kind: TypeKind::Tuple(elems),
                    }
                }
            }
            Tok::Kw(Kw::Fn) => {
                self.bump();
                self.expect(&Tok::LParen)?;
                let mut params = Vec::new();
                while !self.is(&Tok::RParen) {
                    params.push(self.parse_type()?);
                    if !self.eat(&Tok::Comma) {
                        break;
                    }
                }
                self.expect(&Tok::RParen)?;
                let ret = if self.eat(&Tok::Arrow) {
                    self.parse_type()?
                } else {
                    Type {
                        span: self.last_span,
                        kind: TypeKind::Named {
                            name: "Unit".into(),
                            args: vec![],
                        },
                    }
                };
                Type {
                    span: start.to(self.last_span),
                    kind: TypeKind::Fn {
                        params,
                        ret: Box::new(ret),
                    },
                }
            }
            _ => return self.unexpected("a type"),
        };
        while self.is(&Tok::Question) && !self.tok().nl_before {
            self.bump();
            ty = Type {
                span: start.to(self.last_span),
                kind: TypeKind::Optional(Box::new(ty)),
            };
        }
        Ok(ty)
    }

    // ----- blocks and statements ----------------------------------------

    fn parse_block(&mut self) -> PResult<Block> {
        let open = self.expect(&Tok::LBrace)?;
        let saved = self.depth;
        self.depth = 0;
        let mut stmts = Vec::new();
        let mut tail = None;
        loop {
            if self.is(&Tok::RBrace) || self.at_eof() {
                break;
            }
            if self.eat(&Tok::Semi) {
                continue;
            }
            let stmt = self.parse_stmt()?;
            match stmt.kind {
                StmtKind::Expr(e) => {
                    if self.is(&Tok::RBrace) {
                        tail = Some(Box::new(e));
                        break;
                    }
                    self.end_of_statement()?;
                    stmts.push(Stmt {
                        span: stmt.span,
                        kind: StmtKind::Expr(e),
                    });
                }
                other => {
                    if !self.is(&Tok::RBrace) {
                        self.end_of_statement()?;
                    }
                    stmts.push(Stmt {
                        span: stmt.span,
                        kind: other,
                    });
                }
            }
        }
        self.depth = saved;
        let close = self.expect(&Tok::RBrace)?;
        Ok(Block {
            span: open.span.to(close.span),
            stmts,
            tail,
        })
    }

    fn end_of_statement(&mut self) -> PResult<()> {
        if self.eat(&Tok::Semi) || self.tok().nl_before {
            Ok(())
        } else {
            self.unexpected("a newline or `}` after the statement")
        }
    }

    fn parse_stmt(&mut self) -> PResult<Stmt> {
        let start = self.tok().span;
        match self.kind().clone() {
            Tok::Kw(k @ (Kw::Let | Kw::Var | Kw::Const)) => {
                self.bump();
                if self.is(&Tok::LParen) {
                    return Err(Diagnostic::error(
                        self.tok().span,
                        "destructuring patterns in `let` are not supported yet",
                    ));
                }
                let (name, name_span) = self.expect_ident("a variable name")?;
                let ty = if self.eat(&Tok::Colon) {
                    Some(self.parse_type()?)
                } else {
                    None
                };
                self.expect(&Tok::Eq)?;
                let init = self.parse_expr()?;
                Ok(Stmt {
                    span: start.to(self.last_span),
                    kind: StmtKind::Let {
                        name,
                        name_span,
                        mutable: k == Kw::Var,
                        ty,
                        init,
                    },
                })
            }
            _ => {
                let e = self.parse_expr()?;
                Ok(Stmt {
                    span: e.span,
                    kind: StmtKind::Expr(e),
                })
            }
        }
    }

    // ----- expressions ---------------------------------------------------

    fn parse_expr(&mut self) -> PResult<Expr> {
        let lhs = self.parse_binary(1)?;
        let op = match self.kind() {
            Tok::Eq => Some(None),
            Tok::PlusEq => Some(Some(BinOp::Add)),
            Tok::MinusEq => Some(Some(BinOp::Sub)),
            Tok::StarEq => Some(Some(BinOp::Mul)),
            Tok::SlashEq => Some(Some(BinOp::Div)),
            Tok::PercentEq => Some(Some(BinOp::Rem)),
            _ => None,
        };
        if let Some(op) = op {
            self.bump();
            let value = self.parse_expr()?;
            let span = lhs.span.to(value.span);
            return Ok(self.mk(
                span,
                ExprKind::Assign {
                    op,
                    target: Box::new(lhs),
                    value: Box::new(value),
                },
            ));
        }
        Ok(lhs)
    }

    fn binop_info(&self) -> Option<(u8, Tok)> {
        let k = self.kind().clone();
        let prec = match k {
            Tok::OrOr => 1,
            Tok::AndAnd => 2,
            Tok::EqEq | Tok::NotEq | Tok::Lt | Tok::Le | Tok::Gt | Tok::Ge => 3,
            Tok::QuestionQuestion => 4,
            Tok::DotDot | Tok::DotDotEq => 5,
            Tok::Plus | Tok::Minus => 6,
            Tok::Star | Tok::Slash | Tok::Percent => 7,
            _ => return None,
        };
        Some((prec, k))
    }

    fn parse_binary(&mut self, min_prec: u8) -> PResult<Expr> {
        let mut lhs = self.parse_unary()?;
        loop {
            let Some((prec, tok)) = self.binop_info() else { break };
            if prec < min_prec {
                break;
            }
            if self.tok().nl_before && self.depth == 0 {
                break;
            }
            self.bump();
            let rhs = self.parse_binary(prec + 1)?;
            let span = lhs.span.to(rhs.span);
            let kind = match tok {
                Tok::DotDot | Tok::DotDotEq => ExprKind::Range {
                    lo: Box::new(lhs),
                    hi: Box::new(rhs),
                    inclusive: tok == Tok::DotDotEq,
                },
                _ => {
                    let op = match tok {
                        Tok::OrOr => BinOp::Or,
                        Tok::AndAnd => BinOp::And,
                        Tok::EqEq => BinOp::Eq,
                        Tok::NotEq => BinOp::Ne,
                        Tok::Lt => BinOp::Lt,
                        Tok::Le => BinOp::Le,
                        Tok::Gt => BinOp::Gt,
                        Tok::Ge => BinOp::Ge,
                        Tok::QuestionQuestion => BinOp::Coalesce,
                        Tok::Plus => BinOp::Add,
                        Tok::Minus => BinOp::Sub,
                        Tok::Star => BinOp::Mul,
                        Tok::Slash => BinOp::Div,
                        _ => BinOp::Rem,
                    };
                    ExprKind::Binary {
                        op,
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    }
                }
            };
            lhs = self.mk(span, kind);
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self) -> PResult<Expr> {
        let start = self.tok().span;
        let op = match self.kind() {
            Tok::Minus => UnOp::Neg,
            Tok::Bang => UnOp::Not,
            _ => return self.parse_postfix(),
        };
        self.bump();
        let inner = self.parse_unary()?;
        let span = start.to(inner.span);
        Ok(self.mk(
            span,
            ExprKind::Unary {
                op,
                expr: Box::new(inner),
            },
        ))
    }

    fn newline_blocks(&self) -> bool {
        self.tok().nl_before && self.depth == 0
    }

    fn parse_postfix(&mut self) -> PResult<Expr> {
        let mut e = self.parse_primary()?;
        loop {
            match self.kind().clone() {
                Tok::Dot | Tok::QuestionDot => {
                    let optional = self.is(&Tok::QuestionDot);
                    self.bump();
                    if !optional && self.is_kw(Kw::Await) {
                        let t = self.bump();
                        let span = e.span.to(t.span);
                        e = self.mk(span, ExprKind::Await(Box::new(e)));
                        continue;
                    }
                    if let Tok::Int(_) = self.kind() {
                        return Err(Diagnostic::error(self.tok().span, "tuple indexing is not supported yet"));
                    }
                    let (name, name_span) = self.expect_ident("a field or method name")?;
                    let span = e.span.to(name_span);
                    let kind = if optional {
                        ExprKind::OptField {
                            obj: Box::new(e),
                            name,
                            name_span,
                        }
                    } else {
                        ExprKind::Field {
                            obj: Box::new(e),
                            name,
                            name_span,
                        }
                    };
                    e = self.mk(span, kind);
                }
                Tok::LParen if !self.newline_blocks() => {
                    self.bump();
                    self.depth += 1;
                    let mut args = Vec::new();
                    while !self.is(&Tok::RParen) {
                        let named = matches!(self.kind(), Tok::Ident(_)) && matches!(self.peek_kind(1), Tok::Colon);
                        let name = if named {
                            let (n, _) = self.expect_ident("an argument name")?;
                            self.expect(&Tok::Colon)?;
                            Some(n)
                        } else {
                            None
                        };
                        let value = self.parse_expr()?;
                        args.push(Arg { name, value });
                        if !self.eat(&Tok::Comma) {
                            break;
                        }
                    }
                    self.depth -= 1;
                    let close = self.expect(&Tok::RParen)?;
                    let span = e.span.to(close.span);
                    e = self.mk(
                        span,
                        ExprKind::Call {
                            callee: Box::new(e),
                            args,
                        },
                    );
                }
                Tok::LBracket if !self.newline_blocks() => {
                    self.bump();
                    self.depth += 1;
                    let index = self.parse_expr()?;
                    self.depth -= 1;
                    let close = self.expect(&Tok::RBracket)?;
                    let span = e.span.to(close.span);
                    e = self.mk(
                        span,
                        ExprKind::Index {
                            obj: Box::new(e),
                            index: Box::new(index),
                        },
                    );
                }
                Tok::Question if !self.tok().nl_before => {
                    let t = self.bump();
                    let span = e.span.to(t.span);
                    e = self.mk(span, ExprKind::Try(Box::new(e)));
                }
                Tok::Bang if !self.tok().nl_before && !self.tok().space_before => {
                    let t = self.bump();
                    let span = e.span.to(t.span);
                    e = self.mk(span, ExprKind::Unwrap(Box::new(e)));
                }
                Tok::Kw(Kw::With) if !self.newline_blocks() => {
                    self.bump();
                    self.expect(&Tok::LBrace)?;
                    let saved = self.depth;
                    self.depth = 0;
                    let mut fields = Vec::new();
                    loop {
                        while self.eat(&Tok::Comma) {}
                        if self.is(&Tok::RBrace) || self.at_eof() {
                            break;
                        }
                        let (fname, fspan) = self.expect_ident("a field name")?;
                        self.expect(&Tok::Colon)?;
                        let value = self.parse_expr()?;
                        fields.push((fname, fspan, value));
                    }
                    self.depth = saved;
                    let close = self.expect(&Tok::RBrace)?;
                    let span = e.span.to(close.span);
                    e = self.mk(
                        span,
                        ExprKind::With {
                            base: Box::new(e),
                            fields,
                        },
                    );
                }
                _ => break,
            }
        }
        Ok(e)
    }

    fn parse_primary(&mut self) -> PResult<Expr> {
        let t = self.tok().clone();
        let start = t.span;
        match t.kind {
            Tok::Int(n) => {
                self.bump();
                Ok(self.mk(start, ExprKind::Int(n)))
            }
            Tok::Float(f) => {
                self.bump();
                Ok(self.mk(start, ExprKind::Float(f)))
            }
            Tok::Kw(Kw::True) => {
                self.bump();
                Ok(self.mk(start, ExprKind::Bool(true)))
            }
            Tok::Kw(Kw::False) => {
                self.bump();
                Ok(self.mk(start, ExprKind::Bool(false)))
            }
            Tok::Kw(Kw::None) => {
                self.bump();
                Ok(self.mk(start, ExprKind::None))
            }
            Tok::Str(pieces) => {
                self.bump();
                self.string_expr(pieces, start)
            }
            Tok::Ident(name) => {
                self.bump();
                Ok(self.mk(start, ExprKind::Ident(name)))
            }
            Tok::Kw(Kw::SelfValue) => {
                self.bump();
                Ok(self.mk(start, ExprKind::Ident("self".into())))
            }
            Tok::LParen => {
                self.bump();
                self.depth += 1;
                if self.eat(&Tok::RParen) {
                    self.depth -= 1;
                    let span = start.to(self.last_span);
                    return Ok(self.mk(span, ExprKind::Tuple(vec![])));
                }
                let first = self.parse_expr()?;
                if self.is(&Tok::Comma) {
                    let mut elems = vec![first];
                    while self.eat(&Tok::Comma) {
                        if self.is(&Tok::RParen) {
                            break;
                        }
                        elems.push(self.parse_expr()?);
                    }
                    self.depth -= 1;
                    self.expect(&Tok::RParen)?;
                    let span = start.to(self.last_span);
                    return Ok(self.mk(span, ExprKind::Tuple(elems)));
                }
                self.depth -= 1;
                self.expect(&Tok::RParen)?;
                Ok(first)
            }
            Tok::LBracket => {
                self.bump();
                self.depth += 1;
                let mut elems = Vec::new();
                while !self.is(&Tok::RBracket) {
                    elems.push(self.parse_expr()?);
                    if !self.eat(&Tok::Comma) {
                        break;
                    }
                }
                self.depth -= 1;
                self.expect(&Tok::RBracket)?;
                let span = start.to(self.last_span);
                Ok(self.mk(span, ExprKind::List(elems)))
            }
            Tok::LBrace => {
                let b = self.parse_block()?;
                Ok(self.mk(b.span, ExprKind::Block(b)))
            }
            Tok::Kw(Kw::If) => self.parse_if(),
            Tok::Kw(Kw::Match) => self.parse_match(),
            Tok::Kw(Kw::While) => {
                self.bump();
                if self.is_kw(Kw::Let) {
                    return Err(Diagnostic::error(self.tok().span, "`while let` is not supported yet")
                        .with_help("use `loop` with `match` or `if let` and `break`"));
                }
                let cond = self.parse_expr()?;
                let body = self.parse_block()?;
                let span = start.to(body.span);
                Ok(self.mk(
                    span,
                    ExprKind::While {
                        cond: Box::new(cond),
                        body,
                    },
                ))
            }
            Tok::Kw(Kw::Loop) => {
                self.bump();
                let body = self.parse_block()?;
                let span = start.to(body.span);
                Ok(self.mk(span, ExprKind::Loop(body)))
            }
            Tok::Kw(Kw::For) => {
                self.bump();
                let (var, var_span) = self.expect_ident("a loop variable name")?;
                self.expect_kw(Kw::In)?;
                let iter = self.parse_expr()?;
                let body = self.parse_block()?;
                let span = start.to(body.span);
                Ok(self.mk(
                    span,
                    ExprKind::For {
                        var,
                        var_span,
                        iter: Box::new(iter),
                        body,
                    },
                ))
            }
            Tok::Kw(Kw::Return) => {
                self.bump();
                let has_value = !(self.tok().nl_before
                    || matches!(
                        self.kind(),
                        Tok::RBrace | Tok::RParen | Tok::RBracket | Tok::Comma | Tok::Semi | Tok::Eof | Tok::FatArrow
                    ));
                let value = if has_value {
                    Some(Box::new(self.parse_expr()?))
                } else {
                    None
                };
                let span = start.to(self.last_span);
                Ok(self.mk(span, ExprKind::Return(value)))
            }
            Tok::Kw(Kw::Break) => {
                self.bump();
                Ok(self.mk(start, ExprKind::Break))
            }
            Tok::Kw(Kw::Continue) => {
                self.bump();
                Ok(self.mk(start, ExprKind::Continue))
            }
            Tok::Pipe | Tok::OrOr | Tok::Kw(Kw::Move) => self.parse_closure(),
            _ => self.unexpected("an expression"),
        }
    }

    fn parse_closure(&mut self) -> PResult<Expr> {
        let start = self.tok().span;
        let is_move = self.eat_kw(Kw::Move);
        let mut params = Vec::new();
        if !self.eat(&Tok::OrOr) {
            self.expect(&Tok::Pipe)?;
            while !self.is(&Tok::Pipe) {
                let (name, _) = self.expect_ident("a parameter name")?;
                let ty = if self.eat(&Tok::Colon) {
                    Some(self.parse_type()?)
                } else {
                    None
                };
                params.push((name, ty));
                if !self.eat(&Tok::Comma) {
                    break;
                }
            }
            self.expect(&Tok::Pipe)?;
        }
        let body = self.parse_expr()?;
        let span = start.to(body.span);
        Ok(self.mk(
            span,
            ExprKind::Closure {
                params,
                body: Box::new(body),
                is_move,
            },
        ))
    }

    fn parse_if(&mut self) -> PResult<Expr> {
        let start = self.expect_kw(Kw::If)?.span;
        if self.eat_kw(Kw::Let) {
            let pat = self.parse_pattern()?;
            self.expect(&Tok::Eq)?;
            let value = self.parse_expr()?;
            let then = self.parse_block()?;
            let els = self.parse_else()?;
            let span = start.to(self.last_span);
            return Ok(self.mk(
                span,
                ExprKind::IfLet {
                    pat,
                    value: Box::new(value),
                    then,
                    els,
                },
            ));
        }
        let cond = self.parse_expr()?;
        let then = self.parse_block()?;
        let els = self.parse_else()?;
        let span = start.to(self.last_span);
        Ok(self.mk(
            span,
            ExprKind::If {
                cond: Box::new(cond),
                then,
                els,
            },
        ))
    }

    fn parse_else(&mut self) -> PResult<Option<Box<Expr>>> {
        if !self.eat_kw(Kw::Else) {
            return Ok(None);
        }
        if self.is_kw(Kw::If) {
            return Ok(Some(Box::new(self.parse_if()?)));
        }
        let b = self.parse_block()?;
        Ok(Some(Box::new(self.mk(b.span, ExprKind::Block(b)))))
    }

    fn parse_match(&mut self) -> PResult<Expr> {
        let start = self.expect_kw(Kw::Match)?.span;
        let scrutinee = self.parse_expr()?;
        self.expect(&Tok::LBrace)?;
        let saved = self.depth;
        self.depth = 0;
        let mut arms = Vec::new();
        loop {
            while self.eat(&Tok::Comma) {}
            if self.is(&Tok::RBrace) || self.at_eof() {
                break;
            }
            let pat = self.parse_pattern()?;
            let guard = if self.eat_kw(Kw::If) {
                Some(self.parse_expr()?)
            } else {
                None
            };
            self.expect(&Tok::FatArrow)?;
            let body = self.parse_expr()?;
            arms.push(MatchArm { pat, guard, body });
        }
        self.depth = saved;
        let close = self.expect(&Tok::RBrace)?;
        let span = start.to(close.span);
        Ok(self.mk(
            span,
            ExprKind::Match {
                scrutinee: Box::new(scrutinee),
                arms,
            },
        ))
    }

    fn string_expr(&mut self, pieces: Vec<StrPiece>, span: Span) -> PResult<Expr> {
        if pieces.iter().all(|p| matches!(p, StrPiece::Lit(_))) {
            let s: String = pieces
                .into_iter()
                .map(|p| match p {
                    StrPiece::Lit(s) => s,
                    _ => String::new(),
                })
                .collect();
            return Ok(self.mk(span, ExprKind::Str(s)));
        }
        let mut parts = Vec::new();
        for piece in pieces {
            match piece {
                StrPiece::Lit(s) => parts.push(StrPart::Lit(s)),
                StrPiece::Expr { src, offset } => {
                    let lexed = lex(&src, offset);
                    self.diags.extend(lexed.diags);
                    let mut sub = Parser::new(lexed.tokens, self.ids.clone());
                    let e = sub.parse_expr();
                    let e = match e {
                        Ok(e) => e,
                        Err(d) => {
                            self.diags.extend(sub.diags);
                            return Err(d);
                        }
                    };
                    if !sub.at_eof() {
                        let d = sub.unexpected::<()>("`}` to end the interpolation").unwrap_err();
                        return Err(d);
                    }
                    self.diags.extend(sub.diags);
                    parts.push(StrPart::Expr(e));
                }
            }
        }
        Ok(self.mk(span, ExprKind::Interp(parts)))
    }

    // ----- patterns ------------------------------------------------------

    fn parse_pattern(&mut self) -> PResult<Pattern> {
        let first = self.parse_pattern_atom()?;
        if !self.is(&Tok::Pipe) {
            return Ok(first);
        }
        let start = first.span;
        let mut alts = vec![first];
        while self.eat(&Tok::Pipe) {
            alts.push(self.parse_pattern_atom()?);
        }
        let span = start.to(self.last_span);
        Ok(Pattern {
            id: self.new_id(),
            span,
            kind: PatKind::Or(alts),
        })
    }

    fn pat(&mut self, span: Span, kind: PatKind) -> Pattern {
        Pattern {
            id: self.new_id(),
            span,
            kind,
        }
    }

    fn parse_pattern_args(&mut self) -> PResult<Vec<Pattern>> {
        self.expect(&Tok::LParen)?;
        let mut args = Vec::new();
        while !self.is(&Tok::RParen) {
            if matches!(self.kind(), Tok::Ident(_)) && matches!(self.peek_kind(1), Tok::Colon) {
                return Err(Diagnostic::error(
                    self.tok().span,
                    "named fields in patterns are not supported yet; match payloads by position",
                ));
            }
            args.push(self.parse_pattern()?);
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        self.expect(&Tok::RParen)?;
        Ok(args)
    }

    fn parse_pattern_atom(&mut self) -> PResult<Pattern> {
        let t = self.tok().clone();
        let start = t.span;
        match t.kind {
            Tok::Ident(name) if name == "_" => {
                self.bump();
                Ok(self.pat(start, PatKind::Wild))
            }
            Tok::Ident(name) => {
                self.bump();
                if self.is(&Tok::Dot) {
                    let mut path = vec![name];
                    while self.eat(&Tok::Dot) {
                        path.push(self.expect_ident("a variant name")?.0);
                    }
                    let args = if self.is(&Tok::LParen) {
                        self.parse_pattern_args()?
                    } else {
                        vec![]
                    };
                    let span = start.to(self.last_span);
                    return Ok(self.pat(span, PatKind::Variant { path, args }));
                }
                if self.is(&Tok::LParen) && !self.tok().nl_before {
                    let args = self.parse_pattern_args()?;
                    let span = start.to(self.last_span);
                    return Ok(self.pat(
                        span,
                        PatKind::Variant {
                            path: vec![name],
                            args,
                        },
                    ));
                }
                Ok(self.pat(start, PatKind::Ident(name)))
            }
            Tok::Kw(Kw::None) => {
                self.bump();
                Ok(self.pat(start, PatKind::None))
            }
            Tok::Kw(Kw::True) => {
                self.bump();
                Ok(self.pat(start, PatKind::Bool(true)))
            }
            Tok::Kw(Kw::False) => {
                self.bump();
                Ok(self.pat(start, PatKind::Bool(false)))
            }
            Tok::Float(f) => {
                self.bump();
                Ok(self.pat(start, PatKind::Float(f)))
            }
            Tok::Str(pieces) => {
                self.bump();
                let mut s = String::new();
                for p in pieces {
                    match p {
                        StrPiece::Lit(l) => s.push_str(&l),
                        StrPiece::Expr { .. } => {
                            return Err(Diagnostic::error(start, "string patterns cannot contain `{}` interpolation"))
                        }
                    }
                }
                Ok(self.pat(start, PatKind::Str(s)))
            }
            Tok::Int(_) | Tok::Minus => {
                let lo = self.parse_pat_int()?;
                if self.is(&Tok::DotDot) || self.is(&Tok::DotDotEq) {
                    let inclusive = self.is(&Tok::DotDotEq);
                    self.bump();
                    let hi = self.parse_pat_int()?;
                    let span = start.to(self.last_span);
                    return Ok(self.pat(span, PatKind::Range { lo, hi, inclusive }));
                }
                let span = start.to(self.last_span);
                Ok(self.pat(span, PatKind::Int(lo)))
            }
            Tok::LParen => {
                self.bump();
                let mut elems = Vec::new();
                while !self.is(&Tok::RParen) {
                    elems.push(self.parse_pattern()?);
                    if !self.eat(&Tok::Comma) {
                        break;
                    }
                }
                self.expect(&Tok::RParen)?;
                let span = start.to(self.last_span);
                Ok(self.pat(span, PatKind::Tuple(elems)))
            }
            _ => self.unexpected("a pattern"),
        }
    }

    fn parse_pat_int(&mut self) -> PResult<i64> {
        let neg = self.eat(&Tok::Minus);
        match self.kind().clone() {
            Tok::Int(n) => {
                self.bump();
                Ok(if neg { -n } else { n })
            }
            _ => self.unexpected("an integer"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok(src: &str) -> Program {
        let r = parse(src);
        assert!(r.diags.is_empty(), "unexpected diagnostics: {:?}", r.diags);
        r.program
    }

    #[test]
    fn parses_function_and_statements() {
        let p = ok("fn add(a: Int, b: Int = 2) -> Int {\n    let x = a + b\n    var y = x * 2\n    y += 1\n    y\n}\n");
        assert_eq!(p.items.len(), 1);
    }

    #[test]
    fn parses_types_and_matches() {
        ok(r#"
enum Shape {
    Circle(radius: Float)
    Rect(width: Float, height: Float)
    Point
}
struct User {
    id: Int
    name: Text
    email: Text?
}
impl User {
    fn display_name(self) -> Text { self.name }
    fn rename(mut self, n: Text) { self.name = n }
}
fn area(s: Shape) -> Float {
    match s {
        Circle(r) => 3.14 * r * r
        Rect(w, h) if w == h => w * w
        Rect(w, h) => w * h
        Point => 0.0
    }
}
"#);
    }

    #[test]
    fn parses_control_flow() {
        ok(r#"
fn main() {
    for i in 0..10 { print("{i}") }
    var n = 0
    while n < 3 { n += 1 }
    loop { break }
    if let e = x { print(e) } else { print("none") }
    let v = x ?? 5
    let w = u with { name: "a" }
    let chained = items
        .iter()
        .len()
}
"#);
    }

    #[test]
    fn parses_generics_traits_closures() {
        ok(r#"
trait Measurable {
    fn size(self) -> Float
    fn describe(self) -> Text { "size" }
}
fn largest<T: Measurable>(items: List<T>) -> T? { none }
async fn fetch(url: Text) -> Result<Text, Text> { none }
fn f() { let g = |x| x * 2 }
test "adds" { assert(1 + 1 == 2) }
@derive(Json)
struct P { x: Int, y: Int }
"#);
    }

    #[test]
    fn newline_ends_statement() {
        let p = ok("fn f() {\n let a = 1\n -a\n}");
        match &p.items[0].kind {
            ItemKind::Fn(f) => {
                let b = f.body.as_ref().unwrap();
                assert_eq!(b.stmts.len(), 1);
                assert!(b.tail.is_some());
            }
            _ => panic!(),
        }
    }

    #[test]
    fn reports_error_and_recovers() {
        let r = parse("fn a( {\nfn b() { }\n");
        assert!(!r.diags.is_empty());
        assert!(r.program.items.len() >= 1);
    }
}
