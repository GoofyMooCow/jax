//! The Jax abstract syntax tree.
//!
//! The parser understands the full surface syntax from the language
//! specification (generics, traits, closures, async, ...). Later stages
//! report a clear "not supported yet" error for the parts that are not
//! implemented in this release.

use crate::diag::Span;

/// Unique id of an expression or pattern; the type checker keys its side
/// tables on it.
pub type NodeId = u32;

#[derive(Debug, Clone, Default)]
pub struct Program {
    pub items: Vec<Item>,
}

#[derive(Debug, Clone)]
pub struct Attr {
    pub name: String,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct Item {
    pub attrs: Vec<Attr>,
    pub is_pub: bool,
    pub span: Span,
    pub kind: ItemKind,
}

#[derive(Debug, Clone)]
pub enum ItemKind {
    Fn(FnDecl),
    Struct(StructDecl),
    Enum(EnumDecl),
    Impl(ImplDecl),
    Trait(TraitDecl),
    Use(UseDecl),
    Const(ConstDecl),
    Test(TestDecl),
}

#[derive(Debug, Clone)]
pub struct GenericParam {
    pub name: String,
    pub bounds: Vec<String>,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelfKind {
    /// `self`: read-only receiver.
    Value,
    /// `mut self`: the method may modify the receiver.
    Mut,
}

#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub span: Span,
    pub ty: Type,
    pub default: Option<Expr>,
}

#[derive(Debug, Clone)]
pub struct FnDecl {
    pub name: String,
    pub name_span: Span,
    pub is_async: bool,
    pub generics: Vec<GenericParam>,
    pub self_param: Option<SelfKind>,
    pub params: Vec<Param>,
    pub ret: Option<Type>,
    /// `None` for signatures inside traits.
    pub body: Option<Block>,
}

#[derive(Debug, Clone)]
pub struct FieldDecl {
    pub name: String,
    pub span: Span,
    pub ty: Type,
    pub is_pub: bool,
}

#[derive(Debug, Clone)]
pub struct StructDecl {
    pub name: String,
    pub name_span: Span,
    pub generics: Vec<GenericParam>,
    pub fields: Vec<FieldDecl>,
}

#[derive(Debug, Clone)]
pub struct VariantField {
    pub name: Option<String>,
    pub ty: Type,
}

#[derive(Debug, Clone)]
pub struct Variant {
    pub name: String,
    pub span: Span,
    pub fields: Vec<VariantField>,
}

#[derive(Debug, Clone)]
pub struct EnumDecl {
    pub name: String,
    pub name_span: Span,
    pub generics: Vec<GenericParam>,
    pub variants: Vec<Variant>,
}

#[derive(Debug, Clone)]
pub struct ImplDecl {
    pub generics: Vec<GenericParam>,
    /// `Some(trait)` for `impl Trait for Type`.
    pub trait_ty: Option<Type>,
    pub self_ty: Type,
    pub items: Vec<Item>,
}

#[derive(Debug, Clone)]
pub struct TraitDecl {
    pub name: String,
    pub name_span: Span,
    pub generics: Vec<GenericParam>,
    pub items: Vec<Item>,
}

#[derive(Debug, Clone)]
pub struct UseDecl {
    pub path: Vec<String>,
    pub alias: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ConstDecl {
    pub name: String,
    pub name_span: Span,
    pub ty: Option<Type>,
    pub value: Expr,
}

#[derive(Debug, Clone)]
pub struct TestDecl {
    pub name: String,
    pub body: Block,
}

#[derive(Debug, Clone)]
pub struct Type {
    pub span: Span,
    pub kind: TypeKind,
}

#[derive(Debug, Clone)]
pub enum TypeKind {
    Named { name: String, args: Vec<Type> },
    Optional(Box<Type>),
    Tuple(Vec<Type>),
    Fn { params: Vec<Type>, ret: Box<Type> },
    Dyn(String),
}

#[derive(Debug, Clone)]
pub struct Block {
    pub span: Span,
    pub stmts: Vec<Stmt>,
    pub tail: Option<Box<Expr>>,
}

#[derive(Debug, Clone)]
pub struct Stmt {
    pub span: Span,
    pub kind: StmtKind,
}

#[derive(Debug, Clone)]
pub enum StmtKind {
    Let {
        name: String,
        name_span: Span,
        mutable: bool,
        ty: Option<Type>,
        init: Expr,
    },
    Expr(Expr),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
    Coalesce,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnOp {
    Neg,
    Not,
}

#[derive(Debug, Clone)]
pub struct Arg {
    pub name: Option<String>,
    pub value: Expr,
}

#[derive(Debug, Clone)]
pub enum StrPart {
    Lit(String),
    Expr(Expr),
}

#[derive(Debug, Clone)]
pub struct MatchArm {
    pub pat: Pattern,
    pub guard: Option<Expr>,
    pub body: Expr,
}

#[derive(Debug, Clone)]
pub struct Expr {
    pub id: NodeId,
    pub span: Span,
    pub kind: ExprKind,
}

#[derive(Debug, Clone)]
pub enum ExprKind {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    Interp(Vec<StrPart>),
    None,
    Ident(String),
    Unary {
        op: UnOp,
        expr: Box<Expr>,
    },
    Binary {
        op: BinOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    /// `target = value` (`op == None`) or `target += value` etc.
    Assign {
        op: Option<BinOp>,
        target: Box<Expr>,
        value: Box<Expr>,
    },
    Call {
        callee: Box<Expr>,
        args: Vec<Arg>,
    },
    Field {
        obj: Box<Expr>,
        name: String,
        name_span: Span,
    },
    /// `obj?.name`
    OptField {
        obj: Box<Expr>,
        name: String,
        name_span: Span,
    },
    Index {
        obj: Box<Expr>,
        index: Box<Expr>,
    },
    If {
        cond: Box<Expr>,
        then: Block,
        els: Option<Box<Expr>>,
    },
    IfLet {
        pat: Pattern,
        value: Box<Expr>,
        then: Block,
        els: Option<Box<Expr>>,
    },
    Match {
        scrutinee: Box<Expr>,
        arms: Vec<MatchArm>,
    },
    Block(Block),
    While {
        cond: Box<Expr>,
        body: Block,
    },
    Loop(Block),
    For {
        var: String,
        var_span: Span,
        iter: Box<Expr>,
        body: Block,
    },
    Return(Option<Box<Expr>>),
    Break,
    Continue,
    Range {
        lo: Box<Expr>,
        hi: Box<Expr>,
        inclusive: bool,
    },
    List(Vec<Expr>),
    Tuple(Vec<Expr>),
    Closure {
        params: Vec<(String, Option<Type>)>,
        body: Box<Expr>,
        is_move: bool,
    },
    /// `expr?` (error propagation)
    Try(Box<Expr>),
    /// `expr!` (unwrap an optional or panic)
    Unwrap(Box<Expr>),
    Await(Box<Expr>),
    /// `base with { field: value, ... }`
    With {
        base: Box<Expr>,
        fields: Vec<(String, Span, Expr)>,
    },
}

#[derive(Debug, Clone)]
pub struct Pattern {
    pub id: NodeId,
    pub span: Span,
    pub kind: PatKind,
}

#[derive(Debug, Clone)]
pub enum PatKind {
    Wild,
    /// A bare identifier: a binding, or a unit enum variant of the scrutinee.
    Ident(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    None,
    /// `Name(p, ...)` or `Enum.Name` / `Enum.Name(p, ...)`.
    Variant {
        path: Vec<String>,
        args: Vec<Pattern>,
    },
    Tuple(Vec<Pattern>),
    Range {
        lo: i64,
        hi: i64,
        inclusive: bool,
    },
    Or(Vec<Pattern>),
}
