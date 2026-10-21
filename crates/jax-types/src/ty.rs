//! Types and the side tables the checker hands to code generation.

use std::collections::HashMap;
use std::fmt;

use jax_bytecode::Builtin;
use jax_syntax::ast::NodeId;

#[derive(Clone, Debug, PartialEq)]
pub enum Ty {
    Int,
    Float,
    Bool,
    Text,
    Unit,
    /// The type of expressions that never produce a value (`return`, `break`, `panic`).
    Never,
    /// The type of the literal `none`.
    NoneLit,
    /// The element type of an empty list literal before it is known.
    Unknown,
    /// Poison type used after an error to avoid cascades.
    Error,
    List(Box<Ty>),
    Opt(Box<Ty>),
    Struct(String),
    Enum(String),
}

impl fmt::Display for Ty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Ty::Int => write!(f, "Int"),
            Ty::Float => write!(f, "Float"),
            Ty::Bool => write!(f, "Bool"),
            Ty::Text => write!(f, "Text"),
            Ty::Unit => write!(f, "Unit"),
            Ty::Never => write!(f, "Never"),
            Ty::NoneLit => write!(f, "none"),
            Ty::Unknown => write!(f, "_"),
            Ty::Error => write!(f, "{{error}}"),
            Ty::List(t) => write!(f, "List<{t}>"),
            Ty::Opt(t) => write!(f, "{t}?"),
            Ty::Struct(n) | Ty::Enum(n) => write!(f, "{n}"),
        }
    }
}

/// Where the value for one parameter of a call comes from.
#[derive(Clone, Debug)]
pub enum ParamSource {
    /// The call's `args[i]` (counting only explicit arguments).
    Arg(usize),
    /// The default expression of parameter `index` of function `func`.
    Default { func: String, index: usize },
}

#[derive(Clone, Debug)]
pub enum Target {
    /// A user function or static method, by mangled name.
    Func(String),
    StructNew(String),
    VariantNew { enum_name: String, tag: u32 },
    /// A user method (`Type.name`); the receiver is passed first.
    Method(String),
    Builtin(Builtin),
}

#[derive(Clone, Debug)]
pub struct CallPlan {
    pub target: Target,
    /// One entry per declared parameter, in declaration order.
    pub order: Vec<ParamSource>,
    /// The callee expression is `receiver.method` and the receiver is argument zero.
    pub has_receiver: bool,
    /// The call returns the updated receiver, which must be stored back into it
    /// (`mut self` methods and mutating builtins such as `List.push`).
    pub writeback: bool,
}

/// Name-resolution results keyed by node id.
#[derive(Clone, Debug)]
pub enum Res {
    /// `Ident` that names a constant.
    Const(String),
    /// `Call` expression.
    Call(CallPlan),
    /// `Field` / `OptField`: the field index.
    Field(usize),
    /// `Enum.Variant` used as a value (a variant without payload).
    VariantUnit { tag: u32 },
    /// `With`: field index for each updated field, in source order.
    With(Vec<usize>),
    /// Pattern naming an enum variant.
    PatVariant { tag: u32 },
    /// Pattern that binds a new local.
    PatBind,
    /// `some(p)` pattern on an optional.
    PatSome,
    /// `if let x = optional { ... }` implicitly unwraps the optional.
    IfLetOpt,
}

#[derive(Clone, Debug)]
pub struct StructInfo {
    pub name: String,
    pub fields: Vec<(String, Ty)>,
}

#[derive(Clone, Debug)]
pub struct VariantInfo {
    pub name: String,
    pub fields: Vec<(Option<String>, Ty)>,
}

#[derive(Clone, Debug)]
pub struct EnumInfo {
    pub name: String,
    pub variants: Vec<VariantInfo>,
}

#[derive(Debug, Default)]
pub struct TypeInfo {
    pub expr_types: HashMap<NodeId, Ty>,
    pub res: HashMap<NodeId, Res>,
    pub structs: HashMap<String, StructInfo>,
    pub enums: HashMap<String, EnumInfo>,
    /// The program defines a valid `fn main()`.
    pub has_main: bool,
}
