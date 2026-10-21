//! Type checking and name resolution.
//!
//! The checker walks the AST once, reports diagnostics and fills a
//! [`TypeInfo`] with the facts code generation needs (call targets, field
//! indices, enum tags, argument order, ...).
//!
//! Supported in this release: functions with named/default arguments,
//! `let`/`var`/`const`, `Int`/`Float`/`Bool`/`Text`/`Unit`, optionals,
//! lists, structs (with methods and `mut self`), enums, `match` with
//! exhaustiveness checking, `if let`, loops and the core builtins.
//! Generics, trait bounds, closures, `Result`/`?`, async, maps, tuples and
//! multi-file modules are parsed but rejected here with an explicit
//! "not supported yet" error (code E0900).

use std::collections::{HashMap, HashSet};

use jax_bytecode::Builtin;
use jax_syntax::ast::*;
use jax_syntax::{suggest, Diagnostic, Span};

use crate::ty::*;

pub struct CheckOutput {
    pub info: TypeInfo,
    pub diags: Vec<Diagnostic>,
}

pub fn check(program: &Program) -> CheckOutput {
    let mut c = Checker {
        program,
        info: TypeInfo::default(),
        diags: Vec::new(),
        fns: HashMap::new(),
        methods: HashMap::new(),
        consts: HashMap::new(),
        scopes: Vec::new(),
        ret_ty: Ty::Unit,
        mut_self_method: false,
        loop_breaks: Vec::new(),
    };
    c.run();
    CheckOutput {
        info: c.info,
        diags: c.diags,
    }
}

#[derive(Clone)]
struct ParamSig {
    name: String,
    ty: Ty,
    has_default: bool,
}

#[derive(Clone)]
struct FnSig {
    params: Vec<ParamSig>,
    ret: Ty,
}

#[derive(Clone)]
struct MethodSig {
    sig: FnSig,
    self_kind: Option<SelfKind>,
    mangled: String,
}

struct Local {
    ty: Ty,
    mutable: bool,
}

struct Checker<'a> {
    program: &'a Program,
    info: TypeInfo,
    diags: Vec<Diagnostic>,
    fns: HashMap<String, FnSig>,
    methods: HashMap<(String, String), MethodSig>,
    consts: HashMap<String, Ty>,
    scopes: Vec<HashMap<String, Local>>,
    ret_ty: Ty,
    mut_self_method: bool,
    loop_breaks: Vec<bool>,
}

const BUILTIN_FNS: [&str; 3] = ["print", "panic", "assert"];

/// Is a value of type `actual` acceptable where `expected` is required?
fn assignable(expected: &Ty, actual: &Ty) -> bool {
    if expected == actual || *actual == Ty::Never || *expected == Ty::Error || *actual == Ty::Error {
        return true;
    }
    match (expected, actual) {
        (Ty::Opt(_), Ty::NoneLit) => true,
        (Ty::Opt(inner), other) => assignable(inner, other),
        (Ty::List(e), Ty::List(a)) => **a == Ty::Unknown || **e == Ty::Unknown || assignable(e, a),
        _ => false,
    }
}

/// The common type of two branches, if there is one.
fn join(a: &Ty, b: &Ty) -> Option<Ty> {
    if *a == Ty::Never {
        return Some(b.clone());
    }
    if *b == Ty::Never {
        return Some(a.clone());
    }
    if *a == Ty::Error || *b == Ty::Error {
        return Some(Ty::Error);
    }
    if a == b {
        return Some(a.clone());
    }
    match (a, b) {
        (Ty::NoneLit, Ty::Opt(_)) => Some(b.clone()),
        (Ty::Opt(_), Ty::NoneLit) => Some(a.clone()),
        (Ty::NoneLit, t) | (t, Ty::NoneLit) => Some(Ty::Opt(Box::new(t.clone()))),
        (Ty::Opt(inner), t) | (t, Ty::Opt(inner)) if **inner == *t => Some(Ty::Opt(inner.clone())),
        (Ty::List(x), Ty::List(y)) => {
            if **x == Ty::Unknown {
                Some(b.clone())
            } else if **y == Ty::Unknown {
                Some(a.clone())
            } else {
                join(x, y).map(|t| Ty::List(Box::new(t)))
            }
        }
        _ => None,
    }
}

fn displayable(t: &Ty) -> bool {
    matches!(t, Ty::Int | Ty::Float | Ty::Bool | Ty::Text | Ty::Error)
}

fn op_symbol(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::Rem => "%",
        BinOp::Eq => "==",
        BinOp::Ne => "!=",
        BinOp::Lt => "<",
        BinOp::Le => "<=",
        BinOp::Gt => ">",
        BinOp::Ge => ">=",
        BinOp::And => "&&",
        BinOp::Or => "||",
        BinOp::Coalesce => "??",
    }
}

impl<'a> Checker<'a> {
    // ----- diagnostics ---------------------------------------------------

    fn error(&mut self, span: Span, msg: impl Into<String>) {
        self.diags.push(Diagnostic::error(span, msg).with_code("E0200"));
    }

    fn error_help(&mut self, span: Span, msg: impl Into<String>, help: impl Into<String>) {
        self.diags
            .push(Diagnostic::error(span, msg).with_code("E0200").with_help(help));
    }

    fn unsupported(&mut self, span: Span, what: &str) {
        self.diags.push(
            Diagnostic::error(span, format!("{what} is not supported yet"))
                .with_code("E0900")
                .with_help("this part of Jax is parsed but not implemented in this release; see docs/STATUS.md"),
        );
    }

    fn mismatch(&mut self, span: Span, expected: &Ty, found: &Ty) {
        if *expected == Ty::Error || *found == Ty::Error {
            return;
        }
        let help = match (expected, found) {
            (Ty::Float, Ty::Int) => Some("convert it with `.to_float()`"),
            (Ty::Int, Ty::Float) => Some("convert it with `.to_int()`"),
            (Ty::Text, Ty::Int | Ty::Float | Ty::Bool) => Some("convert it with `.to_text()`"),
            (t, f) if !matches!(t, Ty::Opt(_)) && matches!(f, Ty::Opt(_)) => {
                Some("this value may be `none`; use `??` to supply a default, or `!` to unwrap it")
            }
            _ => None,
        };
        let msg = format!("type mismatch: expected `{expected}`, found `{found}`");
        match help {
            Some(h) => self.error_help(span, msg, h),
            None => self.error(span, msg),
        }
    }

    fn rec(&mut self, id: NodeId, res: Res) {
        self.info.res.insert(id, res);
    }

    // ----- scopes --------------------------------------------------------

    fn lookup(&self, name: &str) -> Option<&Local> {
        self.scopes.iter().rev().find_map(|s| s.get(name))
    }

    fn declare(&mut self, name: &str, ty: Ty, mutable: bool) {
        if let Some(s) = self.scopes.last_mut() {
            s.insert(name.to_string(), Local { ty, mutable });
        }
    }

    fn is_type_name(&self, name: &str) -> bool {
        self.info.structs.contains_key(name) || self.info.enums.contains_key(name)
    }

    // ----- driver --------------------------------------------------------

    fn run(&mut self) {
        let program = self.program;
        self.collect_types(program);
        self.collect_fns(program);
        self.collect_consts(program);
        self.check_bodies(program);
        self.check_main(program);
    }

    fn collect_types(&mut self, program: &'a Program) {
        // First register every name so types can refer to each other.
        let mut seen: HashSet<&str> = HashSet::new();
        for item in &program.items {
            match &item.kind {
                ItemKind::Struct(s) => {
                    if !seen.insert(&s.name) {
                        self.error(s.name_span, format!("`{}` is defined more than once", s.name));
                        continue;
                    }
                    if !s.generics.is_empty() {
                        self.unsupported(s.name_span, "generic structs");
                    }
                    self.info.structs.insert(
                        s.name.clone(),
                        StructInfo {
                            name: s.name.clone(),
                            fields: Vec::new(),
                        },
                    );
                }
                ItemKind::Enum(e) => {
                    if !seen.insert(&e.name) {
                        self.error(e.name_span, format!("`{}` is defined more than once", e.name));
                        continue;
                    }
                    if !e.generics.is_empty() {
                        self.unsupported(e.name_span, "generic enums");
                    }
                    self.info.enums.insert(
                        e.name.clone(),
                        EnumInfo {
                            name: e.name.clone(),
                            variants: Vec::new(),
                        },
                    );
                }
                _ => {}
            }
        }
        // Then resolve field and payload types.
        for item in &program.items {
            match &item.kind {
                ItemKind::Struct(s) => {
                    let mut fields: Vec<(String, Ty)> = Vec::new();
                    for f in &s.fields {
                        if fields.iter().any(|(n, _)| *n == f.name) {
                            self.error(f.span, format!("field `{}` is declared twice in `{}`", f.name, s.name));
                            continue;
                        }
                        let ty = self.resolve_type(&f.ty);
                        fields.push((f.name.clone(), ty));
                    }
                    if let Some(info) = self.info.structs.get_mut(&s.name) {
                        info.fields = fields;
                    }
                }
                ItemKind::Enum(e) => {
                    let mut variants: Vec<VariantInfo> = Vec::new();
                    for v in &e.variants {
                        if variants.iter().any(|x| x.name == v.name) {
                            self.error(v.span, format!("variant `{}` is declared twice in `{}`", v.name, e.name));
                            continue;
                        }
                        let fields = v
                            .fields
                            .iter()
                            .map(|f| (f.name.clone(), self.resolve_type(&f.ty)))
                            .collect();
                        variants.push(VariantInfo {
                            name: v.name.clone(),
                            fields,
                        });
                    }
                    if let Some(info) = self.info.enums.get_mut(&e.name) {
                        info.variants = variants;
                    }
                }
                _ => {}
            }
        }
    }

    fn resolve_type(&mut self, t: &jax_syntax::ast::Type) -> Ty {
        match &t.kind {
            TypeKind::Named { name, args } => {
                let simple = |this: &mut Self, ty: Ty| -> Ty {
                    if !args.is_empty() {
                        this.error(t.span, format!("`{name}` does not take type arguments"));
                    }
                    ty
                };
                match name.as_str() {
                    "Int" => simple(self, Ty::Int),
                    "Float" => simple(self, Ty::Float),
                    "Bool" => simple(self, Ty::Bool),
                    "Text" => simple(self, Ty::Text),
                    "Unit" => simple(self, Ty::Unit),
                    "List" => {
                        if args.len() != 1 {
                            self.error(t.span, "`List` takes exactly one type argument, e.g. `List<Int>`");
                            return Ty::Error;
                        }
                        let inner = self.resolve_type(&args[0]);
                        Ty::List(Box::new(inner))
                    }
                    "Int8" | "Int16" | "Int32" | "UInt8" | "UInt16" | "UInt32" | "UInt64" | "Float32" | "Char"
                    | "Bytes" => {
                        self.unsupported(t.span, &format!("the `{name}` type"));
                        Ty::Error
                    }
                    "Map" | "Set" | "Result" | "Shared" => {
                        self.unsupported(t.span, &format!("the `{name}` type"));
                        Ty::Error
                    }
                    n if self.info.structs.contains_key(n) => {
                        if !args.is_empty() {
                            self.unsupported(t.span, "generic types");
                        }
                        Ty::Struct(n.to_string())
                    }
                    n if self.info.enums.contains_key(n) => {
                        if !args.is_empty() {
                            self.unsupported(t.span, "generic types");
                        }
                        Ty::Enum(n.to_string())
                    }
                    n => {
                        let known: Vec<&str> = ["Int", "Float", "Bool", "Text", "Unit", "List"]
                            .into_iter()
                            .chain(self.info.structs.keys().map(|s| s.as_str()))
                            .chain(self.info.enums.keys().map(|s| s.as_str()))
                            .collect();
                        let hint = suggest(n, known);
                        let msg = format!("unknown type `{n}`");
                        match hint {
                            Some(h) => self.error_help(t.span, msg, format!("did you mean `{h}`?")),
                            None => self.error(t.span, msg),
                        }
                        Ty::Error
                    }
                }
            }
            TypeKind::Optional(inner) => {
                let i = self.resolve_type(inner);
                if matches!(i, Ty::Opt(_)) {
                    self.error(t.span, "nested optional types (`T??`) are not allowed");
                    return Ty::Error;
                }
                if i == Ty::Unit {
                    self.error(t.span, "`Unit?` is not a useful type");
                    return Ty::Error;
                }
                Ty::Opt(Box::new(i))
            }
            TypeKind::Tuple(_) => {
                self.unsupported(t.span, "tuple types");
                Ty::Error
            }
            TypeKind::Fn { .. } => {
                self.unsupported(t.span, "function types");
                Ty::Error
            }
            TypeKind::Dyn(_) => {
                self.unsupported(t.span, "`dyn` trait objects");
                Ty::Error
            }
        }
    }

    fn signature(&mut self, f: &FnDecl) -> FnSig {
        if !f.generics.is_empty() {
            self.unsupported(f.name_span, "generic functions");
        }
        if f.is_async {
            self.unsupported(f.name_span, "`async fn`");
        }
        let params = f
            .params
            .iter()
            .map(|p| ParamSig {
                name: p.name.clone(),
                ty: self.resolve_type(&p.ty),
                has_default: p.default.is_some(),
            })
            .collect();
        let ret = match &f.ret {
            Some(t) => self.resolve_type(t),
            None => Ty::Unit,
        };
        FnSig { params, ret }
    }

    fn collect_fns(&mut self, program: &'a Program) {
        for item in &program.items {
            match &item.kind {
                ItemKind::Fn(f) => {
                    if BUILTIN_FNS.contains(&f.name.as_str()) {
                        self.error(f.name_span, format!("`{}` is a built-in function and cannot be redefined", f.name));
                        continue;
                    }
                    if self.fns.contains_key(&f.name) || self.is_type_name(&f.name) {
                        self.error(f.name_span, format!("`{}` is defined more than once", f.name));
                        continue;
                    }
                    let sig = self.signature(f);
                    self.fns.insert(f.name.clone(), sig);
                }
                ItemKind::Impl(im) => self.collect_impl(im),
                ItemKind::Use(u) => {
                    if u.path.first().map(|s| s.as_str()) != Some("std") {
                        self.unsupported(item.span, "importing modules (`use`); Jax programs are single-file for now");
                    }
                }
                ItemKind::Trait(_) | ItemKind::Const(_) | ItemKind::Test(_) | ItemKind::Struct(_) | ItemKind::Enum(_) => {}
            }
        }
    }

    fn collect_impl(&mut self, im: &ImplDecl) {
        if !im.generics.is_empty() {
            self.unsupported(im.self_ty.span, "generic `impl` blocks");
            return;
        }
        let self_ty = self.resolve_type(&im.self_ty);
        let type_name = match &self_ty {
            Ty::Struct(n) | Ty::Enum(n) => n.clone(),
            Ty::Error => return,
            other => {
                self.error(
                    im.self_ty.span,
                    format!("cannot add methods to `{other}`; `impl` blocks are for your own structs and enums"),
                );
                return;
            }
        };
        for item in &im.items {
            let ItemKind::Fn(f) = &item.kind else { continue };
            let key = (type_name.clone(), f.name.clone());
            if self.methods.contains_key(&key) {
                self.error(f.name_span, format!("`{}.{}` is defined more than once", type_name, f.name));
                continue;
            }
            let sig = self.signature(f);
            if f.self_param == Some(SelfKind::Mut) && sig.ret != Ty::Unit && sig.ret != Ty::Error {
                self.error_help(
                    f.name_span,
                    "methods that take `mut self` must return `Unit`",
                    "the updated value is written back to the receiver; return results through a second method",
                );
            }
            if f.body.is_none() {
                self.error(f.name_span, "methods in an `impl` block need a body");
            }
            self.methods.insert(
                key,
                MethodSig {
                    sig,
                    self_kind: f.self_param,
                    mangled: format!("{}.{}", type_name, f.name),
                },
            );
        }
    }

    fn collect_consts(&mut self, program: &'a Program) {
        for item in &program.items {
            let ItemKind::Const(c) = &item.kind else { continue };
            if self.consts.contains_key(&c.name) || self.fns.contains_key(&c.name) || self.is_type_name(&c.name) {
                self.error(c.name_span, format!("`{}` is defined more than once", c.name));
                continue;
            }
            self.scopes = vec![HashMap::new()];
            if !self.is_const_expr(&c.value) {
                self.error_help(
                    c.value.span,
                    "constant initializers must be literals, arithmetic or other constants",
                    "use `let` inside a function for values computed at run time",
                );
            }
            let declared = c.ty.as_ref().map(|t| self.resolve_type(t));
            let t = self.expr(&c.value, declared.as_ref());
            let ty = match declared {
                Some(d) => {
                    if !assignable(&d, &t) {
                        self.mismatch(c.value.span, &d, &t);
                    }
                    d
                }
                None => t,
            };
            self.consts.insert(c.name.clone(), ty);
        }
        self.scopes.clear();
    }

    fn is_const_expr(&self, e: &Expr) -> bool {
        match &e.kind {
            ExprKind::Int(_) | ExprKind::Float(_) | ExprKind::Bool(_) | ExprKind::Str(_) => true,
            ExprKind::Unary { expr, .. } => self.is_const_expr(expr),
            ExprKind::Binary { lhs, rhs, .. } => self.is_const_expr(lhs) && self.is_const_expr(rhs),
            ExprKind::Ident(n) => self.consts.contains_key(n),
            _ => false,
        }
    }

    fn check_bodies(&mut self, program: &'a Program) {
        for item in &program.items {
            match &item.kind {
                ItemKind::Fn(f) => {
                    if let Some(sig) = self.fns.get(&f.name).cloned() {
                        self.check_fn(f, &f.name, &sig, None);
                    }
                }
                ItemKind::Impl(im) => {
                    if !im.generics.is_empty() {
                        continue;
                    }
                    let self_ty = self.resolve_type_quiet(&im.self_ty);
                    let Some(self_ty) = self_ty else { continue };
                    let tname = match &self_ty {
                        Ty::Struct(n) | Ty::Enum(n) => n.clone(),
                        _ => continue,
                    };
                    for it in &im.items {
                        let ItemKind::Fn(f) = &it.kind else { continue };
                        if let Some(ms) = self.methods.get(&(tname.clone(), f.name.clone())).cloned() {
                            self.check_fn(f, &ms.mangled, &ms.sig, Some(self_ty.clone()));
                        }
                    }
                }
                ItemKind::Test(t) => {
                    self.scopes = vec![HashMap::new()];
                    self.ret_ty = Ty::Unit;
                    self.mut_self_method = false;
                    self.loop_breaks.clear();
                    let bt = self.block(&t.body, None);
                    let _ = bt;
                    self.scopes.clear();
                }
                ItemKind::Trait(t) => {
                    if t.items.iter().any(|i| matches!(&i.kind, ItemKind::Fn(f) if f.body.is_some())) {
                        self.unsupported(t.name_span, "default method bodies in traits");
                    }
                }
                _ => {}
            }
        }
    }

    /// Resolve an already-validated type without reporting errors again.
    fn resolve_type_quiet(&mut self, t: &jax_syntax::ast::Type) -> Option<Ty> {
        let before = self.diags.len();
        let ty = self.resolve_type(t);
        self.diags.truncate(before);
        if ty == Ty::Error {
            None
        } else {
            Some(ty)
        }
    }

    fn check_fn(&mut self, f: &FnDecl, mangled: &str, sig: &FnSig, self_ty: Option<Ty>) {
        let Some(body) = &f.body else { return };
        self.scopes = vec![HashMap::new()];
        self.loop_breaks.clear();
        // Default values are checked before the parameters are in scope.
        for (p, ps) in f.params.iter().zip(&sig.params) {
            if let Some(d) = &p.default {
                if !self.is_const_expr(d) {
                    self.error(d.span, "default argument values must be literals or constants");
                }
                let t = self.expr(d, Some(&ps.ty));
                if !assignable(&ps.ty, &t) {
                    self.mismatch(d.span, &ps.ty, &t);
                }
            }
        }
        if let (Some(kind), Some(st)) = (f.self_param, self_ty) {
            self.declare("self", st, kind == SelfKind::Mut);
        } else if f.self_param.is_some() {
            self.error(f.name_span, "`self` can only be used in methods");
        }
        for ps in &sig.params {
            self.declare(&ps.name, ps.ty.clone(), false);
        }
        self.ret_ty = sig.ret.clone();
        self.mut_self_method = f.self_param == Some(SelfKind::Mut);
        let _ = mangled;
        let ret = sig.ret.clone();
        let t = self.block(body, Some(&ret));
        if !assignable(&ret, &t) {
            let span = body.tail.as_ref().map(|e| e.span).unwrap_or(f.name_span);
            if body.tail.is_none() && ret != Ty::Unit {
                self.error_help(
                    span,
                    format!("function `{}` must return a value of type `{ret}`", f.name),
                    "end the function with an expression, or use `return`",
                );
            } else {
                self.mismatch(span, &ret, &t);
            }
        }
        self.scopes.clear();
    }

    fn check_main(&mut self, program: &'a Program) {
        for item in &program.items {
            let ItemKind::Fn(f) = &item.kind else { continue };
            if f.name != "main" {
                continue;
            }
            self.info.has_main = true;
            if !f.params.is_empty() || f.self_param.is_some() {
                self.error(f.name_span, "`main` cannot take parameters yet");
            }
            if let Some(ret) = &f.ret {
                let is_unit = matches!(&ret.kind, TypeKind::Named { name, .. } if name == "Unit");
                if !is_unit {
                    self.unsupported(ret.span, "returning a value (such as `Result`) from `main`");
                }
            }
        }
    }

    // ----- blocks and statements -----------------------------------------

    fn block(&mut self, b: &Block, expected: Option<&Ty>) -> Ty {
        self.scopes.push(HashMap::new());
        let mut diverges = false;
        for s in &b.stmts {
            match &s.kind {
                StmtKind::Let {
                    name,
                    mutable,
                    ty,
                    init,
                    ..
                } => {
                    let declared = ty.as_ref().map(|t| self.resolve_type(t));
                    let it = self.expr(init, declared.as_ref());
                    let final_ty = match declared {
                        Some(d) => {
                            if !assignable(&d, &it) {
                                self.mismatch(init.span, &d, &it);
                            }
                            d
                        }
                        None => match &it {
                            Ty::NoneLit => {
                                self.error_help(
                                    init.span,
                                    "cannot infer the type of `none`",
                                    "add a type annotation, e.g. `let x: Int? = none`",
                                );
                                Ty::Error
                            }
                            Ty::List(inner) if **inner == Ty::Unknown => {
                                self.error_help(
                                    init.span,
                                    "cannot infer the element type of an empty list",
                                    "add a type annotation, e.g. `let xs: List<Int> = []`",
                                );
                                Ty::Error
                            }
                            Ty::Never => {
                                diverges = true;
                                Ty::Error
                            }
                            _ => it.clone(),
                        },
                    };
                    if it == Ty::Never {
                        diverges = true;
                    }
                    self.declare(name, final_ty, *mutable);
                }
                StmtKind::Expr(e) => {
                    let t = self.expr(e, None);
                    if t == Ty::Never {
                        diverges = true;
                    }
                }
            }
        }
        let ty = match &b.tail {
            Some(e) => self.expr(e, expected),
            None => {
                if diverges {
                    Ty::Never
                } else {
                    Ty::Unit
                }
            }
        };
        self.scopes.pop();
        ty
    }

    // ----- expressions ---------------------------------------------------

    fn expr(&mut self, e: &Expr, expected: Option<&Ty>) -> Ty {
        let t = self.expr_inner(e, expected);
        self.info.expr_types.insert(e.id, t.clone());
        t
    }

    fn expr_inner(&mut self, e: &Expr, expected: Option<&Ty>) -> Ty {
        match &e.kind {
            ExprKind::Int(_) => Ty::Int,
            ExprKind::Float(_) => Ty::Float,
            ExprKind::Bool(_) => Ty::Bool,
            ExprKind::Str(_) => Ty::Text,
            ExprKind::None => Ty::NoneLit,
            ExprKind::Interp(parts) => {
                for p in parts {
                    if let StrPart::Expr(x) = p {
                        let t = self.expr(x, None);
                        if !displayable(&t) {
                            self.error_help(
                                x.span,
                                format!("a value of type `{t}` cannot be used in a string interpolation"),
                                "only Int, Float, Bool and Text values can be interpolated",
                            );
                        }
                    }
                }
                Ty::Text
            }
            ExprKind::Ident(name) => self.ident(e, name),
            ExprKind::Unary { op, expr } => {
                let t = self.expr(expr, None);
                match (op, &t) {
                    (_, Ty::Error) => Ty::Error,
                    (UnOp::Neg, Ty::Int) | (UnOp::Neg, Ty::Float) => t,
                    (UnOp::Not, Ty::Bool) => Ty::Bool,
                    (UnOp::Neg, other) => {
                        self.error(e.span, format!("cannot negate a value of type `{other}`"));
                        Ty::Error
                    }
                    (UnOp::Not, other) => {
                        self.error(e.span, format!("`!` needs a `Bool`, found `{other}`"));
                        Ty::Error
                    }
                }
            }
            ExprKind::Binary { op, lhs, rhs } => self.binary(e, *op, lhs, rhs),
            ExprKind::Assign { op, target, value } => {
                let tt = self.expr(target, None);
                self.check_place(target);
                let vt = self.expr(value, Some(&tt));
                match op {
                    None => {
                        if !assignable(&tt, &vt) {
                            self.mismatch(value.span, &tt, &vt);
                        }
                    }
                    Some(bop) => {
                        let rt = self.binary_result(*bop, &tt, &vt, e.span);
                        if !assignable(&tt, &rt) {
                            self.mismatch(value.span, &tt, &rt);
                        }
                    }
                }
                Ty::Unit
            }
            ExprKind::Call { callee, args } => self.call(e, callee, args),
            ExprKind::Field { obj, name, name_span } => self.field(e, obj, name, *name_span),
            ExprKind::OptField { obj, name, name_span } => {
                let ot = self.expr(obj, None);
                match &ot {
                    Ty::Opt(inner) => match &**inner {
                        Ty::Struct(sn) => match self.field_of(sn, name) {
                            Some((idx, fty)) => {
                                self.rec(e.id, Res::Field(idx));
                                if matches!(fty, Ty::Opt(_)) {
                                    fty
                                } else {
                                    Ty::Opt(Box::new(fty))
                                }
                            }
                            None => {
                                self.no_such_field(*name_span, sn, name);
                                Ty::Error
                            }
                        },
                        other => {
                            self.error(*name_span, format!("`{other}` has no field `{name}`"));
                            Ty::Error
                        }
                    },
                    Ty::Error => Ty::Error,
                    other => {
                        self.error_help(
                            e.span,
                            format!("`?.` needs an optional value, found `{other}`"),
                            "use `.` for values that are never `none`",
                        );
                        Ty::Error
                    }
                }
            }
            ExprKind::Index { obj, index } => {
                let ot = self.expr(obj, None);
                let it = self.expr(index, Some(&Ty::Int));
                if !assignable(&Ty::Int, &it) {
                    self.mismatch(index.span, &Ty::Int, &it);
                }
                match ot {
                    Ty::List(t) => *t,
                    Ty::Error => Ty::Error,
                    other => {
                        self.error(obj.span, format!("cannot index into a value of type `{other}`"));
                        Ty::Error
                    }
                }
            }
            ExprKind::If { cond, then, els } => {
                let ct = self.expr(cond, Some(&Ty::Bool));
                if !assignable(&Ty::Bool, &ct) {
                    self.mismatch(cond.span, &Ty::Bool, &ct);
                }
                let tt = self.block(then, expected);
                match els {
                    None => Ty::Unit,
                    Some(el) => {
                        let et = self.expr(el, expected);
                        self.join_or_error(e.span, &tt, &et, "`if` and `else`")
                    }
                }
            }
            ExprKind::IfLet { pat, value, then, els } => {
                let vt = self.expr(value, None);
                self.scopes.push(HashMap::new());
                let mut binds = Vec::new();
                match (&vt, &pat.kind) {
                    (Ty::Opt(inner), PatKind::Ident(_) | PatKind::Wild) => {
                        self.rec(e.id, Res::IfLetOpt);
                        let inner = (**inner).clone();
                        self.pattern(pat, &inner, &mut binds);
                    }
                    _ => self.pattern(pat, &vt, &mut binds),
                }
                for (n, t) in binds {
                    self.declare(&n, t, false);
                }
                let tt = self.block(then, expected);
                self.scopes.pop();
                match els {
                    None => Ty::Unit,
                    Some(el) => {
                        let et = self.expr(el, expected);
                        self.join_or_error(e.span, &tt, &et, "`if let` and `else`")
                    }
                }
            }
            ExprKind::Match { scrutinee, arms } => self.match_expr(e, scrutinee, arms, expected),
            ExprKind::Block(b) => self.block(b, expected),
            ExprKind::While { cond, body } => {
                let ct = self.expr(cond, Some(&Ty::Bool));
                if !assignable(&Ty::Bool, &ct) {
                    self.mismatch(cond.span, &Ty::Bool, &ct);
                }
                self.loop_breaks.push(false);
                self.block(body, None);
                self.loop_breaks.pop();
                Ty::Unit
            }
            ExprKind::Loop(body) => {
                self.loop_breaks.push(false);
                self.block(body, None);
                let broke = self.loop_breaks.pop().unwrap_or(false);
                if broke {
                    Ty::Unit
                } else {
                    Ty::Never
                }
            }
            ExprKind::For {
                var,
                iter,
                body,
                ..
            } => {
                let var_ty = match &iter.kind {
                    ExprKind::Range { lo, hi, .. } => {
                        for x in [lo, hi] {
                            let t = self.expr(x, Some(&Ty::Int));
                            if !assignable(&Ty::Int, &t) {
                                self.mismatch(x.span, &Ty::Int, &t);
                            }
                        }
                        self.info.expr_types.insert(iter.id, Ty::Unit);
                        Ty::Int
                    }
                    _ => {
                        let it = self.expr(iter, None);
                        match it {
                            Ty::List(t) => *t,
                            Ty::Error => Ty::Error,
                            other => {
                                self.error_help(
                                    iter.span,
                                    format!("cannot loop over a value of type `{other}`"),
                                    "loop over a range like `0..10` or over a `List`",
                                );
                                Ty::Error
                            }
                        }
                    }
                };
                self.scopes.push(HashMap::new());
                self.declare(var, var_ty, false);
                self.loop_breaks.push(false);
                self.block(body, None);
                self.loop_breaks.pop();
                self.scopes.pop();
                Ty::Unit
            }
            ExprKind::Return(v) => {
                if self.mut_self_method {
                    if let Some(x) = v {
                        self.expr(x, None);
                        self.error(x.span, "`mut self` methods cannot return a value");
                    }
                } else {
                    let ret = self.ret_ty.clone();
                    match v {
                        Some(x) => {
                            let t = self.expr(x, Some(&ret));
                            if !assignable(&ret, &t) {
                                self.mismatch(x.span, &ret, &t);
                            }
                        }
                        None => {
                            if ret != Ty::Unit && ret != Ty::Error {
                                self.error(e.span, format!("`return` needs a value of type `{ret}`"));
                            }
                        }
                    }
                }
                Ty::Never
            }
            ExprKind::Break | ExprKind::Continue => {
                if self.loop_breaks.is_empty() {
                    let kw = if matches!(e.kind, ExprKind::Break) { "break" } else { "continue" };
                    self.error(e.span, format!("`{kw}` can only be used inside a loop"));
                } else if matches!(e.kind, ExprKind::Break) {
                    if let Some(last) = self.loop_breaks.last_mut() {
                        *last = true;
                    }
                }
                Ty::Never
            }
            ExprKind::Range { .. } => {
                self.error_help(
                    e.span,
                    "a range can only be used as the iterable of a `for` loop",
                    "write `for i in 0..10 { ... }`",
                );
                Ty::Error
            }
            ExprKind::List(elems) => self.list_literal(elems, expected),
            ExprKind::Tuple(_) => {
                self.unsupported(e.span, "tuples");
                Ty::Error
            }
            ExprKind::Closure { .. } => {
                self.unsupported(e.span, "closures");
                Ty::Error
            }
            ExprKind::Try(_) => {
                self.unsupported(e.span, "the `?` operator (error propagation with `Result`)");
                Ty::Error
            }
            ExprKind::Await(_) => {
                self.unsupported(e.span, "`await`");
                Ty::Error
            }
            ExprKind::Unwrap(inner) => {
                let t = self.expr(inner, None);
                match t {
                    Ty::Opt(i) => *i,
                    Ty::Error => Ty::Error,
                    other => {
                        self.error(e.span, format!("`!` can only unwrap optional values, found `{other}`"));
                        Ty::Error
                    }
                }
            }
            ExprKind::With { base, fields } => {
                let bt = self.expr(base, None);
                let Ty::Struct(sn) = &bt else {
                    if bt != Ty::Error {
                        self.error(base.span, format!("`with` updates structs, found `{bt}`"));
                    }
                    for (_, _, v) in fields {
                        self.expr(v, None);
                    }
                    return Ty::Error;
                };
                let mut idxs = Vec::new();
                let mut seen = HashSet::new();
                for (fname, fspan, v) in fields {
                    match self.field_of(sn, fname) {
                        Some((idx, fty)) => {
                            if !seen.insert(idx) {
                                self.error(*fspan, format!("field `{fname}` is updated twice"));
                            }
                            let vt = self.expr(v, Some(&fty));
                            if !assignable(&fty, &vt) {
                                self.mismatch(v.span, &fty, &vt);
                            }
                            idxs.push(idx);
                        }
                        None => {
                            self.no_such_field(*fspan, sn, fname);
                            self.expr(v, None);
                            idxs.push(0);
                        }
                    }
                }
                self.rec(e.id, Res::With(idxs));
                bt
            }
        }
    }

    fn join_or_error(&mut self, span: Span, a: &Ty, b: &Ty, what: &str) -> Ty {
        match join(a, b) {
            Some(t) => t,
            None => {
                self.error(span, format!("{what} branches have different types: `{a}` and `{b}`"));
                Ty::Error
            }
        }
    }

    fn ident(&mut self, e: &Expr, name: &str) -> Ty {
        if let Some(l) = self.lookup(name) {
            return l.ty.clone();
        }
        if let Some(t) = self.consts.get(name).cloned() {
            self.rec(e.id, Res::Const(name.to_string()));
            return t;
        }
        if self.fns.contains_key(name) || BUILTIN_FNS.contains(&name) {
            self.error_help(
                e.span,
                format!("`{name}` is a function and cannot be used as a value yet"),
                "call it with parentheses",
            );
            return Ty::Error;
        }
        if self.is_type_name(name) {
            self.error_help(
                e.span,
                format!("`{name}` is a type, not a value"),
                "construct it with `Name(field: value, ...)` or `Enum.Variant`",
            );
            return Ty::Error;
        }
        let mut candidates: Vec<&str> = Vec::new();
        for s in &self.scopes {
            candidates.extend(s.keys().map(|k| k.as_str()));
        }
        candidates.extend(self.consts.keys().map(|k| k.as_str()));
        candidates.extend(self.fns.keys().map(|k| k.as_str()));
        let hint = suggest(name, candidates);
        let msg = format!("cannot find `{name}` in this scope");
        match hint {
            Some(h) => self.error_help(e.span, msg, format!("did you mean `{h}`?")),
            None => self.error(e.span, msg),
        }
        Ty::Error
    }

    /// Ensure `e` is an assignable place whose root binding is mutable.
    fn check_place(&mut self, e: &Expr) {
        match &e.kind {
            ExprKind::Ident(n) => match self.lookup(n) {
                Some(l) if l.mutable => {}
                Some(_) => {
                    if n == "self" {
                        self.error_help(
                            e.span,
                            "cannot modify `self` in a method that takes `self`",
                            "declare the method with `mut self`",
                        );
                    } else {
                        self.error_help(
                            e.span,
                            format!("cannot modify `{n}` because it is immutable"),
                            format!("declare it with `var {n} = ...` instead of `let`"),
                        );
                    }
                }
                None => {
                    if self.consts.contains_key(n) {
                        self.error(e.span, format!("cannot assign to the constant `{n}`"));
                    }
                }
            },
            ExprKind::Field { obj, .. } | ExprKind::Index { obj, .. } => self.check_place(obj),
            _ => self.error(e.span, "this expression cannot be assigned to"),
        }
    }

    fn binary(&mut self, e: &Expr, op: BinOp, lhs: &Expr, rhs: &Expr) -> Ty {
        if op == BinOp::Coalesce {
            let lt = self.expr(lhs, None);
            return match &lt {
                Ty::Opt(inner) => {
                    let inner = (**inner).clone();
                    let rt = self.expr(rhs, Some(&inner));
                    if assignable(&inner, &rt) {
                        inner
                    } else if assignable(&lt, &rt) {
                        lt
                    } else {
                        self.mismatch(rhs.span, &inner, &rt);
                        Ty::Error
                    }
                }
                Ty::Error => {
                    self.expr(rhs, None);
                    Ty::Error
                }
                other => {
                    self.expr(rhs, None);
                    self.error_help(
                        lhs.span,
                        format!("`??` needs an optional value on the left, found `{other}`"),
                        "this value can never be `none`, so no default is needed",
                    );
                    Ty::Error
                }
            };
        }
        let lt = self.expr(lhs, None);
        let hint = if matches!(op, BinOp::Eq | BinOp::Ne) { Some(lt.clone()) } else { None };
        let rt = self.expr(rhs, hint.as_ref());
        self.binary_result(op, &lt, &rt, e.span)
    }

    fn binary_result(&mut self, op: BinOp, lt: &Ty, rt: &Ty, span: Span) -> Ty {
        if *lt == Ty::Error || *rt == Ty::Error {
            return Ty::Error;
        }
        let ok = match op {
            BinOp::Add => match (lt, rt) {
                (Ty::Int, Ty::Int) => Some(Ty::Int),
                (Ty::Float, Ty::Float) => Some(Ty::Float),
                (Ty::Text, Ty::Text) => Some(Ty::Text),
                _ => None,
            },
            BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem => match (lt, rt) {
                (Ty::Int, Ty::Int) => Some(Ty::Int),
                (Ty::Float, Ty::Float) => Some(Ty::Float),
                _ => None,
            },
            BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => match (lt, rt) {
                (Ty::Int, Ty::Int) | (Ty::Float, Ty::Float) | (Ty::Text, Ty::Text) => Some(Ty::Bool),
                _ => None,
            },
            BinOp::Eq | BinOp::Ne => {
                if *lt != Ty::Never && *rt != Ty::Never && (assignable(lt, rt) || assignable(rt, lt)) {
                    Some(Ty::Bool)
                } else {
                    None
                }
            }
            BinOp::And | BinOp::Or => match (lt, rt) {
                (Ty::Bool, Ty::Bool) => Some(Ty::Bool),
                _ => None,
            },
            BinOp::Coalesce => None,
        };
        match ok {
            Some(t) => t,
            None => {
                let help = match (lt, rt) {
                    (Ty::Int, Ty::Float) => Some("convert the Int with `.to_float()`"),
                    (Ty::Float, Ty::Int) => Some("convert the Int with `.to_float()`"),
                    (Ty::Text, Ty::Int | Ty::Float | Ty::Bool) | (Ty::Int | Ty::Float | Ty::Bool, Ty::Text)
                        if op == BinOp::Add =>
                    {
                        Some("convert with `.to_text()` or use string interpolation: \"{x}\"")
                    }
                    _ => None,
                };
                let msg = format!("cannot apply `{}` to `{lt}` and `{rt}`", op_symbol(op));
                match help {
                    Some(h) => self.error_help(span, msg, h),
                    None => self.error(span, msg),
                }
                Ty::Error
            }
        }
    }

    fn list_literal(&mut self, elems: &[Expr], expected: Option<&Ty>) -> Ty {
        let hint = match expected {
            Some(Ty::List(t)) if **t != Ty::Unknown => Some((**t).clone()),
            _ => None,
        };
        if let Some(h) = hint {
            for el in elems {
                let t = self.expr(el, Some(&h));
                if !assignable(&h, &t) {
                    self.mismatch(el.span, &h, &t);
                }
            }
            return Ty::List(Box::new(h));
        }
        let mut cur: Option<Ty> = None;
        for el in elems {
            let t = self.expr(el, cur.as_ref());
            cur = match cur {
                None => Some(t),
                Some(c) => match join(&c, &t) {
                    Some(j) => Some(j),
                    None => {
                        self.mismatch(el.span, &c, &t);
                        Some(c)
                    }
                },
            };
        }
        match cur {
            None => Ty::List(Box::new(Ty::Unknown)),
            Some(Ty::NoneLit) => Ty::List(Box::new(Ty::NoneLit)),
            Some(t) => Ty::List(Box::new(t)),
        }
    }

    // ----- fields, calls --------------------------------------------------

    fn field_of(&self, struct_name: &str, field: &str) -> Option<(usize, Ty)> {
        let s = self.info.structs.get(struct_name)?;
        s.fields
            .iter()
            .enumerate()
            .find(|(_, (n, _))| n == field)
            .map(|(i, (_, t))| (i, t.clone()))
    }

    fn no_such_field(&mut self, span: Span, struct_name: &str, field: &str) {
        let names: Vec<String> = self
            .info
            .structs
            .get(struct_name)
            .map(|s| s.fields.iter().map(|(n, _)| n.clone()).collect())
            .unwrap_or_default();
        let hint = suggest(field, names.iter().map(|s| s.as_str()));
        let msg = format!("`{struct_name}` has no field `{field}`");
        match hint {
            Some(h) => self.error_help(span, msg, format!("did you mean `{h}`?")),
            None => self.error(span, msg),
        }
    }

    fn variant_of(&self, enum_name: &str, variant: &str) -> Option<(u32, VariantInfo)> {
        let en = self.info.enums.get(enum_name)?;
        en.variants
            .iter()
            .enumerate()
            .find(|(_, v)| v.name == variant)
            .map(|(i, v)| (i as u32, v.clone()))
    }

    fn field(&mut self, e: &Expr, obj: &Expr, name: &str, name_span: Span) -> Ty {
        // `Enum.Variant` (unit variant) or a misuse of a payload variant.
        if let ExprKind::Ident(tn) = &obj.kind {
            if self.lookup(tn).is_none() && self.info.enums.contains_key(tn) {
                return match self.variant_of(tn, name) {
                    Some((tag, v)) => {
                        if !v.fields.is_empty() {
                            self.error_help(
                                e.span,
                                format!("variant `{tn}.{name}` carries data"),
                                format!("construct it with `{tn}.{name}(...)`"),
                            );
                            Ty::Error
                        } else {
                            self.rec(e.id, Res::VariantUnit { tag });
                            Ty::Enum(tn.clone())
                        }
                    }
                    None => {
                        self.error(name_span, format!("enum `{tn}` has no variant `{name}`"));
                        Ty::Error
                    }
                };
            }
        }
        let ot = self.expr(obj, None);
        match &ot {
            Ty::Struct(sn) => match self.field_of(sn, name) {
                Some((idx, t)) => {
                    self.rec(e.id, Res::Field(idx));
                    t
                }
                None => {
                    if self.methods.contains_key(&(sn.clone(), name.to_string())) {
                        self.error_help(
                            name_span,
                            format!("`{name}` is a method, not a field"),
                            format!("call it: `.{name}(...)`"),
                        );
                    } else {
                        self.no_such_field(name_span, sn, name);
                    }
                    Ty::Error
                }
            },
            Ty::Opt(_) => {
                self.error_help(
                    name_span,
                    format!("cannot access `{name}` on `{ot}` because it may be `none`"),
                    "use `?.` to access it safely, or unwrap with `!` / `if let`",
                );
                Ty::Error
            }
            Ty::Error => Ty::Error,
            other => {
                self.error(name_span, format!("`{other}` has no field `{name}`"));
                Ty::Error
            }
        }
    }

    /// Match explicit arguments to parameters; returns the argument order for codegen.
    fn bind_args(
        &mut self,
        label: &str,
        params: &[ParamSig],
        func_key: Option<&str>,
        args: &[Arg],
        call_span: Span,
    ) -> Option<Vec<ParamSource>> {
        let mut slots: Vec<Option<usize>> = vec![None; params.len()];
        let mut ok = true;
        let mut seen_named = false;
        let mut too_many_reported = false;
        let mut unbound: Vec<usize> = Vec::new();
        for (i, a) in args.iter().enumerate() {
            match &a.name {
                None => {
                    if seen_named {
                        self.error(a.value.span, "positional arguments must come before named arguments");
                        ok = false;
                        unbound.push(i);
                    } else if i >= params.len() {
                        if !too_many_reported {
                            self.error(
                                a.value.span,
                                format!(
                                    "too many arguments: `{label}` takes {} argument{}",
                                    params.len(),
                                    if params.len() == 1 { "" } else { "s" }
                                ),
                            );
                            too_many_reported = true;
                        }
                        ok = false;
                        unbound.push(i);
                    } else {
                        slots[i] = Some(i);
                    }
                }
                Some(n) => {
                    seen_named = true;
                    match params.iter().position(|p| p.name == *n) {
                        None => {
                            let hint = suggest(n, params.iter().map(|p| p.name.as_str()));
                            let msg = format!("`{label}` has no parameter named `{n}`");
                            match hint {
                                Some(h) => self.error_help(a.value.span, msg, format!("did you mean `{h}`?")),
                                None => self.error(a.value.span, msg),
                            }
                            ok = false;
                            unbound.push(i);
                        }
                        Some(pi) => {
                            if slots[pi].is_some() {
                                self.error(a.value.span, format!("argument `{n}` is given more than once"));
                                ok = false;
                                unbound.push(i);
                            } else {
                                slots[pi] = Some(i);
                            }
                        }
                    }
                }
            }
        }
        for (pi, slot) in slots.iter().enumerate() {
            if let Some(ai) = slot {
                let expected = params[pi].ty.clone();
                let t = self.expr(&args[*ai].value, Some(&expected));
                if !assignable(&expected, &t) {
                    self.mismatch(args[*ai].value.span, &expected, &t);
                    ok = false;
                }
            }
        }
        for ai in unbound {
            self.expr(&args[ai].value, None);
        }
        for (pi, slot) in slots.iter().enumerate() {
            if slot.is_none() && !params[pi].has_default {
                self.error(
                    call_span,
                    format!("missing argument `{}` in call to `{label}`", params[pi].name),
                );
                ok = false;
            }
        }
        if !ok {
            return None;
        }
        Some(
            slots
                .iter()
                .enumerate()
                .map(|(pi, slot)| match slot {
                    Some(ai) => ParamSource::Arg(*ai),
                    None => ParamSource::Default {
                        func: func_key.unwrap_or("").to_string(),
                        index: pi,
                    },
                })
                .collect(),
        )
    }

    fn call(&mut self, e: &Expr, callee: &Expr, args: &[Arg]) -> Ty {
        match &callee.kind {
            ExprKind::Ident(name) => {
                if self.lookup(name).is_some() {
                    self.error_help(
                        callee.span,
                        format!("`{name}` is a variable and cannot be called"),
                        "closures and function values are not supported yet",
                    );
                    self.check_args_only(args);
                    return Ty::Error;
                }
                match name.as_str() {
                    "print" => return self.builtin_print(e, args),
                    "panic" => return self.builtin_simple(e, args, "panic", Builtin::Panic, &[Ty::Text], Ty::Never),
                    "assert" => return self.builtin_simple(e, args, "assert", Builtin::Assert, &[Ty::Bool], Ty::Unit),
                    _ => {}
                }
                if let Some(sig) = self.fns.get(name).cloned() {
                    let order = self.bind_args(name, &sig.params, Some(name), args, e.span);
                    if let Some(order) = order {
                        self.rec(
                            e.id,
                            Res::Call(CallPlan {
                                target: Target::Func(name.clone()),
                                order,
                                has_receiver: false,
                                writeback: false,
                            }),
                        );
                    }
                    return sig.ret;
                }
                if let Some(si) = self.info.structs.get(name).cloned() {
                    let params: Vec<ParamSig> = si
                        .fields
                        .iter()
                        .map(|(n, t)| ParamSig {
                            name: n.clone(),
                            ty: t.clone(),
                            has_default: false,
                        })
                        .collect();
                    if let Some(order) = self.bind_args(name, &params, None, args, e.span) {
                        self.rec(
                            e.id,
                            Res::Call(CallPlan {
                                target: Target::StructNew(name.clone()),
                                order,
                                has_receiver: false,
                                writeback: false,
                            }),
                        );
                    }
                    return Ty::Struct(name.clone());
                }
                if self.info.enums.contains_key(name) {
                    self.error_help(
                        callee.span,
                        format!("`{name}` is an enum and cannot be called"),
                        format!("pick a variant: `{name}.Variant(...)`"),
                    );
                    self.check_args_only(args);
                    return Ty::Error;
                }
                let mut cands: Vec<&str> = self.fns.keys().map(|s| s.as_str()).collect();
                cands.extend(BUILTIN_FNS);
                cands.extend(self.info.structs.keys().map(|s| s.as_str()));
                let hint = suggest(name, cands);
                let msg = format!("cannot find function `{name}`");
                match hint {
                    Some(h) => self.error_help(callee.span, msg, format!("did you mean `{h}`?")),
                    None => self.error(callee.span, msg),
                }
                self.check_args_only(args);
                Ty::Error
            }
            ExprKind::Field { obj, name, name_span } => {
                // `Type.something(...)`: enum variant constructor or static method.
                if let ExprKind::Ident(tn) = &obj.kind {
                    if self.lookup(tn).is_none() && self.is_type_name(tn) {
                        return self.type_path_call(e, tn, name, *name_span, args);
                    }
                }
                self.method_call(e, obj, name, *name_span, args)
            }
            _ => {
                self.error(callee.span, "this expression cannot be called");
                self.check_args_only(args);
                Ty::Error
            }
        }
    }

    fn check_args_only(&mut self, args: &[Arg]) {
        for a in args {
            self.expr(&a.value, None);
        }
    }

    fn builtin_print(&mut self, e: &Expr, args: &[Arg]) -> Ty {
        if args.len() != 1 || args[0].name.is_some() {
            self.error(e.span, "`print` takes exactly one argument");
            self.check_args_only(args);
            return Ty::Unit;
        }
        let t = self.expr(&args[0].value, None);
        if !displayable(&t) {
            self.error_help(
                args[0].value.span,
                format!("a value of type `{t}` cannot be printed"),
                "only Int, Float, Bool and Text can be printed; convert other values first",
            );
        }
        self.rec(
            e.id,
            Res::Call(CallPlan {
                target: Target::Builtin(Builtin::Print),
                order: vec![ParamSource::Arg(0)],
                has_receiver: false,
                writeback: false,
            }),
        );
        Ty::Unit
    }

    fn builtin_simple(
        &mut self,
        e: &Expr,
        args: &[Arg],
        label: &str,
        b: Builtin,
        param_tys: &[Ty],
        ret: Ty,
    ) -> Ty {
        let params: Vec<ParamSig> = param_tys
            .iter()
            .enumerate()
            .map(|(i, t)| ParamSig {
                name: format!("arg{i}"),
                ty: t.clone(),
                has_default: false,
            })
            .collect();
        if let Some(order) = self.bind_args(label, &params, None, args, e.span) {
            self.rec(
                e.id,
                Res::Call(CallPlan {
                    target: Target::Builtin(b),
                    order,
                    has_receiver: false,
                    writeback: false,
                }),
            );
        }
        ret
    }

    fn type_path_call(&mut self, e: &Expr, tn: &str, name: &str, name_span: Span, args: &[Arg]) -> Ty {
        // Enum variant constructor?
        if self.info.enums.contains_key(tn) {
            if let Some((tag, v)) = self.variant_of(tn, name) {
                let params: Vec<ParamSig> = v
                    .fields
                    .iter()
                    .enumerate()
                    .map(|(i, (n, t))| ParamSig {
                        name: n.clone().unwrap_or_else(|| format!("_{i}")),
                        ty: t.clone(),
                        has_default: false,
                    })
                    .collect();
                let label = format!("{tn}.{name}");
                if let Some(order) = self.bind_args(&label, &params, None, args, e.span) {
                    self.rec(
                        e.id,
                        Res::Call(CallPlan {
                            target: Target::VariantNew {
                                enum_name: tn.to_string(),
                                tag,
                            },
                            order,
                            has_receiver: false,
                            writeback: false,
                        }),
                    );
                }
                return Ty::Enum(tn.to_string());
            }
        }
        // Static method?
        if let Some(ms) = self.methods.get(&(tn.to_string(), name.to_string())).cloned() {
            if ms.self_kind.is_some() {
                self.error_help(
                    name_span,
                    format!("`{tn}.{name}` is a method and needs a receiver"),
                    format!("call it on a value: `value.{name}(...)`"),
                );
                self.check_args_only(args);
                return Ty::Error;
            }
            let label = format!("{tn}.{name}");
            if let Some(order) = self.bind_args(&label, &ms.sig.params, Some(&ms.mangled), args, e.span) {
                self.rec(
                    e.id,
                    Res::Call(CallPlan {
                        target: Target::Func(ms.mangled.clone()),
                        order,
                        has_receiver: false,
                        writeback: false,
                    }),
                );
            }
            return ms.sig.ret;
        }
        let msg = if self.info.enums.contains_key(tn) {
            format!("enum `{tn}` has no variant or function `{name}`")
        } else {
            format!("`{tn}` has no function `{name}`")
        };
        self.error(name_span, msg);
        self.check_args_only(args);
        Ty::Error
    }

    fn method_call(&mut self, e: &Expr, obj: &Expr, name: &str, name_span: Span, args: &[Arg]) -> Ty {
        let rt = self.expr(obj, None);
        // User-defined methods.
        if let Ty::Struct(tn) | Ty::Enum(tn) = &rt {
            let key = (tn.clone(), name.to_string());
            if let Some(ms) = self.methods.get(&key).cloned() {
                let Some(kind) = ms.self_kind else {
                    self.error_help(
                        name_span,
                        format!("`{tn}.{name}` is a function without `self`"),
                        format!("call it as `{tn}.{name}(...)`"),
                    );
                    self.check_args_only(args);
                    return Ty::Error;
                };
                let writeback = kind == SelfKind::Mut;
                if writeback {
                    self.check_place(obj);
                }
                let label = format!("{tn}.{name}");
                if let Some(order) = self.bind_args(&label, &ms.sig.params, Some(&ms.mangled), args, e.span) {
                    self.rec(
                        e.id,
                        Res::Call(CallPlan {
                            target: Target::Method(ms.mangled.clone()),
                            order,
                            has_receiver: true,
                            writeback,
                        }),
                    );
                }
                return ms.sig.ret;
            }
            let names: Vec<String> = self
                .methods
                .keys()
                .filter(|(t, _)| t == tn)
                .map(|(_, m)| m.clone())
                .collect();
            let hint = suggest(name, names.iter().map(|s| s.as_str()));
            let msg = format!("`{tn}` has no method `{name}`");
            match hint {
                Some(h) => self.error_help(name_span, msg, format!("did you mean `{h}`?")),
                None => self.error(name_span, msg),
            }
            self.check_args_only(args);
            return Ty::Error;
        }
        if rt == Ty::Error {
            self.check_args_only(args);
            return Ty::Error;
        }
        // Built-in methods.
        let spec: Option<(Builtin, Vec<Ty>, Ty, bool)> = match (&rt, name) {
            (Ty::Text, "len") | (Ty::List(_), "len") => Some((Builtin::Len, vec![], Ty::Int, false)),
            (Ty::List(t), "push") => Some((Builtin::Push, vec![(**t).clone()], Ty::Unit, true)),
            (Ty::List(t), "get") => {
                let elem = (**t).clone();
                let ret = if matches!(elem, Ty::Opt(_)) { elem } else { Ty::Opt(Box::new(elem)) };
                Some((Builtin::Get, vec![Ty::Int], ret, false))
            }
            (Ty::Int, "to_float") => Some((Builtin::ToFloat, vec![], Ty::Float, false)),
            (Ty::Float, "to_int") => Some((Builtin::ToInt, vec![], Ty::Int, false)),
            (Ty::Float, "sqrt") => Some((Builtin::Sqrt, vec![], Ty::Float, false)),
            (Ty::Int | Ty::Float | Ty::Bool | Ty::Text, "to_text") => Some((Builtin::ToText, vec![], Ty::Text, false)),
            (Ty::List(t), "to_text") if displayable(t) => Some((Builtin::ToText, vec![], Ty::Text, false)),
            _ => None,
        };
        match spec {
            Some((b, param_tys, ret, mutating)) => {
                if mutating {
                    self.check_place(obj);
                }
                let params: Vec<ParamSig> = param_tys
                    .iter()
                    .enumerate()
                    .map(|(i, t)| ParamSig {
                        name: format!("arg{i}"),
                        ty: t.clone(),
                        has_default: false,
                    })
                    .collect();
                let label = format!("{rt}.{name}");
                if let Some(order) = self.bind_args(&label, &params, None, args, e.span) {
                    self.rec(
                        e.id,
                        Res::Call(CallPlan {
                            target: Target::Builtin(b),
                            order,
                            has_receiver: true,
                            writeback: mutating,
                        }),
                    );
                }
                ret
            }
            None => {
                self.error(name_span, format!("`{rt}` has no method `{name}`"));
                self.check_args_only(args);
                Ty::Error
            }
        }
    }

    // ----- match and patterns --------------------------------------------

    fn match_expr(&mut self, e: &Expr, scrutinee: &Expr, arms: &[MatchArm], expected: Option<&Ty>) -> Ty {
        let st = self.expr(scrutinee, None);
        let mut result: Option<Ty> = None;
        for arm in arms {
            self.scopes.push(HashMap::new());
            let mut binds = Vec::new();
            self.pattern(&arm.pat, &st, &mut binds);
            for (n, t) in binds {
                self.declare(&n, t, false);
            }
            if let Some(g) = &arm.guard {
                let gt = self.expr(g, Some(&Ty::Bool));
                if !assignable(&Ty::Bool, &gt) {
                    self.mismatch(g.span, &Ty::Bool, &gt);
                }
            }
            let hint = result.as_ref().or(expected).cloned();
            let bt = self.expr(&arm.body, hint.as_ref());
            self.scopes.pop();
            result = match result {
                None => Some(bt),
                Some(r) => match join(&r, &bt) {
                    Some(j) => Some(j),
                    None => {
                        self.error(
                            arm.body.span,
                            format!("`match` arms have different types: `{r}` and `{bt}`"),
                        );
                        Some(Ty::Error)
                    }
                },
            };
        }
        if st != Ty::Error {
            self.check_exhaustive(e.span, &st, arms);
        }
        result.unwrap_or(Ty::Never)
    }

    fn is_irrefutable(&self, p: &Pattern) -> bool {
        match &p.kind {
            PatKind::Wild => true,
            PatKind::Ident(_) => matches!(self.info.res.get(&p.id), Some(Res::PatBind)),
            PatKind::Or(alts) => alts.iter().any(|a| self.is_irrefutable(a)),
            _ => false,
        }
    }

    fn collect_variants(&self, p: &Pattern, covered: &mut HashSet<u32>) {
        match &p.kind {
            PatKind::Ident(_) => {
                if let Some(Res::PatVariant { tag }) = self.info.res.get(&p.id) {
                    covered.insert(*tag);
                }
            }
            PatKind::Variant { args, .. } => {
                if let Some(Res::PatVariant { tag }) = self.info.res.get(&p.id) {
                    if args.iter().all(|a| self.is_irrefutable(a)) {
                        covered.insert(*tag);
                    }
                }
            }
            PatKind::Or(alts) => {
                for a in alts {
                    self.collect_variants(a, covered);
                }
            }
            _ => {}
        }
    }

    fn collect_bool_payloads(&self, p: &Pattern, tag: u32, seen: &mut HashSet<bool>) {
        match &p.kind {
            PatKind::Variant { args, .. } => {
                if matches!(self.info.res.get(&p.id), Some(Res::PatVariant { tag: t }) if *t == tag) && args.len() == 1 {
                    match &args[0].kind {
                        PatKind::Bool(b) => {
                            seen.insert(*b);
                        }
                        _ if self.is_irrefutable(&args[0]) => {
                            seen.insert(true);
                            seen.insert(false);
                        }
                        _ => {}
                    }
                }
            }
            PatKind::Or(alts) => {
                for a in alts {
                    self.collect_bool_payloads(a, tag, seen);
                }
            }
            _ => {}
        }
    }

    fn check_exhaustive(&mut self, span: Span, st: &Ty, arms: &[MatchArm]) {
        let unguarded: Vec<&Pattern> = arms.iter().filter(|a| a.guard.is_none()).map(|a| &a.pat).collect();
        if unguarded.iter().any(|p| self.is_irrefutable(p)) {
            return;
        }
        let missing: Vec<String> = match st {
            Ty::Enum(n) => {
                let mut covered = HashSet::new();
                for p in &unguarded {
                    self.collect_variants(p, &mut covered);
                }
                // A variant with a single Bool payload is also covered by `V(true)` + `V(false)`.
                if let Some(en) = self.info.enums.get(n) {
                    for (i, v) in en.variants.iter().enumerate() {
                        let tag = i as u32;
                        if covered.contains(&tag) || v.fields.len() != 1 || v.fields[0].1 != Ty::Bool {
                            continue;
                        }
                        let mut seen = HashSet::new();
                        for p in &unguarded {
                            self.collect_bool_payloads(p, tag, &mut seen);
                        }
                        if seen.len() == 2 {
                            covered.insert(tag);
                        }
                    }
                }
                self.info
                    .enums
                    .get(n)
                    .map(|en| {
                        en.variants
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| !covered.contains(&(*i as u32)))
                            .map(|(_, v)| format!("{n}.{}", v.name))
                            .collect()
                    })
                    .unwrap_or_default()
            }
            Ty::Bool => {
                let mut m = Vec::new();
                for want in [true, false] {
                    let has = unguarded.iter().any(|p| matches!(&p.kind, PatKind::Bool(b) if *b == want));
                    if !has {
                        m.push(want.to_string());
                    }
                }
                m
            }
            Ty::Opt(_) => {
                let mut m = Vec::new();
                if !unguarded.iter().any(|p| matches!(p.kind, PatKind::None)) {
                    m.push("none".to_string());
                }
                let some_ok = unguarded.iter().any(|p| match &p.kind {
                    PatKind::Variant { args, .. } => {
                        matches!(self.info.res.get(&p.id), Some(Res::PatSome)) && args.iter().all(|a| self.is_irrefutable(a))
                    }
                    _ => false,
                });
                if !some_ok {
                    m.push("some(_)".to_string());
                }
                m
            }
            _ => vec!["_".to_string()],
        };
        if !missing.is_empty() {
            self.error_help(
                span,
                format!("non-exhaustive `match`: not covered: {}", missing.join(", ")),
                "add the missing cases or a wildcard `_` arm",
            );
        }
    }

    fn pattern(&mut self, p: &Pattern, scrut: &Ty, binds: &mut Vec<(String, Ty)>) {
        let err_ty = *scrut == Ty::Error;
        let expect = |this: &mut Self, want: Ty, what: &str| {
            if !err_ty && *scrut != want {
                this.error(p.span, format!("a {what} pattern cannot match a value of type `{scrut}`"));
            }
        };
        match &p.kind {
            PatKind::Wild => {}
            PatKind::Ident(name) => {
                if let Ty::Enum(en) = scrut {
                    if let Some((tag, v)) = self.variant_of(en, name) {
                        if !v.fields.is_empty() {
                            self.error_help(
                                p.span,
                                format!("variant `{name}` carries data"),
                                format!("match its payload: `{name}(...)`"),
                            );
                        }
                        self.rec(p.id, Res::PatVariant { tag });
                        return;
                    }
                }
                self.rec(p.id, Res::PatBind);
                if binds.iter().any(|(n, _)| n == name) {
                    self.error(p.span, format!("`{name}` is bound more than once in this pattern"));
                }
                binds.push((name.clone(), scrut.clone()));
            }
            PatKind::Int(_) | PatKind::Range { .. } => expect(self, Ty::Int, "number"),
            PatKind::Float(_) => expect(self, Ty::Float, "number"),
            PatKind::Bool(_) => expect(self, Ty::Bool, "boolean"),
            PatKind::Str(_) => expect(self, Ty::Text, "text"),
            PatKind::None => {
                if !err_ty && !matches!(scrut, Ty::Opt(_)) {
                    self.error(p.span, format!("`none` can only match optional values, not `{scrut}`"));
                }
            }
            PatKind::Variant { path, args } => {
                if path.len() == 1 && path[0] == "some" {
                    match scrut {
                        Ty::Opt(inner) => {
                            self.rec(p.id, Res::PatSome);
                            if args.len() != 1 {
                                self.error(p.span, "`some(...)` takes exactly one pattern");
                                return;
                            }
                            let inner = (**inner).clone();
                            self.pattern(&args[0], &inner, binds);
                        }
                        _ => {
                            if !err_ty {
                                self.error(p.span, format!("`some(...)` can only match optional values, not `{scrut}`"));
                            }
                        }
                    }
                    return;
                }
                let Ty::Enum(en) = scrut else {
                    if !err_ty {
                        self.error(p.span, format!("an enum pattern cannot match a value of type `{scrut}`"));
                    }
                    return;
                };
                let variant_name = if path.len() == 2 {
                    if path[0] != *en {
                        self.error(p.span, format!("pattern is for enum `{}` but the value is `{en}`", path[0]));
                        return;
                    }
                    &path[1]
                } else if path.len() == 1 {
                    &path[0]
                } else {
                    self.error(p.span, "invalid enum pattern path");
                    return;
                };
                match self.variant_of(en, variant_name) {
                    None => {
                        let names: Vec<String> = self
                            .info
                            .enums
                            .get(en)
                            .map(|x| x.variants.iter().map(|v| v.name.clone()).collect())
                            .unwrap_or_default();
                        let hint = suggest(variant_name, names.iter().map(|s| s.as_str()));
                        let msg = format!("enum `{en}` has no variant `{variant_name}`");
                        match hint {
                            Some(h) => self.error_help(p.span, msg, format!("did you mean `{h}`?")),
                            None => self.error(p.span, msg),
                        }
                    }
                    Some((tag, v)) => {
                        self.rec(p.id, Res::PatVariant { tag });
                        if args.len() != v.fields.len() {
                            self.error(
                                p.span,
                                format!(
                                    "variant `{}` has {} field{}, but the pattern has {}",
                                    v.name,
                                    v.fields.len(),
                                    if v.fields.len() == 1 { "" } else { "s" },
                                    args.len()
                                ),
                            );
                            return;
                        }
                        for (sub, (_, fty)) in args.iter().zip(&v.fields) {
                            self.pattern(sub, fty, binds);
                        }
                    }
                }
            }
            PatKind::Tuple(_) => self.unsupported(p.span, "tuple patterns"),
            PatKind::Or(alts) => {
                for a in alts {
                    let mut inner = Vec::new();
                    self.pattern(a, scrut, &mut inner);
                    if !inner.is_empty() {
                        self.unsupported(a.span, "bindings inside `|` patterns");
                    }
                }
            }
        }
    }
}
