//! Code generation: type-checked AST to Jax bytecode.
//!
//! Locals live in fixed frame slots; temporaries use the operand stack.
//! Every expression leaves exactly one value on the stack. Control flow
//! that never produces a value (`return`, `break`) pushes a dummy `Unit`
//! afterwards so the stack model stays consistent; that code is unreachable.

use std::collections::HashMap;

use jax_bytecode::{Builtin, Function, Op, Program as Bytecode};
use jax_syntax::ast::*;
use jax_types::{ParamSource, Res, Target, TypeInfo};

pub struct GenOptions {
    /// Version string recorded in the output (the toolchain version).
    pub jax_version: String,
    /// Also compile `test "..." { }` blocks.
    pub include_tests: bool,
}

/// Generate bytecode for a program that passed type checking.
pub fn generate<'a>(program: &'a Program, info: &'a TypeInfo, opts: &GenOptions) -> Result<Bytecode, String> {
    let mut cx = Cx {
        info,
        strings: Vec::new(),
        string_ix: HashMap::new(),
        fn_index: HashMap::new(),
        consts: HashMap::new(),
        fn_decls: HashMap::new(),
    };

    enum Work<'a> {
        Func(&'a FnDecl),
        Test(&'a Block),
    }
    let mut work: Vec<(String, Work<'a>)> = Vec::new();

    for item in &program.items {
        match &item.kind {
            ItemKind::Fn(f) => {
                cx.fn_decls.insert(f.name.clone(), f);
                work.push((f.name.clone(), Work::Func(f)));
            }
            ItemKind::Impl(im) => {
                let TypeKind::Named { name: tname, .. } = &im.self_ty.kind else { continue };
                for it in &im.items {
                    if let ItemKind::Fn(f) = &it.kind {
                        if f.body.is_none() {
                            continue;
                        }
                        let mangled = format!("{tname}.{}", f.name);
                        cx.fn_decls.insert(mangled.clone(), f);
                        work.push((mangled, Work::Func(f)));
                    }
                }
            }
            ItemKind::Const(c) => {
                cx.consts.insert(c.name.as_str(), &c.value);
            }
            ItemKind::Test(t) if opts.include_tests => {
                work.push((format!("test \"{}\"", t.name), Work::Test(&t.body)));
            }
            _ => {}
        }
    }
    for (i, (name, _)) in work.iter().enumerate() {
        cx.fn_index.insert(name.clone(), i as u32);
    }

    let mut functions = Vec::new();
    let mut tests = Vec::new();
    for (i, (name, w)) in work.iter().enumerate() {
        let gen = FnGen::new(&mut cx);
        let f = match w {
            Work::Func(decl) => gen.gen_fn(name, decl)?,
            Work::Test(body) => {
                tests.push((name.trim_start_matches("test \"").trim_end_matches('"').to_string(), i as u32));
                gen.gen_test(name, body)?
            }
        };
        functions.push(f);
    }

    let entry = if info.has_main { cx.fn_index.get("main").copied() } else { None };
    let program = Bytecode {
        jax_version: opts.jax_version.clone(),
        strings: cx.strings,
        functions,
        entry,
        tests,
    };
    program.validate()?;
    Ok(program)
}

struct Cx<'a> {
    info: &'a TypeInfo,
    strings: Vec<String>,
    string_ix: HashMap<String, u32>,
    fn_index: HashMap<String, u32>,
    consts: HashMap<&'a str, &'a Expr>,
    fn_decls: HashMap<String, &'a FnDecl>,
}

impl<'a> Cx<'a> {
    fn intern(&mut self, s: &str) -> u32 {
        if let Some(i) = self.string_ix.get(s) {
            return *i;
        }
        let i = self.strings.len() as u32;
        self.strings.push(s.to_string());
        self.string_ix.insert(s.to_string(), i);
        i
    }
}

struct LoopCtx {
    breaks: Vec<usize>,
    continues: Vec<usize>,
}

struct FnGen<'a, 'c> {
    cx: &'c mut Cx<'a>,
    code: Vec<Op>,
    scopes: Vec<HashMap<String, u32>>,
    nlocals: u32,
    loops: Vec<LoopCtx>,
    mut_self: bool,
    err: Option<String>,
}

fn binop(op: BinOp) -> Op {
    match op {
        BinOp::Add => Op::Add,
        BinOp::Sub => Op::Sub,
        BinOp::Mul => Op::Mul,
        BinOp::Div => Op::Div,
        BinOp::Rem => Op::Rem,
        BinOp::Eq => Op::Eq,
        BinOp::Ne => Op::Ne,
        BinOp::Lt => Op::Lt,
        BinOp::Le => Op::Le,
        BinOp::Gt => Op::Gt,
        BinOp::Ge => Op::Ge,
        BinOp::And | BinOp::Or | BinOp::Coalesce => unreachable!("handled separately"),
    }
}

impl<'a, 'c> FnGen<'a, 'c> {
    fn new(cx: &'c mut Cx<'a>) -> FnGen<'a, 'c> {
        FnGen {
            cx,
            code: Vec::new(),
            scopes: vec![HashMap::new()],
            nlocals: 0,
            loops: Vec::new(),
            mut_self: false,
            err: None,
        }
    }

    fn fail(&mut self, msg: impl Into<String>) {
        if self.err.is_none() {
            self.err = Some(msg.into());
        }
    }

    fn emit(&mut self, op: Op) -> usize {
        self.code.push(op);
        self.code.len() - 1
    }

    fn here(&self) -> u32 {
        self.code.len() as u32
    }

    fn patch(&mut self, at: usize, target: u32) {
        match &mut self.code[at] {
            Op::Jump(t) | Op::JumpIfFalse(t) | Op::JumpIfTrue(t) | Op::JumpIfNotTag(_, t) => *t = target,
            other => panic!("patching a non-jump instruction: {other:?}"),
        }
    }

    fn patch_all(&mut self, sites: Vec<usize>, target: u32) {
        for s in sites {
            self.patch(s, target);
        }
    }

    fn alloc(&mut self) -> u32 {
        let s = self.nlocals;
        self.nlocals += 1;
        s
    }

    fn declare(&mut self, name: &str) -> u32 {
        let s = self.alloc();
        self.scopes.last_mut().unwrap().insert(name.to_string(), s);
        s
    }

    fn local(&self, name: &str) -> Option<u32> {
        self.scopes.iter().rev().find_map(|s| s.get(name).copied())
    }

    fn text(&mut self, s: &str) -> u32 {
        self.cx.intern(s)
    }

    fn finish(self, name: &str, arity: u32) -> Result<Function, String> {
        if let Some(e) = self.err {
            return Err(format!("code generation failed in `{name}`: {e}"));
        }
        Ok(Function {
            name: name.to_string(),
            arity,
            locals: self.nlocals,
            code: self.code,
        })
    }

    fn gen_fn(mut self, name: &str, decl: &'a FnDecl) -> Result<Function, String> {
        if decl.self_param.is_some() {
            self.declare("self");
        }
        for p in &decl.params {
            self.declare(&p.name);
        }
        let arity = self.nlocals;
        self.mut_self = decl.self_param == Some(SelfKind::Mut);
        let body = decl.body.as_ref().ok_or_else(|| format!("`{name}` has no body"))?;
        self.block(body);
        if self.mut_self {
            self.emit(Op::Pop);
            self.emit(Op::LoadLocal(0));
        }
        self.emit(Op::Ret);
        self.finish(name, arity)
    }

    fn gen_test(mut self, name: &str, body: &'a Block) -> Result<Function, String> {
        self.block(body);
        self.emit(Op::Pop);
        self.emit(Op::PushUnit);
        self.emit(Op::Ret);
        self.finish(name, 0)
    }

    // ----- blocks --------------------------------------------------------

    fn block(&mut self, b: &'a Block) {
        self.scopes.push(HashMap::new());
        for s in &b.stmts {
            match &s.kind {
                StmtKind::Let { name, init, .. } => {
                    self.expr(init);
                    let slot = self.declare(name);
                    self.emit(Op::StoreLocal(slot));
                }
                StmtKind::Expr(e) => {
                    self.expr(e);
                    self.emit(Op::Pop);
                }
            }
        }
        match &b.tail {
            Some(e) => self.expr(e),
            None => {
                self.emit(Op::PushUnit);
            }
        }
        self.scopes.pop();
    }

    // ----- expressions ---------------------------------------------------

    fn expr(&mut self, e: &'a Expr) {
        match &e.kind {
            ExprKind::Int(n) => {
                self.emit(Op::PushInt(*n));
            }
            ExprKind::Float(f) => {
                self.emit(Op::PushFloat(*f));
            }
            ExprKind::Bool(b) => {
                self.emit(Op::PushBool(*b));
            }
            ExprKind::Str(s) => {
                let i = self.text(s);
                self.emit(Op::PushText(i));
            }
            ExprKind::None => {
                self.emit(Op::PushNone);
            }
            ExprKind::Interp(parts) => {
                let mut n = 0;
                for p in parts {
                    match p {
                        StrPart::Lit(s) => {
                            if !s.is_empty() {
                                let i = self.text(s);
                                self.emit(Op::PushText(i));
                                n += 1;
                            }
                        }
                        StrPart::Expr(x) => {
                            self.expr(x);
                            n += 1;
                        }
                    }
                }
                if n == 0 {
                    let i = self.text("");
                    self.emit(Op::PushText(i));
                } else {
                    self.emit(Op::Concat(n));
                }
            }
            ExprKind::Ident(name) => {
                if let Some(Res::Const(c)) = self.cx.info.res.get(&e.id) {
                    match self.cx.consts.get(c.as_str()).copied() {
                        Some(ce) => self.expr(ce),
                        None => self.fail(format!("unknown constant `{c}`")),
                    }
                } else if let Some(slot) = self.local(name) {
                    self.emit(Op::LoadLocal(slot));
                } else {
                    self.fail(format!("unresolved identifier `{name}`"));
                }
            }
            ExprKind::Unary { op, expr } => {
                self.expr(expr);
                self.emit(match op {
                    UnOp::Neg => Op::Neg,
                    UnOp::Not => Op::Not,
                });
            }
            ExprKind::Binary { op, lhs, rhs } => self.binary(*op, lhs, rhs),
            ExprKind::Assign { op, target, value } => {
                match op {
                    None => self.expr(value),
                    Some(bop) => {
                        self.expr(target);
                        self.expr(value);
                        self.emit(binop(*bop));
                    }
                }
                self.store_place(target);
                self.emit(Op::PushUnit);
            }
            ExprKind::Call { callee, args } => self.call(e, callee, args),
            ExprKind::Field { obj, .. } => match self.cx.info.res.get(&e.id) {
                Some(Res::VariantUnit { tag }) => {
                    let tag = *tag;
                    self.emit(Op::MakeEnum(tag, 0));
                }
                Some(Res::Field(i)) => {
                    let i = *i as u32;
                    self.expr(obj);
                    self.emit(Op::GetField(i));
                }
                _ => self.fail("unresolved field access"),
            },
            ExprKind::OptField { obj, .. } => {
                let Some(Res::Field(i)) = self.cx.info.res.get(&e.id) else {
                    self.fail("unresolved optional field access");
                    return;
                };
                let i = *i as u32;
                self.expr(obj);
                self.emit(Op::Dup);
                self.emit(Op::IsNone);
                let jn = self.emit(Op::JumpIfTrue(0));
                self.emit(Op::GetField(i));
                let end = self.here();
                self.patch(jn, end);
            }
            ExprKind::Index { obj, index } => {
                self.expr(obj);
                self.expr(index);
                self.emit(Op::Index);
            }
            ExprKind::If { cond, then, els } => {
                self.expr(cond);
                let jf = self.emit(Op::JumpIfFalse(0));
                self.block(then);
                let je = self.emit(Op::Jump(0));
                let else_at = self.here();
                self.patch(jf, else_at);
                match els {
                    Some(el) => self.expr(el),
                    None => {
                        self.emit(Op::PushUnit);
                    }
                }
                let end = self.here();
                self.patch(je, end);
            }
            ExprKind::IfLet { pat, value, then, els } => {
                self.expr(value);
                let slot = self.alloc();
                self.emit(Op::StoreLocal(slot));
                self.scopes.push(HashMap::new());
                let mut fails = Vec::new();
                if matches!(self.cx.info.res.get(&e.id), Some(Res::IfLetOpt)) {
                    self.emit(Op::LoadLocal(slot));
                    self.emit(Op::IsNone);
                    fails.push(self.emit(Op::JumpIfTrue(0)));
                }
                self.pat(pat, slot, &mut fails);
                self.block(then);
                self.scopes.pop();
                let je = self.emit(Op::Jump(0));
                let else_at = self.here();
                self.patch_all(fails, else_at);
                match els {
                    Some(el) => self.expr(el),
                    None => {
                        self.emit(Op::PushUnit);
                    }
                }
                let end = self.here();
                self.patch(je, end);
            }
            ExprKind::Match { scrutinee, arms } => self.match_expr(scrutinee, arms),
            ExprKind::Block(b) => self.block(b),
            ExprKind::While { cond, body } => {
                let start = self.here();
                self.expr(cond);
                let jf = self.emit(Op::JumpIfFalse(0));
                self.loops.push(LoopCtx {
                    breaks: vec![],
                    continues: vec![],
                });
                self.block(body);
                self.emit(Op::Pop);
                self.emit(Op::Jump(start));
                let exit = self.here();
                let ctx = self.loops.pop().unwrap();
                self.patch(jf, exit);
                self.patch_all(ctx.breaks, exit);
                self.patch_all(ctx.continues, start);
                self.emit(Op::PushUnit);
            }
            ExprKind::Loop(body) => {
                let start = self.here();
                self.loops.push(LoopCtx {
                    breaks: vec![],
                    continues: vec![],
                });
                self.block(body);
                self.emit(Op::Pop);
                self.emit(Op::Jump(start));
                let exit = self.here();
                let ctx = self.loops.pop().unwrap();
                self.patch_all(ctx.breaks, exit);
                self.patch_all(ctx.continues, start);
                self.emit(Op::PushUnit);
            }
            ExprKind::For { var, iter, body, .. } => self.for_loop(var, iter, body),
            ExprKind::Return(v) => {
                if self.mut_self {
                    self.emit(Op::LoadLocal(0));
                } else {
                    match v {
                        Some(x) => self.expr(x),
                        None => {
                            self.emit(Op::PushUnit);
                        }
                    }
                }
                self.emit(Op::Ret);
                self.emit(Op::PushUnit);
            }
            ExprKind::Break => {
                let j = self.emit(Op::Jump(0));
                match self.loops.last_mut() {
                    Some(l) => l.breaks.push(j),
                    None => self.fail("`break` outside a loop"),
                }
                self.emit(Op::PushUnit);
            }
            ExprKind::Continue => {
                let j = self.emit(Op::Jump(0));
                match self.loops.last_mut() {
                    Some(l) => l.continues.push(j),
                    None => self.fail("`continue` outside a loop"),
                }
                self.emit(Op::PushUnit);
            }
            ExprKind::List(elems) => {
                for x in elems {
                    self.expr(x);
                }
                self.emit(Op::MakeList(elems.len() as u32));
            }
            ExprKind::Unwrap(inner) => {
                self.expr(inner);
                self.emit(Op::Dup);
                self.emit(Op::IsNone);
                let jf = self.emit(Op::JumpIfFalse(0));
                let msg = self.text("unwrapped a `none` value with `!`");
                self.emit(Op::Panic(msg));
                let ok = self.here();
                self.patch(jf, ok);
            }
            ExprKind::With { base, fields } => {
                let Some(Res::With(idxs)) = self.cx.info.res.get(&e.id) else {
                    self.fail("unresolved `with` expression");
                    return;
                };
                let idxs = idxs.clone();
                self.expr(base);
                for ((_, _, v), idx) in fields.iter().zip(idxs) {
                    self.expr(v);
                    self.emit(Op::SetField(idx as u32));
                }
            }
            ExprKind::Range { .. }
            | ExprKind::Tuple(_)
            | ExprKind::Closure { .. }
            | ExprKind::Try(_)
            | ExprKind::Await(_) => self.fail("unsupported expression reached code generation"),
        }
    }

    fn binary(&mut self, op: BinOp, lhs: &'a Expr, rhs: &'a Expr) {
        match op {
            BinOp::And => {
                self.expr(lhs);
                let jf = self.emit(Op::JumpIfFalse(0));
                self.expr(rhs);
                let je = self.emit(Op::Jump(0));
                let f = self.here();
                self.patch(jf, f);
                self.emit(Op::PushBool(false));
                let end = self.here();
                self.patch(je, end);
            }
            BinOp::Or => {
                self.expr(lhs);
                let jt = self.emit(Op::JumpIfTrue(0));
                self.expr(rhs);
                let je = self.emit(Op::Jump(0));
                let t = self.here();
                self.patch(jt, t);
                self.emit(Op::PushBool(true));
                let end = self.here();
                self.patch(je, end);
            }
            BinOp::Coalesce => {
                self.expr(lhs);
                self.emit(Op::Dup);
                self.emit(Op::IsNone);
                let jf = self.emit(Op::JumpIfFalse(0));
                self.emit(Op::Pop);
                self.expr(rhs);
                let have = self.here();
                self.patch(jf, have);
            }
            _ => {
                self.expr(lhs);
                self.expr(rhs);
                self.emit(binop(op));
            }
        }
    }

    /// Store the value on top of the stack into `place`, rebuilding any
    /// enclosing structs/lists (values are immutable, updates are copies).
    fn store_place(&mut self, place: &'a Expr) {
        match &place.kind {
            ExprKind::Ident(n) => match self.local(n) {
                Some(slot) => {
                    self.emit(Op::StoreLocal(slot));
                }
                None => self.fail(format!("cannot assign to `{n}`")),
            },
            ExprKind::Field { obj, .. } => {
                let Some(Res::Field(i)) = self.cx.info.res.get(&place.id) else {
                    self.fail("unresolved field in assignment");
                    return;
                };
                let i = *i as u32;
                let tmp = self.alloc();
                self.emit(Op::StoreLocal(tmp));
                self.expr(obj);
                self.emit(Op::LoadLocal(tmp));
                self.emit(Op::SetField(i));
                self.store_place(obj);
            }
            ExprKind::Index { obj, index } => {
                let tmp = self.alloc();
                self.emit(Op::StoreLocal(tmp));
                self.expr(obj);
                self.expr(index);
                self.emit(Op::LoadLocal(tmp));
                self.emit(Op::SetIndex);
                self.store_place(obj);
            }
            _ => self.fail("invalid assignment target"),
        }
    }

    fn call(&mut self, e: &'a Expr, callee: &'a Expr, args: &'a [Arg]) {
        let plan = match self.cx.info.res.get(&e.id) {
            Some(Res::Call(p)) => p.clone(),
            _ => {
                self.fail("unresolved call");
                return;
            }
        };
        let receiver: Option<&'a Expr> = if plan.has_receiver {
            match &callee.kind {
                ExprKind::Field { obj, .. } => Some(obj),
                _ => None,
            }
        } else {
            None
        };
        if let Some(r) = receiver {
            self.expr(r);
        }
        for src in &plan.order {
            match src {
                ParamSource::Arg(i) => self.expr(&args[*i].value),
                ParamSource::Default { func, index } => {
                    let decl: Option<&'a FnDecl> = self.cx.fn_decls.get(func).copied();
                    match decl.and_then(|d| d.params.get(*index)).and_then(|p| p.default.as_ref()) {
                        Some(d) => self.expr(d),
                        None => self.fail(format!("missing default value for parameter {index} of `{func}`")),
                    }
                }
            }
        }
        let argc = (plan.order.len() + receiver.is_some() as usize) as u32;
        match &plan.target {
            Target::Func(n) | Target::Method(n) => match self.cx.fn_index.get(n) {
                Some(i) => {
                    let i = *i;
                    self.emit(Op::Call(i, argc));
                }
                None => self.fail(format!("call to unknown function `{n}`")),
            },
            Target::StructNew(_) => {
                self.emit(Op::MakeStruct(argc));
            }
            Target::VariantNew { tag, .. } => {
                self.emit(Op::MakeEnum(*tag, argc));
            }
            Target::Builtin(b) => {
                self.emit(Op::Builtin(*b, argc));
                if *b == Builtin::Panic {
                    // `panic` never returns; nothing more to do.
                }
            }
        }
        if plan.writeback {
            match receiver {
                Some(r) => {
                    self.store_place(r);
                    self.emit(Op::PushUnit);
                }
                None => self.fail("write-back call without a receiver"),
            }
        }
    }

    fn for_loop(&mut self, var: &str, iter: &'a Expr, body: &'a Block) {
        self.loops.push(LoopCtx {
            breaks: vec![],
            continues: vec![],
        });
        self.scopes.push(HashMap::new());
        let exit_jump;
        let cont_target;
        let loop_start;
        let counter;
        if let ExprKind::Range { lo, hi, inclusive } = &iter.kind {
            let ctr = self.alloc();
            let end = self.alloc();
            self.expr(lo);
            self.emit(Op::StoreLocal(ctr));
            self.expr(hi);
            self.emit(Op::StoreLocal(end));
            loop_start = self.here();
            self.emit(Op::LoadLocal(ctr));
            self.emit(Op::LoadLocal(end));
            self.emit(if *inclusive { Op::Le } else { Op::Lt });
            exit_jump = self.emit(Op::JumpIfFalse(0));
            self.scopes.last_mut().unwrap().insert(var.to_string(), ctr);
            counter = ctr;
            self.block(body);
            self.emit(Op::Pop);
            cont_target = self.here();
        } else {
            let lst = self.alloc();
            let idx = self.alloc();
            self.expr(iter);
            self.emit(Op::StoreLocal(lst));
            self.emit(Op::PushInt(0));
            self.emit(Op::StoreLocal(idx));
            loop_start = self.here();
            self.emit(Op::LoadLocal(idx));
            self.emit(Op::LoadLocal(lst));
            self.emit(Op::Builtin(Builtin::Len, 1));
            self.emit(Op::Lt);
            exit_jump = self.emit(Op::JumpIfFalse(0));
            let v = self.declare(var);
            self.emit(Op::LoadLocal(lst));
            self.emit(Op::LoadLocal(idx));
            self.emit(Op::Index);
            self.emit(Op::StoreLocal(v));
            counter = idx;
            self.block(body);
            self.emit(Op::Pop);
            cont_target = self.here();
        }
        self.emit(Op::LoadLocal(counter));
        self.emit(Op::PushInt(1));
        self.emit(Op::Add);
        self.emit(Op::StoreLocal(counter));
        self.emit(Op::Jump(loop_start));
        let exit = self.here();
        let ctx = self.loops.pop().unwrap();
        self.scopes.pop();
        self.patch(exit_jump, exit);
        self.patch_all(ctx.breaks, exit);
        self.patch_all(ctx.continues, cont_target);
        self.emit(Op::PushUnit);
    }

    // ----- match ---------------------------------------------------------

    fn match_expr(&mut self, scrutinee: &'a Expr, arms: &'a [MatchArm]) {
        self.expr(scrutinee);
        let slot = self.alloc();
        self.emit(Op::StoreLocal(slot));
        let mut end_jumps = Vec::new();
        for arm in arms {
            self.scopes.push(HashMap::new());
            let mut fails = Vec::new();
            self.pat(&arm.pat, slot, &mut fails);
            if let Some(g) = &arm.guard {
                self.expr(g);
                fails.push(self.emit(Op::JumpIfFalse(0)));
            }
            self.expr(&arm.body);
            end_jumps.push(self.emit(Op::Jump(0)));
            let next = self.here();
            self.patch_all(fails, next);
            self.scopes.pop();
        }
        let msg = self.text("no `match` arm matched (non-exhaustive match)");
        self.emit(Op::Panic(msg));
        let end = self.here();
        self.patch_all(end_jumps, end);
    }

    /// Emit a test of the value in `slot` against `p`. Jumps to a site in
    /// `fails` when it does not match; binds pattern variables when it does.
    fn pat(&mut self, p: &'a Pattern, slot: u32, fails: &mut Vec<usize>) {
        match &p.kind {
            PatKind::Wild => {}
            PatKind::Ident(name) => match self.cx.info.res.get(&p.id) {
                Some(Res::PatVariant { tag }) => {
                    let tag = *tag;
                    self.emit(Op::LoadLocal(slot));
                    fails.push(self.emit(Op::JumpIfNotTag(tag, 0)));
                }
                _ => {
                    self.emit(Op::LoadLocal(slot));
                    let s = self.declare(name);
                    self.emit(Op::StoreLocal(s));
                }
            },
            PatKind::Int(n) => {
                self.emit(Op::LoadLocal(slot));
                self.emit(Op::PushInt(*n));
                self.emit(Op::Eq);
                fails.push(self.emit(Op::JumpIfFalse(0)));
            }
            PatKind::Float(f) => {
                self.emit(Op::LoadLocal(slot));
                self.emit(Op::PushFloat(*f));
                self.emit(Op::Eq);
                fails.push(self.emit(Op::JumpIfFalse(0)));
            }
            PatKind::Bool(b) => {
                self.emit(Op::LoadLocal(slot));
                self.emit(Op::PushBool(*b));
                self.emit(Op::Eq);
                fails.push(self.emit(Op::JumpIfFalse(0)));
            }
            PatKind::Str(s) => {
                let i = self.text(s);
                self.emit(Op::LoadLocal(slot));
                self.emit(Op::PushText(i));
                self.emit(Op::Eq);
                fails.push(self.emit(Op::JumpIfFalse(0)));
            }
            PatKind::None => {
                self.emit(Op::LoadLocal(slot));
                self.emit(Op::IsNone);
                fails.push(self.emit(Op::JumpIfFalse(0)));
            }
            PatKind::Variant { args, .. } => match self.cx.info.res.get(&p.id) {
                Some(Res::PatSome) => {
                    self.emit(Op::LoadLocal(slot));
                    self.emit(Op::IsNone);
                    fails.push(self.emit(Op::JumpIfTrue(0)));
                    if let Some(a) = args.first() {
                        self.pat(a, slot, fails);
                    }
                }
                Some(Res::PatVariant { tag }) => {
                    let tag = *tag;
                    self.emit(Op::LoadLocal(slot));
                    fails.push(self.emit(Op::JumpIfNotTag(tag, 0)));
                    for (i, a) in args.iter().enumerate() {
                        if matches!(a.kind, PatKind::Wild) {
                            continue;
                        }
                        let tmp = self.alloc();
                        self.emit(Op::LoadLocal(slot));
                        self.emit(Op::EnumField(i as u32));
                        self.emit(Op::StoreLocal(tmp));
                        self.pat(a, tmp, fails);
                    }
                }
                _ => self.fail("unresolved variant pattern"),
            },
            PatKind::Range { lo, hi, inclusive } => {
                self.emit(Op::LoadLocal(slot));
                self.emit(Op::PushInt(*lo));
                self.emit(Op::Ge);
                fails.push(self.emit(Op::JumpIfFalse(0)));
                self.emit(Op::LoadLocal(slot));
                self.emit(Op::PushInt(*hi));
                self.emit(if *inclusive { Op::Le } else { Op::Lt });
                fails.push(self.emit(Op::JumpIfFalse(0)));
            }
            PatKind::Or(alts) => {
                let mut oks = Vec::new();
                for alt in alts {
                    let mut alt_fails = Vec::new();
                    self.pat(alt, slot, &mut alt_fails);
                    oks.push(self.emit(Op::Jump(0)));
                    let next = self.here();
                    self.patch_all(alt_fails, next);
                }
                fails.push(self.emit(Op::Jump(0)));
                let ok = self.here();
                self.patch_all(oks, ok);
            }
            PatKind::Tuple(_) => self.fail("tuple patterns are not supported"),
        }
    }
}
