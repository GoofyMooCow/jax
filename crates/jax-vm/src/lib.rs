//! The Jax virtual machine: executes `jax-bytecode` programs.
//!
//! Values have value semantics: lists and structs are reference counted and
//! copied on write, so assigning or passing them never aliases.

use std::fmt;
use std::io::Write;
use std::rc::Rc;

use jax_bytecode::{Builtin, Op, Program};

/// Maximum call depth before the VM reports a stack overflow.
pub const MAX_CALL_DEPTH: usize = 200_000;

#[derive(Clone, Debug)]
pub enum Value {
    Unit,
    None,
    Int(i64),
    Float(f64),
    Bool(bool),
    Text(Rc<str>),
    List(Rc<Vec<Value>>),
    Struct(Rc<Vec<Value>>),
    Enum(Rc<EnumVal>),
}

#[derive(Clone, Debug)]
pub struct EnumVal {
    pub tag: u32,
    pub fields: Vec<Value>,
}

#[derive(Debug)]
pub enum RunError {
    Panic { message: String, trace: Vec<String> },
    Io(String),
    Internal(String),
}

impl fmt::Display for RunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RunError::Panic { message, trace } => {
                write!(f, "panic: {message}")?;
                for t in trace {
                    write!(f, "\n  at {t}")?;
                }
                Ok(())
            }
            RunError::Io(m) => write!(f, "i/o error: {m}"),
            RunError::Internal(m) => write!(f, "internal error: {m}"),
        }
    }
}

impl std::error::Error for RunError {}

impl RunError {
    /// Process exit code for a program that ended with this error.
    pub fn exit_code(&self) -> i32 {
        match self {
            RunError::Panic { .. } => 101,
            _ => 1,
        }
    }
}

#[derive(Clone, Copy)]
struct Frame {
    func: usize,
    ip: usize,
    base: usize,
}

pub struct Vm<'p> {
    prog: &'p Program,
    strings: Vec<Rc<str>>,
    stack: Vec<Value>,
    frames: Vec<Frame>,
    out: Box<dyn Write + 'p>,
}

impl<'p> Vm<'p> {
    pub fn new(prog: &'p Program, out: Box<dyn Write + 'p>) -> Vm<'p> {
        Vm {
            prog,
            strings: prog.strings.iter().map(|s| Rc::from(s.as_str())).collect(),
            stack: Vec::with_capacity(1024),
            frames: Vec::new(),
            out,
        }
    }

    /// Run the program's `main`.
    pub fn run_entry(&mut self) -> Result<(), RunError> {
        let entry = self
            .prog
            .entry
            .ok_or_else(|| RunError::Internal("program has no entry point (`fn main`)".into()))?;
        self.run_function(entry).map(|_| ())
    }

    /// Run a zero-argument function by index and return its result.
    pub fn run_function(&mut self, idx: u32) -> Result<Value, RunError> {
        self.stack.clear();
        self.frames.clear();
        let func = self
            .prog
            .functions
            .get(idx as usize)
            .ok_or_else(|| RunError::Internal("function index out of range".into()))?;
        if func.arity != 0 {
            return Err(RunError::Internal(format!(
                "`{}` takes arguments and cannot be run directly",
                func.name
            )));
        }
        for _ in 0..func.locals {
            self.stack.push(Value::Unit);
        }
        self.frames.push(Frame {
            func: idx as usize,
            ip: 0,
            base: 0,
        });
        let result = self.exec();
        let flushed = self.out.flush();
        let v = result?;
        flushed.map_err(|e| RunError::Io(e.to_string()))?;
        Ok(v)
    }

    fn pop(&mut self) -> Value {
        self.stack.pop().expect("VM stack underflow (compiler bug)")
    }

    fn trace(&self, current: usize) -> Vec<String> {
        let mut names = vec![self.prog.functions[current].name.clone()];
        for f in self.frames.iter().rev().skip(1) {
            names.push(self.prog.functions[f.func].name.clone());
        }
        if names.len() > 12 {
            let extra = names.len() - 10;
            names.truncate(10);
            names.push(format!("... {extra} more"));
        }
        names
    }

    fn panic_at(&self, current: usize, message: impl Into<String>) -> RunError {
        RunError::Panic {
            message: message.into(),
            trace: self.trace(current),
        }
    }

    fn exec(&mut self) -> Result<Value, RunError> {
        let prog = self.prog;
        let mut fr = *self.frames.last().unwrap();
        let mut code: &[Op] = &prog.functions[fr.func].code;

        macro_rules! fail {
            ($msg:expr) => {
                return Err(self.panic_at(fr.func, $msg))
            };
        }

        loop {
            let Some(&op) = code.get(fr.ip) else {
                return Err(RunError::Internal(format!(
                    "ran off the end of function `{}`",
                    prog.functions[fr.func].name
                )));
            };
            fr.ip += 1;
            match op {
                Op::PushUnit => self.stack.push(Value::Unit),
                Op::PushNone => self.stack.push(Value::None),
                Op::PushBool(b) => self.stack.push(Value::Bool(b)),
                Op::PushInt(n) => self.stack.push(Value::Int(n)),
                Op::PushFloat(f) => self.stack.push(Value::Float(f)),
                Op::PushText(i) => self.stack.push(Value::Text(self.strings[i as usize].clone())),
                Op::Pop => {
                    self.pop();
                }
                Op::Dup => {
                    let v = self.stack.last().expect("VM stack underflow (compiler bug)").clone();
                    self.stack.push(v);
                }
                Op::LoadLocal(i) => {
                    let v = self.stack[fr.base + i as usize].clone();
                    self.stack.push(v);
                }
                Op::StoreLocal(i) => {
                    let v = self.pop();
                    self.stack[fr.base + i as usize] = v;
                }

                Op::Add | Op::Sub | Op::Mul | Op::Div | Op::Rem => {
                    let b = self.pop();
                    let a = self.pop();
                    match arith(op, a, b) {
                        Ok(v) => self.stack.push(v),
                        Err(m) => fail!(m),
                    }
                }
                Op::Neg => match self.pop() {
                    Value::Int(n) => match n.checked_neg() {
                        Some(r) => self.stack.push(Value::Int(r)),
                        None => fail!("integer overflow"),
                    },
                    Value::Float(f) => self.stack.push(Value::Float(-f)),
                    _ => fail!("cannot negate this value"),
                },
                Op::Not => match self.pop() {
                    Value::Bool(b) => self.stack.push(Value::Bool(!b)),
                    _ => fail!("`!` needs a Bool"),
                },
                Op::Eq | Op::Ne => {
                    let b = self.pop();
                    let a = self.pop();
                    let eq = values_eq(&a, &b);
                    self.stack.push(Value::Bool(if op == Op::Eq { eq } else { !eq }));
                }
                Op::Lt | Op::Le | Op::Gt | Op::Ge => {
                    let b = self.pop();
                    let a = self.pop();
                    match compare(op, &a, &b) {
                        Some(r) => self.stack.push(Value::Bool(r)),
                        None => fail!("values cannot be compared"),
                    }
                }

                Op::Jump(t) => fr.ip = t as usize,
                Op::JumpIfFalse(t) => {
                    if let Value::Bool(false) = self.pop() {
                        fr.ip = t as usize;
                    }
                }
                Op::JumpIfTrue(t) => {
                    if let Value::Bool(true) = self.pop() {
                        fr.ip = t as usize;
                    }
                }

                Op::Call(f, argc) => {
                    if self.frames.len() >= MAX_CALL_DEPTH {
                        fail!(format!("stack overflow (call depth exceeded {MAX_CALL_DEPTH})"));
                    }
                    *self.frames.last_mut().unwrap() = fr;
                    let func = &prog.functions[f as usize];
                    let base = self.stack.len() - argc as usize;
                    for _ in argc..func.locals {
                        self.stack.push(Value::Unit);
                    }
                    fr = Frame {
                        func: f as usize,
                        ip: 0,
                        base,
                    };
                    self.frames.push(fr);
                    code = &func.code;
                }
                Op::Ret => {
                    let v = self.pop();
                    self.stack.truncate(fr.base);
                    self.frames.pop();
                    match self.frames.last() {
                        None => return Ok(v),
                        Some(&caller) => {
                            self.stack.push(v);
                            fr = caller;
                            code = &prog.functions[fr.func].code;
                        }
                    }
                }
                Op::Builtin(b, argc) => {
                    let at = self.stack.len() - argc as usize;
                    let args: Vec<Value> = self.stack.drain(at..).collect();
                    match self.builtin(b, args) {
                        Ok(v) => self.stack.push(v),
                        Err(BuiltinError::Panic(m)) => fail!(m),
                        Err(BuiltinError::Io(m)) => return Err(RunError::Io(m)),
                    }
                }

                Op::MakeList(n) => {
                    let at = self.stack.len() - n as usize;
                    let items: Vec<Value> = self.stack.drain(at..).collect();
                    self.stack.push(Value::List(Rc::new(items)));
                }
                Op::Index => {
                    let i = self.pop();
                    let l = self.pop();
                    match (l, i) {
                        (Value::List(l), Value::Int(i)) => match usize::try_from(i).ok().and_then(|i| l.get(i)) {
                            Some(v) => self.stack.push(v.clone()),
                            None => fail!(format!("index {i} out of range (length {})", l.len())),
                        },
                        _ => fail!("cannot index this value"),
                    }
                }
                Op::SetIndex => {
                    let v = self.pop();
                    let i = self.pop();
                    let l = self.pop();
                    match (l, i) {
                        (Value::List(mut l), Value::Int(i)) => {
                            let len = l.len();
                            match usize::try_from(i).ok().filter(|i| *i < len) {
                                Some(i) => {
                                    Rc::make_mut(&mut l)[i] = v;
                                    self.stack.push(Value::List(l));
                                }
                                None => fail!(format!("index {i} out of range (length {len})")),
                            }
                        }
                        _ => fail!("cannot index this value"),
                    }
                }
                Op::MakeStruct(n) => {
                    let at = self.stack.len() - n as usize;
                    let fields: Vec<Value> = self.stack.drain(at..).collect();
                    self.stack.push(Value::Struct(Rc::new(fields)));
                }
                Op::GetField(i) => match self.pop() {
                    Value::Struct(s) => self.stack.push(s[i as usize].clone()),
                    _ => fail!("value has no fields"),
                },
                Op::SetField(i) => {
                    let v = self.pop();
                    match self.pop() {
                        Value::Struct(mut s) => {
                            Rc::make_mut(&mut s)[i as usize] = v;
                            self.stack.push(Value::Struct(s));
                        }
                        _ => fail!("value has no fields"),
                    }
                }
                Op::MakeEnum(tag, n) => {
                    let at = self.stack.len() - n as usize;
                    let fields: Vec<Value> = self.stack.drain(at..).collect();
                    self.stack.push(Value::Enum(Rc::new(EnumVal { tag, fields })));
                }
                Op::JumpIfNotTag(tag, target) => match self.pop() {
                    Value::Enum(e) => {
                        if e.tag != tag {
                            fr.ip = target as usize;
                        }
                    }
                    _ => fail!("expected an enum value"),
                },
                Op::EnumField(i) => match self.pop() {
                    Value::Enum(e) => self.stack.push(e.fields[i as usize].clone()),
                    _ => fail!("expected an enum value"),
                },
                Op::IsNone => {
                    let v = self.pop();
                    self.stack.push(Value::Bool(matches!(v, Value::None)));
                }
                Op::Concat(n) => {
                    let at = self.stack.len() - n as usize;
                    let parts: Vec<Value> = self.stack.drain(at..).collect();
                    let mut s = String::new();
                    for p in &parts {
                        s.push_str(&display(p));
                    }
                    self.stack.push(Value::Text(Rc::from(s)));
                }
                Op::Panic(i) => fail!(prog.strings[i as usize].clone()),
            }
        }
    }

    fn builtin(&mut self, b: Builtin, mut args: Vec<Value>) -> Result<Value, BuiltinError> {
        let bad = |what: &str| BuiltinError::Panic(format!("internal error: bad arguments to `{what}`"));
        match b {
            Builtin::Print => {
                let s = display(args.first().ok_or_else(|| bad("print"))?);
                writeln!(self.out, "{s}").map_err(|e| BuiltinError::Io(e.to_string()))?;
                Ok(Value::Unit)
            }
            Builtin::Len => match args.first() {
                Some(Value::Text(t)) => Ok(Value::Int(t.chars().count() as i64)),
                Some(Value::List(l)) => Ok(Value::Int(l.len() as i64)),
                _ => Err(bad("len")),
            },
            Builtin::Push => {
                if args.len() != 2 {
                    return Err(bad("push"));
                }
                let v = args.pop().unwrap();
                match args.pop().unwrap() {
                    Value::List(mut l) => {
                        Rc::make_mut(&mut l).push(v);
                        Ok(Value::List(l))
                    }
                    _ => Err(bad("push")),
                }
            }
            Builtin::Get => match (args.first(), args.get(1)) {
                (Some(Value::List(l)), Some(Value::Int(i))) => Ok(usize::try_from(*i)
                    .ok()
                    .and_then(|i| l.get(i))
                    .cloned()
                    .unwrap_or(Value::None)),
                _ => Err(bad("get")),
            },
            Builtin::ToText => Ok(Value::Text(Rc::from(display(
                args.first().ok_or_else(|| bad("to_text"))?,
            )))),
            Builtin::ToFloat => match args.first() {
                Some(Value::Int(n)) => Ok(Value::Float(*n as f64)),
                _ => Err(bad("to_float")),
            },
            Builtin::ToInt => match args.first() {
                Some(Value::Float(f)) => {
                    if f.is_nan() {
                        Err(BuiltinError::Panic("cannot convert NaN to Int".into()))
                    } else {
                        Ok(Value::Int(*f as i64))
                    }
                }
                _ => Err(bad("to_int")),
            },
            Builtin::Sqrt => match args.first() {
                Some(Value::Float(f)) => Ok(Value::Float(f.sqrt())),
                _ => Err(bad("sqrt")),
            },
            Builtin::Assert => match args.first() {
                Some(Value::Bool(true)) => Ok(Value::Unit),
                Some(Value::Bool(false)) => Err(BuiltinError::Panic("assertion failed".into())),
                _ => Err(bad("assert")),
            },
            Builtin::Panic => match args.first() {
                Some(Value::Text(t)) => Err(BuiltinError::Panic(t.to_string())),
                _ => Err(bad("panic")),
            },
        }
    }
}

enum BuiltinError {
    Panic(String),
    Io(String),
}

fn arith(op: Op, a: Value, b: Value) -> Result<Value, String> {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => {
            let r = match op {
                Op::Add => x.checked_add(y),
                Op::Sub => x.checked_sub(y),
                Op::Mul => x.checked_mul(y),
                Op::Div => {
                    if y == 0 {
                        return Err("division by zero".into());
                    }
                    x.checked_div(y)
                }
                Op::Rem => {
                    if y == 0 {
                        return Err("remainder by zero".into());
                    }
                    x.checked_rem(y)
                }
                _ => unreachable!(),
            };
            r.map(Value::Int).ok_or_else(|| "integer overflow".to_string())
        }
        (Value::Float(x), Value::Float(y)) => Ok(Value::Float(match op {
            Op::Add => x + y,
            Op::Sub => x - y,
            Op::Mul => x * y,
            Op::Div => x / y,
            Op::Rem => x % y,
            _ => unreachable!(),
        })),
        (Value::Text(x), Value::Text(y)) if op == Op::Add => {
            let mut s = String::with_capacity(x.len() + y.len());
            s.push_str(&x);
            s.push_str(&y);
            Ok(Value::Text(Rc::from(s)))
        }
        _ => Err("invalid operands for arithmetic".into()),
    }
}

fn compare(op: Op, a: &Value, b: &Value) -> Option<bool> {
    use std::cmp::Ordering;
    let ord: Option<Ordering> = match (a, b) {
        (Value::Int(x), Value::Int(y)) => Some(x.cmp(y)),
        (Value::Float(x), Value::Float(y)) => x.partial_cmp(y),
        (Value::Text(x), Value::Text(y)) => Some(x.cmp(y)),
        _ => return None,
    };
    Some(match (op, ord) {
        (_, None) => false, // NaN comparisons are always false
        (Op::Lt, Some(o)) => o == Ordering::Less,
        (Op::Le, Some(o)) => o != Ordering::Greater,
        (Op::Gt, Some(o)) => o == Ordering::Greater,
        (Op::Ge, Some(o)) => o != Ordering::Less,
        _ => return None,
    })
}

/// Structural equality.
pub fn values_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Unit, Value::Unit) | (Value::None, Value::None) => true,
        (Value::Int(x), Value::Int(y)) => x == y,
        (Value::Float(x), Value::Float(y)) => x == y,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::Text(x), Value::Text(y)) => x == y,
        (Value::List(x), Value::List(y)) | (Value::Struct(x), Value::Struct(y)) => {
            x.len() == y.len() && x.iter().zip(y.iter()).all(|(p, q)| values_eq(p, q))
        }
        (Value::Enum(x), Value::Enum(y)) => {
            x.tag == y.tag
                && x.fields.len() == y.fields.len()
                && x.fields.iter().zip(y.fields.iter()).all(|(p, q)| values_eq(p, q))
        }
        _ => false,
    }
}

/// How `print`, `to_text` and string interpolation render a value.
pub fn display(v: &Value) -> String {
    match v {
        Value::Unit => "()".into(),
        Value::None => "none".into(),
        Value::Int(n) => n.to_string(),
        Value::Float(f) => format!("{f:?}"),
        Value::Bool(b) => b.to_string(),
        Value::Text(t) => t.to_string(),
        Value::List(l) => {
            let items: Vec<String> = l
                .iter()
                .map(|x| match x {
                    Value::Text(t) => format!("{t:?}"),
                    other => display(other),
                })
                .collect();
            format!("[{}]", items.join(", "))
        }
        Value::Struct(_) => "<struct>".into(),
        Value::Enum(_) => "<enum>".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jax_bytecode::Function;

    fn prog(code: Vec<Op>) -> Program {
        Program {
            jax_version: "test".into(),
            strings: vec![],
            functions: vec![Function {
                name: "main".into(),
                arity: 0,
                locals: 0,
                code,
            }],
            entry: Some(0),
            tests: vec![],
        }
    }

    fn run(p: &Program) -> (Result<(), RunError>, String) {
        let mut buf: Vec<u8> = Vec::new();
        let r = {
            let mut vm = Vm::new(p, Box::new(&mut buf));
            vm.run_entry()
        };
        (r, String::from_utf8(buf).unwrap())
    }

    #[test]
    fn prints_arithmetic() {
        let p = prog(vec![
            Op::PushInt(2),
            Op::PushInt(3),
            Op::Add,
            Op::Builtin(Builtin::Print, 1),
            Op::Pop,
            Op::PushUnit,
            Op::Ret,
        ]);
        let (r, out) = run(&p);
        r.unwrap();
        assert_eq!(out, "5\n");
    }

    #[test]
    fn division_by_zero_panics() {
        let p = prog(vec![Op::PushInt(1), Op::PushInt(0), Op::Div, Op::Ret]);
        let (r, _) = run(&p);
        match r {
            Err(RunError::Panic { message, .. }) => assert_eq!(message, "division by zero"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn display_floats() {
        assert_eq!(display(&Value::Float(1.0)), "1.0");
        assert_eq!(display(&Value::Float(0.5)), "0.5");
    }
}
