//! The Jax bytecode: the instruction set, the program container, its stable
//! binary encoding (`.jxb`) and a disassembler.
//!
//! This is the "Jax IR" of the stage-0 compiler. The same bytes are embedded
//! into standalone executables and shipped as portable bundles.

use std::fmt::Write as _;

/// Bumped whenever the binary layout or the meaning of an instruction changes.
pub const FORMAT_VERSION: u16 = 1;
pub const BUNDLE_MAGIC: &[u8; 4] = b"JAXB";
/// Trailer appended to executables that carry an embedded program.
pub const EXE_MAGIC: &[u8; 8] = b"JAXEXE01";

/// Built-in operations callable from Jax code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Builtin {
    Print = 0,
    Len = 1,
    /// `list.push(x)`: returns the new list (value semantics).
    Push = 2,
    /// `list.get(i)`: returns an optional.
    Get = 3,
    ToText = 4,
    ToFloat = 5,
    ToInt = 6,
    Sqrt = 7,
    Assert = 8,
    /// `panic(message)`
    Panic = 9,
}

impl Builtin {
    pub fn from_u8(b: u8) -> Option<Builtin> {
        Some(match b {
            0 => Builtin::Print,
            1 => Builtin::Len,
            2 => Builtin::Push,
            3 => Builtin::Get,
            4 => Builtin::ToText,
            5 => Builtin::ToFloat,
            6 => Builtin::ToInt,
            7 => Builtin::Sqrt,
            8 => Builtin::Assert,
            9 => Builtin::Panic,
            _ => return None,
        })
    }
}

/// One stack-machine instruction. Jump targets are absolute instruction indices.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Op {
    PushUnit,
    PushNone,
    PushBool(bool),
    PushInt(i64),
    PushFloat(f64),
    /// Push string constant `strings[i]`.
    PushText(u32),
    Pop,
    Dup,
    LoadLocal(u32),
    StoreLocal(u32),

    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Neg,
    Not,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,

    Jump(u32),
    JumpIfFalse(u32),
    JumpIfTrue(u32),

    /// `Call(function, argc)`
    Call(u32, u32),
    Ret,
    /// `Builtin(which, argc)`
    Builtin(Builtin, u32),

    MakeList(u32),
    Index,
    /// Stack: list, index, value -> new list.
    SetIndex,
    MakeStruct(u32),
    GetField(u32),
    /// Stack: struct, value -> new struct.
    SetField(u32),
    /// `MakeEnum(tag, field_count)`
    MakeEnum(u32, u32),
    /// Pop an enum value; jump when its tag differs from `tag`.
    JumpIfNotTag(u32, u32),
    /// Pop an enum value and push its payload field `i`.
    EnumField(u32),
    /// Pop the top of the stack and push `true` if it was `none`.
    IsNone,
    /// Pop `n` values, convert each to text and push the concatenation.
    Concat(u32),
    /// Abort with the message `strings[i]`.
    Panic(u32),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Function {
    pub name: String,
    pub arity: u32,
    /// Total local slots, including parameters.
    pub locals: u32,
    pub code: Vec<Op>,
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct Program {
    /// Exact Jax toolchain version that produced this program.
    pub jax_version: String,
    pub strings: Vec<String>,
    pub functions: Vec<Function>,
    pub entry: Option<u32>,
    /// `test "name" { ... }` blocks: (name, function index).
    pub tests: Vec<(String, u32)>,
}

// ----- binary encoding ---------------------------------------------------------

struct W(Vec<u8>);

impl W {
    fn u8(&mut self, b: u8) {
        self.0.push(b);
    }
    fn uv(&mut self, mut v: u64) {
        loop {
            let b = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                self.0.push(b);
                return;
            }
            self.0.push(b | 0x80);
        }
    }
    fn iv(&mut self, v: i64) {
        self.uv(((v << 1) ^ (v >> 63)) as u64);
    }
    fn f64(&mut self, v: f64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn str(&mut self, s: &str) {
        self.uv(s.len() as u64);
        self.0.extend_from_slice(s.as_bytes());
    }
}

struct Rd<'a> {
    b: &'a [u8],
    i: usize,
}

type DR<T> = Result<T, String>;

impl<'a> Rd<'a> {
    fn u8(&mut self) -> DR<u8> {
        let v = *self.b.get(self.i).ok_or("unexpected end of bytecode")?;
        self.i += 1;
        Ok(v)
    }
    fn uv(&mut self) -> DR<u64> {
        let mut shift = 0;
        let mut out = 0u64;
        loop {
            let b = self.u8()?;
            out |= ((b & 0x7f) as u64) << shift;
            if b & 0x80 == 0 {
                return Ok(out);
            }
            shift += 7;
            if shift > 63 {
                return Err("malformed varint".into());
            }
        }
    }
    fn u32(&mut self) -> DR<u32> {
        u32::try_from(self.uv()?).map_err(|_| "value out of range".to_string())
    }
    fn iv(&mut self) -> DR<i64> {
        let v = self.uv()?;
        Ok(((v >> 1) as i64) ^ -((v & 1) as i64))
    }
    fn f64(&mut self) -> DR<f64> {
        let s = self.b.get(self.i..self.i + 8).ok_or("unexpected end of bytecode")?;
        self.i += 8;
        Ok(f64::from_le_bytes(s.try_into().unwrap()))
    }
    fn str(&mut self) -> DR<String> {
        let n = self.uv()? as usize;
        let s = self.b.get(self.i..self.i + n).ok_or("unexpected end of bytecode")?;
        self.i += n;
        String::from_utf8(s.to_vec()).map_err(|_| "invalid UTF-8 in bytecode".to_string())
    }
}

fn encode_op(w: &mut W, op: &Op) {
    match *op {
        Op::PushUnit => w.u8(0),
        Op::PushNone => w.u8(1),
        Op::PushBool(b) => {
            w.u8(2);
            w.u8(b as u8)
        }
        Op::PushInt(n) => {
            w.u8(3);
            w.iv(n)
        }
        Op::PushFloat(f) => {
            w.u8(4);
            w.f64(f)
        }
        Op::PushText(i) => {
            w.u8(5);
            w.uv(i as u64)
        }
        Op::Pop => w.u8(6),
        Op::Dup => w.u8(7),
        Op::LoadLocal(i) => {
            w.u8(8);
            w.uv(i as u64)
        }
        Op::StoreLocal(i) => {
            w.u8(9);
            w.uv(i as u64)
        }
        Op::Add => w.u8(10),
        Op::Sub => w.u8(11),
        Op::Mul => w.u8(12),
        Op::Div => w.u8(13),
        Op::Rem => w.u8(14),
        Op::Neg => w.u8(15),
        Op::Not => w.u8(16),
        Op::Eq => w.u8(17),
        Op::Ne => w.u8(18),
        Op::Lt => w.u8(19),
        Op::Le => w.u8(20),
        Op::Gt => w.u8(21),
        Op::Ge => w.u8(22),
        Op::Jump(t) => {
            w.u8(23);
            w.uv(t as u64)
        }
        Op::JumpIfFalse(t) => {
            w.u8(24);
            w.uv(t as u64)
        }
        Op::JumpIfTrue(t) => {
            w.u8(25);
            w.uv(t as u64)
        }
        Op::Call(f, n) => {
            w.u8(26);
            w.uv(f as u64);
            w.uv(n as u64)
        }
        Op::Ret => w.u8(27),
        Op::Builtin(b, n) => {
            w.u8(28);
            w.u8(b as u8);
            w.uv(n as u64)
        }
        Op::MakeList(n) => {
            w.u8(29);
            w.uv(n as u64)
        }
        Op::Index => w.u8(30),
        Op::SetIndex => w.u8(31),
        Op::MakeStruct(n) => {
            w.u8(32);
            w.uv(n as u64)
        }
        Op::GetField(i) => {
            w.u8(33);
            w.uv(i as u64)
        }
        Op::SetField(i) => {
            w.u8(34);
            w.uv(i as u64)
        }
        Op::MakeEnum(t, n) => {
            w.u8(35);
            w.uv(t as u64);
            w.uv(n as u64)
        }
        Op::JumpIfNotTag(t, target) => {
            w.u8(36);
            w.uv(t as u64);
            w.uv(target as u64)
        }
        Op::EnumField(i) => {
            w.u8(37);
            w.uv(i as u64)
        }
        Op::IsNone => w.u8(38),
        Op::Concat(n) => {
            w.u8(39);
            w.uv(n as u64)
        }
        Op::Panic(i) => {
            w.u8(40);
            w.uv(i as u64)
        }
    }
}

fn decode_op(r: &mut Rd) -> DR<Op> {
    Ok(match r.u8()? {
        0 => Op::PushUnit,
        1 => Op::PushNone,
        2 => Op::PushBool(r.u8()? != 0),
        3 => Op::PushInt(r.iv()?),
        4 => Op::PushFloat(r.f64()?),
        5 => Op::PushText(r.u32()?),
        6 => Op::Pop,
        7 => Op::Dup,
        8 => Op::LoadLocal(r.u32()?),
        9 => Op::StoreLocal(r.u32()?),
        10 => Op::Add,
        11 => Op::Sub,
        12 => Op::Mul,
        13 => Op::Div,
        14 => Op::Rem,
        15 => Op::Neg,
        16 => Op::Not,
        17 => Op::Eq,
        18 => Op::Ne,
        19 => Op::Lt,
        20 => Op::Le,
        21 => Op::Gt,
        22 => Op::Ge,
        23 => Op::Jump(r.u32()?),
        24 => Op::JumpIfFalse(r.u32()?),
        25 => Op::JumpIfTrue(r.u32()?),
        26 => Op::Call(r.u32()?, r.u32()?),
        27 => Op::Ret,
        28 => {
            let b = Builtin::from_u8(r.u8()?).ok_or("unknown builtin in bytecode")?;
            Op::Builtin(b, r.u32()?)
        }
        29 => Op::MakeList(r.u32()?),
        30 => Op::Index,
        31 => Op::SetIndex,
        32 => Op::MakeStruct(r.u32()?),
        33 => Op::GetField(r.u32()?),
        34 => Op::SetField(r.u32()?),
        35 => Op::MakeEnum(r.u32()?, r.u32()?),
        36 => Op::JumpIfNotTag(r.u32()?, r.u32()?),
        37 => Op::EnumField(r.u32()?),
        38 => Op::IsNone,
        39 => Op::Concat(r.u32()?),
        40 => Op::Panic(r.u32()?),
        other => return Err(format!("unknown opcode {other}")),
    })
}

impl Program {
    /// Serialize to the `.jxb` byte format.
    pub fn encode(&self) -> Vec<u8> {
        let mut w = W(Vec::new());
        w.0.extend_from_slice(BUNDLE_MAGIC);
        w.0.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        w.str(&self.jax_version);
        w.uv(self.strings.len() as u64);
        for s in &self.strings {
            w.str(s);
        }
        w.uv(self.functions.len() as u64);
        for f in &self.functions {
            w.str(&f.name);
            w.uv(f.arity as u64);
            w.uv(f.locals as u64);
            w.uv(f.code.len() as u64);
            for op in &f.code {
                encode_op(&mut w, op);
            }
        }
        match self.entry {
            Some(e) => {
                w.u8(1);
                w.uv(e as u64);
            }
            None => w.u8(0),
        }
        w.uv(self.tests.len() as u64);
        for (name, idx) in &self.tests {
            w.str(name);
            w.uv(*idx as u64);
        }
        w.0
    }

    pub fn decode(bytes: &[u8]) -> Result<Program, String> {
        if bytes.len() < 6 || &bytes[..4] != BUNDLE_MAGIC {
            return Err("not a Jax bundle (bad magic)".into());
        }
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        if version != FORMAT_VERSION {
            return Err(format!(
                "bytecode format version {version} is not supported by this runtime (expected {FORMAT_VERSION}); run it with the Jax version it was built with"
            ));
        }
        let mut r = Rd { b: bytes, i: 6 };
        let jax_version = r.str()?;
        let mut strings = Vec::new();
        for _ in 0..r.uv()? {
            strings.push(r.str()?);
        }
        let mut functions = Vec::new();
        for _ in 0..r.uv()? {
            let name = r.str()?;
            let arity = r.u32()?;
            let locals = r.u32()?;
            let n = r.uv()?;
            let mut code = Vec::new();
            for _ in 0..n {
                code.push(decode_op(&mut r)?);
            }
            functions.push(Function {
                name,
                arity,
                locals,
                code,
            });
        }
        let entry = if r.u8()? == 1 { Some(r.u32()?) } else { None };
        let mut tests = Vec::new();
        for _ in 0..r.uv()? {
            let name = r.str()?;
            tests.push((name, r.u32()?));
        }
        let p = Program {
            jax_version,
            strings,
            functions,
            entry,
            tests,
        };
        p.validate()?;
        Ok(p)
    }

    /// Cheap structural checks so a corrupt bundle fails early instead of crashing the VM.
    pub fn validate(&self) -> Result<(), String> {
        for f in &self.functions {
            let n = f.code.len() as u32;
            for op in &f.code {
                match *op {
                    Op::Jump(t) | Op::JumpIfFalse(t) | Op::JumpIfTrue(t) | Op::JumpIfNotTag(_, t) if t > n => {
                        return Err(format!("function `{}`: jump target out of range", f.name));
                    }
                    Op::Call(i, _) if i as usize >= self.functions.len() => {
                        return Err(format!("function `{}`: call to unknown function", f.name));
                    }
                    Op::PushText(i) | Op::Panic(i) if i as usize >= self.strings.len() => {
                        return Err(format!("function `{}`: unknown string constant", f.name));
                    }
                    Op::LoadLocal(i) | Op::StoreLocal(i) if i >= f.locals => {
                        return Err(format!("function `{}`: local slot out of range", f.name));
                    }
                    _ => {}
                }
            }
        }
        if let Some(e) = self.entry {
            if e as usize >= self.functions.len() {
                return Err("entry point out of range".into());
            }
        }
        Ok(())
    }

    /// Append this program to a runtime executable image so the result is a
    /// single self-contained file.
    pub fn embed_in(&self, runtime: &[u8]) -> Vec<u8> {
        let payload = self.encode();
        let mut out = Vec::with_capacity(runtime.len() + payload.len() + 16);
        out.extend_from_slice(runtime);
        out.extend_from_slice(&payload);
        out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        out.extend_from_slice(EXE_MAGIC);
        out
    }

    /// Extract an embedded program from an executable image, if there is one.
    pub fn extract_from(image: &[u8]) -> Option<Result<Program, String>> {
        if image.len() < 16 || &image[image.len() - 8..] != EXE_MAGIC {
            return None;
        }
        let len_bytes: [u8; 8] = image[image.len() - 16..image.len() - 8].try_into().ok()?;
        let len = u64::from_le_bytes(len_bytes) as usize;
        let end = image.len() - 16;
        if len > end {
            return Some(Err("corrupt embedded program".into()));
        }
        Some(Program::decode(&image[end - len..end]))
    }

    /// Human-readable listing of the whole program.
    pub fn disassemble(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(
            out,
            "; jax {}  ({} functions, {} strings)",
            self.jax_version,
            self.functions.len(),
            self.strings.len()
        );
        for (i, f) in self.functions.iter().enumerate() {
            let mark = if self.entry == Some(i as u32) { "  ; entry" } else { "" };
            let _ = writeln!(out, "\nfn #{i} {}  (arity {}, locals {}){mark}", f.name, f.arity, f.locals);
            for (pc, op) in f.code.iter().enumerate() {
                let extra = match op {
                    Op::PushText(s) | Op::Panic(s) => format!("   ; {:?}", self.strings[*s as usize]),
                    Op::Call(t, _) => format!("   ; {}", self.functions[*t as usize].name),
                    _ => String::new(),
                };
                let _ = writeln!(out, "  {pc:4}  {op:?}{extra}");
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Program {
        Program {
            jax_version: "0.1.0".into(),
            strings: vec!["hello".into()],
            functions: vec![Function {
                name: "main".into(),
                arity: 0,
                locals: 1,
                code: vec![
                    Op::PushText(0),
                    Op::Builtin(Builtin::Print, 1),
                    Op::Pop,
                    Op::PushInt(-42),
                    Op::PushFloat(1.5),
                    Op::MakeEnum(2, 0),
                    Op::JumpIfNotTag(2, 8),
                    Op::Ret,
                ],
            }],
            entry: Some(0),
            tests: vec![("t".into(), 0)],
        }
    }

    #[test]
    fn roundtrip() {
        let p = sample();
        let q = Program::decode(&p.encode()).unwrap();
        assert_eq!(p, q);
    }

    #[test]
    fn embed_and_extract() {
        let p = sample();
        let image = p.embed_in(b"fake runtime bytes");
        let q = Program::extract_from(&image).unwrap().unwrap();
        assert_eq!(p, q);
        assert!(Program::extract_from(b"plain exe").is_none());
    }

    #[test]
    fn rejects_garbage() {
        assert!(Program::decode(b"nope").is_err());
        let mut bytes = sample().encode();
        bytes[4] = 99;
        assert!(Program::decode(&bytes).unwrap_err().contains("not supported"));
    }
}
