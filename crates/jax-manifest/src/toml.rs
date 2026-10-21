//! A small, dependency-free TOML parser.
//!
//! Supported: comments, `[table.headers]`, bare/quoted/dotted keys, basic and
//! literal strings, integers, floats, booleans, (multi-line) arrays and
//! single-line inline tables. Not supported (reported as errors): arrays of
//! tables (`[[x]]`), multi-line strings and datetimes. This is everything
//! `jax.toml`, `jax.lock` and `config.toml` need.

use std::collections::BTreeMap;
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Str(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    Array(Vec<Value>),
    Table(Table),
}

pub type Table = BTreeMap<String, Value>;

impl Value {
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Str(_) => "a string",
            Value::Int(_) => "an integer",
            Value::Float(_) => "a float",
            Value::Bool(_) => "a boolean",
            Value::Array(_) => "an array",
            Value::Table(_) => "a table",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TomlError {
    pub line: usize,
    pub message: String,
}

impl fmt::Display for TomlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}: {}", self.line, self.message)
    }
}

impl std::error::Error for TomlError {}

pub fn parse(src: &str) -> Result<Table, TomlError> {
    let mut p = Parser {
        chars: src.chars().collect(),
        i: 0,
        line: 1,
    };
    p.document()
}

struct Parser {
    chars: Vec<char>,
    i: usize,
    line: usize,
}

type R<T> = Result<T, TomlError>;

impl Parser {
    fn err<T>(&self, msg: impl Into<String>) -> R<T> {
        Err(TomlError {
            line: self.line,
            message: msg.into(),
        })
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.i).copied()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.peek();
        if c == Some('\n') {
            self.line += 1;
        }
        if c.is_some() {
            self.i += 1;
        }
        c
    }

    fn skip_inline_ws(&mut self) {
        while matches!(self.peek(), Some(' ') | Some('\t')) {
            self.i += 1;
        }
    }

    fn skip_comment(&mut self) {
        if self.peek() == Some('#') {
            while !matches!(self.peek(), Some('\n') | None) {
                self.i += 1;
            }
        }
    }

    /// Skip whitespace, newlines and comments.
    fn skip_all(&mut self) {
        loop {
            match self.peek() {
                Some(' ') | Some('\t') | Some('\r') | Some('\n') => {
                    self.bump();
                }
                Some('#') => self.skip_comment(),
                _ => break,
            }
        }
    }

    fn end_of_line(&mut self) -> R<()> {
        self.skip_inline_ws();
        self.skip_comment();
        match self.peek() {
            None => Ok(()),
            Some('\r') => {
                self.i += 1;
                self.end_of_line()
            }
            Some('\n') => {
                self.bump();
                Ok(())
            }
            Some(c) => self.err(format!("unexpected `{c}` (expected end of line)")),
        }
    }

    fn document(&mut self) -> R<Table> {
        let mut root = Table::new();
        let mut current: Vec<String> = Vec::new();
        let mut defined_tables: Vec<Vec<String>> = Vec::new();
        loop {
            self.skip_all();
            match self.peek() {
                None => break,
                Some('[') => {
                    self.i += 1;
                    if self.peek() == Some('[') {
                        return self.err("arrays of tables (`[[...]]`) are not supported");
                    }
                    let path = self.key_path(']')?;
                    if self.bump() != Some(']') {
                        return self.err("expected `]` to close the table header");
                    }
                    self.end_of_line()?;
                    if defined_tables.contains(&path) {
                        return self.err(format!("table `[{}]` is defined more than once", path.join(".")));
                    }
                    defined_tables.push(path.clone());
                    ensure_table(&mut root, &path).map_err(|m| TomlError {
                        line: self.line,
                        message: m,
                    })?;
                    current = path;
                }
                Some(_) => {
                    let key_path = self.key_path('=')?;
                    self.skip_inline_ws();
                    if self.bump() != Some('=') {
                        return self.err("expected `=` after key");
                    }
                    self.skip_inline_ws();
                    let value = self.value()?;
                    self.end_of_line()?;
                    let mut full = current.clone();
                    full.extend(key_path);
                    let (last, parents) = full.split_last().unwrap();
                    let table = ensure_table(&mut root, parents).map_err(|m| TomlError {
                        line: self.line,
                        message: m,
                    })?;
                    if table.contains_key(last) {
                        return self.err(format!("duplicate key `{last}`"));
                    }
                    table.insert(last.clone(), value);
                }
            }
        }
        Ok(root)
    }

    /// A dotted key path; stops before `stop` (without consuming it).
    fn key_path(&mut self, stop: char) -> R<Vec<String>> {
        let mut path = Vec::new();
        loop {
            self.skip_inline_ws();
            path.push(self.key()?);
            self.skip_inline_ws();
            if self.peek() == Some('.') {
                self.i += 1;
                continue;
            }
            if self.peek() == Some(stop) {
                return Ok(path);
            }
            return self.err(format!("expected `.` or `{stop}` after key `{}`", path.last().unwrap()));
        }
    }

    fn key(&mut self) -> R<String> {
        match self.peek() {
            Some('"') => self.basic_string(),
            Some('\'') => self.literal_string(),
            Some(c) if c.is_ascii_alphanumeric() || c == '_' || c == '-' => {
                let mut s = String::new();
                while let Some(c) = self.peek() {
                    if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                        s.push(c);
                        self.i += 1;
                    } else {
                        break;
                    }
                }
                Ok(s)
            }
            Some(c) => self.err(format!("unexpected `{c}` in key")),
            None => self.err("unexpected end of file in key"),
        }
    }

    fn basic_string(&mut self) -> R<String> {
        self.i += 1; // opening quote
        if self.peek() == Some('"') && self.chars.get(self.i + 1) == Some(&'"') {
            return self.err("multi-line strings are not supported");
        }
        let mut s = String::new();
        loop {
            match self.bump() {
                None | Some('\n') => return self.err("unterminated string"),
                Some('"') => return Ok(s),
                Some('\\') => match self.bump() {
                    Some('n') => s.push('\n'),
                    Some('t') => s.push('\t'),
                    Some('r') => s.push('\r'),
                    Some('"') => s.push('"'),
                    Some('\\') => s.push('\\'),
                    Some(c) => return self.err(format!("unknown escape `\\{c}`")),
                    None => return self.err("unterminated string"),
                },
                Some(c) => s.push(c),
            }
        }
    }

    fn literal_string(&mut self) -> R<String> {
        self.i += 1;
        let mut s = String::new();
        loop {
            match self.bump() {
                None | Some('\n') => return self.err("unterminated string"),
                Some('\'') => return Ok(s),
                Some(c) => s.push(c),
            }
        }
    }

    fn value(&mut self) -> R<Value> {
        match self.peek() {
            Some('"') => Ok(Value::Str(self.basic_string()?)),
            Some('\'') => Ok(Value::Str(self.literal_string()?)),
            Some('[') => self.array(),
            Some('{') => self.inline_table(),
            Some(c) if c == '-' || c == '+' || c.is_ascii_digit() => self.number(),
            Some(c) if c.is_ascii_alphabetic() => {
                let mut word = String::new();
                while let Some(c) = self.peek() {
                    if c.is_ascii_alphanumeric() {
                        word.push(c);
                        self.i += 1;
                    } else {
                        break;
                    }
                }
                match word.as_str() {
                    "true" => Ok(Value::Bool(true)),
                    "false" => Ok(Value::Bool(false)),
                    _ => self.err(format!("unexpected `{word}` (strings must be quoted)")),
                }
            }
            Some(c) => self.err(format!("unexpected `{c}` where a value was expected")),
            None => self.err("unexpected end of file where a value was expected"),
        }
    }

    fn number(&mut self) -> R<Value> {
        let mut s = String::new();
        while let Some(c) = self.peek() {
            if c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.' | '_') {
                s.push(c);
                self.i += 1;
            } else {
                break;
            }
        }
        let clean: String = s.chars().filter(|c| *c != '_').collect();
        if let Ok(n) = clean.parse::<i64>() {
            return Ok(Value::Int(n));
        }
        if let Ok(f) = clean.parse::<f64>() {
            return Ok(Value::Float(f));
        }
        self.err(format!("invalid number `{s}`"))
    }

    fn array(&mut self) -> R<Value> {
        self.i += 1;
        let mut items = Vec::new();
        loop {
            self.skip_all();
            match self.peek() {
                Some(']') => {
                    self.i += 1;
                    return Ok(Value::Array(items));
                }
                None => return self.err("unterminated array"),
                _ => {}
            }
            items.push(self.value()?);
            self.skip_all();
            match self.peek() {
                Some(',') => {
                    self.i += 1;
                }
                Some(']') => {}
                _ => return self.err("expected `,` or `]` in array"),
            }
        }
    }

    fn inline_table(&mut self) -> R<Value> {
        self.i += 1;
        let mut table = Table::new();
        loop {
            self.skip_inline_ws();
            if self.peek() == Some('}') {
                self.i += 1;
                return Ok(Value::Table(table));
            }
            let path = self.key_path('=')?;
            self.skip_inline_ws();
            if self.bump() != Some('=') {
                return self.err("expected `=` in inline table");
            }
            self.skip_inline_ws();
            let v = self.value()?;
            let (last, parents) = path.split_last().unwrap();
            let t = ensure_table(&mut table, parents).map_err(|m| TomlError {
                line: self.line,
                message: m,
            })?;
            if t.contains_key(last) {
                return self.err(format!("duplicate key `{last}`"));
            }
            t.insert(last.clone(), v);
            self.skip_inline_ws();
            match self.peek() {
                Some(',') => {
                    self.i += 1;
                }
                Some('}') => {}
                _ => return self.err("expected `,` or `}` in inline table"),
            }
        }
    }
}

fn ensure_table<'a>(root: &'a mut Table, path: &[String]) -> Result<&'a mut Table, String> {
    let mut cur = root;
    for seg in path {
        let entry = cur.entry(seg.clone()).or_insert_with(|| Value::Table(Table::new()));
        match entry {
            Value::Table(t) => cur = t,
            other => {
                return Err(format!("`{seg}` is {} and cannot also be a table", other.type_name()));
            }
        }
    }
    Ok(cur)
}

/// Escape a string for emission as a TOML basic string (with quotes).
pub fn quote(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Format a key, quoting it only if necessary.
pub fn key(k: &str) -> String {
    if !k.is_empty() && k.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') {
        k.to_string()
    } else {
        quote(k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_manifest() {
        let t = parse(
            r#"
# comment
[project]
name = "my-app"   # trailing
version = "0.3.0"
count = 1_000

[dependencies]
"acme.chart" = "2.1.0"
"web.router" = { version = "1.3.0", features = ["tls", "gzip"] }

[build]
targets = [
    "linux-x64",
    "windows-x64", # comment in array
]
"#,
        )
        .unwrap();
        let project = match &t["project"] {
            Value::Table(t) => t,
            _ => panic!(),
        };
        assert_eq!(project["name"], Value::Str("my-app".into()));
        assert_eq!(project["count"], Value::Int(1000));
        let deps = match &t["dependencies"] {
            Value::Table(t) => t,
            _ => panic!(),
        };
        assert!(matches!(deps["web.router"], Value::Table(_)));
        match &t["build"] {
            Value::Table(b) => assert_eq!(b["targets"], Value::Array(vec![Value::Str("linux-x64".into()), Value::Str("windows-x64".into())])),
            _ => panic!(),
        }
    }

    #[test]
    fn dotted_keys_and_headers() {
        let t = parse("[profile.release]\noptimize = \"speed\"\na.b = true\n").unwrap();
        match &t["profile"] {
            Value::Table(p) => match &p["release"] {
                Value::Table(r) => {
                    assert_eq!(r["optimize"], Value::Str("speed".into()));
                    assert!(matches!(r["a"], Value::Table(_)));
                }
                _ => panic!(),
            },
            _ => panic!(),
        }
    }

    #[test]
    fn errors_have_lines() {
        let e = parse("a = 1\nb = \n").unwrap_err();
        assert_eq!(e.line, 2);
        let e = parse("a = 1\na = 2\n").unwrap_err();
        assert!(e.message.contains("duplicate"));
        let e = parse("[[x]]\n").unwrap_err();
        assert!(e.message.contains("not supported"));
    }

    #[test]
    fn quoting_roundtrip() {
        let s = "a \"b\" \\ c\n";
        let t = parse(&format!("k = {}", quote(s))).unwrap();
        assert_eq!(t["k"], Value::Str(s.into()));
    }
}
