//! Source spans, source files and human-friendly diagnostics.

/// A half-open byte range `[lo, hi)` into a source file.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Span {
    pub lo: u32,
    pub hi: u32,
}

impl Span {
    pub const DUMMY: Span = Span { lo: 0, hi: 0 };

    pub fn new(lo: u32, hi: u32) -> Span {
        Span { lo, hi }
    }

    /// The smallest span covering both `self` and `other`.
    pub fn to(self, other: Span) -> Span {
        Span {
            lo: self.lo.min(other.lo),
            hi: self.hi.max(other.hi),
        }
    }
}

/// A named source text with a precomputed line table.
#[derive(Debug, Clone)]
pub struct SourceFile {
    pub name: String,
    pub src: String,
    line_starts: Vec<u32>,
}

impl SourceFile {
    pub fn new(name: impl Into<String>, src: impl Into<String>) -> SourceFile {
        let src = src.into();
        let mut line_starts = vec![0u32];
        for (i, b) in src.bytes().enumerate() {
            if b == b'\n' {
                line_starts.push(i as u32 + 1);
            }
        }
        SourceFile {
            name: name.into(),
            src,
            line_starts,
        }
    }

    /// 1-based (line, column) for a byte offset. Columns count characters.
    pub fn line_col(&self, offset: u32) -> (usize, usize) {
        let offset = offset.min(self.src.len() as u32);
        let line = match self.line_starts.binary_search(&offset) {
            Ok(i) => i,
            Err(i) => i - 1,
        };
        let start = self.line_starts[line] as usize;
        let mut end = offset as usize;
        while end > start && !self.src.is_char_boundary(end) {
            end -= 1;
        }
        let col = self.src[start..end].chars().count();
        (line + 1, col + 1)
    }

    /// Text of a 1-based line, without its trailing newline.
    pub fn line_text(&self, line: usize) -> &str {
        let idx = line.saturating_sub(1).min(self.line_starts.len() - 1);
        let start = self.line_starts[idx] as usize;
        let end = self
            .line_starts
            .get(idx + 1)
            .map(|e| *e as usize)
            .unwrap_or(self.src.len());
        self.src[start..end].trim_end_matches(['\n', '\r'])
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
}

#[derive(Clone, Debug)]
pub struct Diagnostic {
    pub severity: Severity,
    pub code: Option<&'static str>,
    pub message: String,
    pub span: Span,
    pub help: Option<String>,
}

impl Diagnostic {
    pub fn error(span: Span, message: impl Into<String>) -> Diagnostic {
        Diagnostic {
            severity: Severity::Error,
            code: None,
            message: message.into(),
            span,
            help: None,
        }
    }

    pub fn warning(span: Span, message: impl Into<String>) -> Diagnostic {
        Diagnostic {
            severity: Severity::Warning,
            ..Diagnostic::error(span, message)
        }
    }

    pub fn with_help(mut self, help: impl Into<String>) -> Diagnostic {
        self.help = Some(help.into());
        self
    }

    pub fn with_code(mut self, code: &'static str) -> Diagnostic {
        self.code = Some(code);
        self
    }

    pub fn is_error(&self) -> bool {
        self.severity == Severity::Error
    }
}

/// Render a diagnostic with a source excerpt and a caret underline.
pub fn render(file: &SourceFile, d: &Diagnostic) -> String {
    let label = match d.severity {
        Severity::Error => "error",
        Severity::Warning => "warning",
    };
    let mut out = String::new();
    match d.code {
        Some(c) => out.push_str(&format!("{label}[{c}]: {}\n", d.message)),
        None => out.push_str(&format!("{label}: {}\n", d.message)),
    }
    let (line, col) = file.line_col(d.span.lo);
    let text = file.line_text(line);
    let gutter = line.to_string().len();
    out.push_str(&format!("{:>g$}--> {}:{}:{}\n", "", file.name, line, col, g = gutter + 1));
    out.push_str(&format!("{:>g$} |\n", "", g = gutter));
    out.push_str(&format!("{:>g$} | {}\n", line, text, g = gutter));
    let (end_line, end_col) = file.line_col(d.span.hi.max(d.span.lo));
    let width = if end_line == line {
        end_col.saturating_sub(col).max(1)
    } else {
        text.chars().count().saturating_sub(col - 1).max(1)
    };
    out.push_str(&format!(
        "{:>g$} | {}{}\n",
        "",
        " ".repeat(col - 1),
        "^".repeat(width),
        g = gutter
    ));
    if let Some(h) = &d.help {
        out.push_str(&format!("{:>g$} = help: {}\n", "", h, g = gutter));
    }
    out
}

/// Levenshtein-based "did you mean" helper.
pub fn suggest<'a>(name: &str, candidates: impl IntoIterator<Item = &'a str>) -> Option<String> {
    let mut best: Option<(usize, &str)> = None;
    for c in candidates {
        let d = edit_distance(name, c);
        let limit = if name.chars().count() <= 3 { 1 } else { 2 };
        if d <= limit && d > 0 && best.map(|(bd, _)| d < bd).unwrap_or(true) {
            best = Some((d, c));
        }
    }
    best.map(|(_, c)| c.to_string())
}

fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for i in 1..=a.len() {
        let mut cur = vec![i];
        for j in 1..=b.len() {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur.push((prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost));
        }
        prev = cur;
    }
    prev[b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_col_works() {
        let f = SourceFile::new("t.jax", "ab\ncd\n");
        assert_eq!(f.line_col(0), (1, 1));
        assert_eq!(f.line_col(4), (2, 2));
        assert_eq!(f.line_text(2), "cd");
    }

    #[test]
    fn suggestions() {
        assert_eq!(suggest("prnt", ["print", "panic"]), Some("print".into()));
        assert_eq!(suggest("zzzz", ["print"]), None);
    }
}
