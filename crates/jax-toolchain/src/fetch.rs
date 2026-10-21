//! Fetching bytes from `file://` paths, plain paths and `http(s)://` URLs.
//!
//! HTTP(S) uses the system `curl` (present on Windows 10+, macOS and every
//! mainstream Linux). Stage 0 deliberately avoids linking a TLS stack.

use std::path::PathBuf;
use std::process::Command;

fn file_path(url: &str) -> Option<PathBuf> {
    if let Some(rest) = url.strip_prefix("file://") {
        // `file:///C:/x` -> `C:/x` on Windows-style paths.
        let bytes = rest.as_bytes();
        if bytes.len() >= 3 && bytes[0] == b'/' && bytes[2] == b':' && bytes[1].is_ascii_alphabetic() {
            return Some(PathBuf::from(&rest[1..]));
        }
        return Some(PathBuf::from(rest));
    }
    if !url.contains("://") {
        return Some(PathBuf::from(url));
    }
    None
}

pub fn fetch_bytes(url: &str) -> Result<Vec<u8>, String> {
    if let Some(path) = file_path(url) {
        return std::fs::read(&path).map_err(|e| format!("cannot read {}: {e}", path.display()));
    }
    if !(url.starts_with("http://") || url.starts_with("https://")) {
        return Err(format!("unsupported URL scheme: {url}"));
    }
    let out = Command::new("curl")
        .args(["--fail", "--location", "--silent", "--show-error", "--max-time", "900", url])
        .output()
        .map_err(|e| {
            format!("cannot run `curl` to download {url}: {e}\nInstall curl, or point JAX_DIST_URL at a local directory (file:///path).")
        })?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr).trim().to_string();
        return Err(format!("download failed for {url}: {err}"));
    }
    Ok(out.stdout)
}

pub fn fetch_text(url: &str) -> Result<String, String> {
    let bytes = fetch_bytes(url)?;
    String::from_utf8(bytes).map_err(|_| format!("{url} is not valid UTF-8"))
}
