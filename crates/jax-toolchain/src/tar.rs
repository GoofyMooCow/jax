//! A minimal ustar reader/writer, enough for toolchain archives.
//!
//! The writer is deterministic (zero timestamps and owners), so identical
//! inputs always produce byte-identical archives. The reader refuses absolute
//! paths, `..` components and anything that is not a regular file or directory.

use std::fs;
use std::io;
use std::path::{Component, Path, PathBuf};

pub struct Entry {
    /// Relative path using `/` separators.
    pub path: String,
    pub data: Vec<u8>,
    pub mode: u32,
}

fn put_octal(dst: &mut [u8], value: u64) {
    let width = dst.len() - 1;
    let s = format!("{value:0width$o}");
    dst[..width].copy_from_slice(s.as_bytes());
    dst[width] = 0;
}

pub fn write_tar(entries: &[Entry]) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    for e in entries {
        if e.path.len() > 100 {
            return Err(format!("path too long for archive: {}", e.path));
        }
        let mut h = [0u8; 512];
        h[..e.path.len()].copy_from_slice(e.path.as_bytes());
        put_octal(&mut h[100..108], e.mode as u64);
        put_octal(&mut h[108..116], 0);
        put_octal(&mut h[116..124], 0);
        put_octal(&mut h[124..136], e.data.len() as u64);
        put_octal(&mut h[136..148], 0);
        h[156] = b'0';
        h[257..263].copy_from_slice(b"ustar\0");
        h[263..265].copy_from_slice(b"00");
        h[148..156].copy_from_slice(b"        ");
        let sum: u32 = h.iter().map(|b| *b as u32).sum();
        let chk = format!("{sum:06o}");
        h[148..154].copy_from_slice(chk.as_bytes());
        h[154] = 0;
        h[155] = b' ';
        out.extend_from_slice(&h);
        out.extend_from_slice(&e.data);
        let pad = (512 - e.data.len() % 512) % 512;
        out.extend(std::iter::repeat(0u8).take(pad));
    }
    out.extend(std::iter::repeat(0u8).take(1024));
    Ok(out)
}

fn parse_octal(field: &[u8]) -> Result<u64, String> {
    let s: String = field
        .iter()
        .take_while(|b| **b != 0)
        .map(|b| *b as char)
        .collect::<String>()
        .trim()
        .to_string();
    if s.is_empty() {
        return Ok(0);
    }
    u64::from_str_radix(&s, 8).map_err(|_| "corrupt tar header".to_string())
}

fn safe_relative(name: &str) -> Result<PathBuf, String> {
    let p = Path::new(name);
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            Component::Normal(s) => out.push(s),
            Component::CurDir => {}
            _ => return Err(format!("unsafe path in archive: {name}")),
        }
    }
    if out.as_os_str().is_empty() {
        return Err("empty path in archive".into());
    }
    Ok(out)
}

/// Unpack `bytes` into `dest` (which is created if needed).
pub fn unpack(bytes: &[u8], dest: &Path) -> Result<(), String> {
    let io_err = |e: io::Error| e.to_string();
    fs::create_dir_all(dest).map_err(io_err)?;
    let mut pos = 0;
    while pos + 512 <= bytes.len() {
        let h = &bytes[pos..pos + 512];
        if h.iter().all(|b| *b == 0) {
            return Ok(());
        }
        let name_end = h[..100].iter().position(|b| *b == 0).unwrap_or(100);
        let name = String::from_utf8_lossy(&h[..name_end]).to_string();
        let mode = parse_octal(&h[100..108])? as u32;
        let size = parse_octal(&h[124..136])? as usize;
        let kind = h[156];
        // Verify the checksum.
        let stored = parse_octal(&h[148..156])? as u32;
        let mut hdr = [0u8; 512];
        hdr.copy_from_slice(h);
        hdr[148..156].copy_from_slice(b"        ");
        let sum: u32 = hdr.iter().map(|b| *b as u32).sum();
        if sum != stored {
            return Err("corrupt tar header (bad checksum)".into());
        }
        pos += 512;
        let data_end = pos.checked_add(size).filter(|e| *e <= bytes.len()).ok_or("truncated tar archive")?;
        match kind {
            b'0' | 0 => {
                let rel = safe_relative(&name)?;
                let target = dest.join(&rel);
                if let Some(parent) = target.parent() {
                    fs::create_dir_all(parent).map_err(io_err)?;
                }
                fs::write(&target, &bytes[pos..data_end]).map_err(io_err)?;
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    let _ = fs::set_permissions(&target, fs::Permissions::from_mode(mode & 0o777));
                }
                #[cfg(not(unix))]
                let _ = mode;
            }
            b'5' => {
                let rel = safe_relative(name.trim_end_matches('/'))?;
                fs::create_dir_all(dest.join(rel)).map_err(io_err)?;
            }
            b'x' | b'g' => {} // pax headers: ignore
            other => return Err(format!("unsupported tar entry type `{}` for {name}", other as char)),
        }
        pos = data_end + (512 - size % 512) % 512;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let dir = std::env::temp_dir().join(format!("jax-tar-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let tar = write_tar(&[
            Entry {
                path: "bin/tool".into(),
                data: b"#!/bin/sh\necho hi\n".to_vec(),
                mode: 0o755,
            },
            Entry {
                path: "readme.txt".into(),
                data: vec![7; 1000],
                mode: 0o644,
            },
        ])
        .unwrap();
        assert_eq!(tar, write_tar(&[
            Entry { path: "bin/tool".into(), data: b"#!/bin/sh\necho hi\n".to_vec(), mode: 0o755 },
            Entry { path: "readme.txt".into(), data: vec![7; 1000], mode: 0o644 },
        ]).unwrap(), "writer must be deterministic");
        unpack(&tar, &dir).unwrap();
        assert_eq!(fs::read(dir.join("bin/tool")).unwrap(), b"#!/bin/sh\necho hi\n");
        assert_eq!(fs::read(dir.join("readme.txt")).unwrap().len(), 1000);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(fs::metadata(dir.join("bin/tool")).unwrap().permissions().mode() & 0o111, 0o111);
        }
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_path_traversal() {
        let dir = std::env::temp_dir().join(format!("jax-tar-test2-{}", std::process::id()));
        let tar = write_tar(&[Entry {
            path: "../evil".into(),
            data: b"x".to_vec(),
            mode: 0o644,
        }])
        .unwrap();
        assert!(unpack(&tar, &dir).unwrap_err().contains("unsafe path"));
        let _ = fs::remove_dir_all(&dir);
    }
}
