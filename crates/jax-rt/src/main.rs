//! `jax-rt`: the Jax runtime stub.
//!
//! `jax build` copies this executable and appends the compiled program to it,
//! producing one self-contained file that needs no Jax installation. When no
//! program is embedded, `jax-rt <file.jxb>` runs a portable bundle instead.

use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom};
use std::process::exit;

use jax_bytecode::{Program, EXE_MAGIC};
use jax_vm::Vm;

/// Read only the tail of our own executable to find an embedded program.
fn read_embedded(path: &std::path::Path) -> Option<Result<Program, String>> {
    let mut f = File::open(path).ok()?;
    let len = f.metadata().ok()?.len();
    if len < 16 {
        return None;
    }
    f.seek(SeekFrom::Start(len - 16)).ok()?;
    let mut trailer = [0u8; 16];
    f.read_exact(&mut trailer).ok()?;
    if &trailer[8..] != EXE_MAGIC {
        return None;
    }
    let payload_len = u64::from_le_bytes(trailer[..8].try_into().ok()?);
    if payload_len > len - 16 {
        return Some(Err("corrupt embedded program".into()));
    }
    f.seek(SeekFrom::Start(len - 16 - payload_len)).ok()?;
    let mut payload = vec![0u8; payload_len as usize];
    if let Err(e) = f.read_exact(&mut payload) {
        return Some(Err(e.to_string()));
    }
    Some(Program::decode(&payload))
}

fn run_program(program: &Program) -> ! {
    let stdout = std::io::stdout();
    let out = BufWriter::with_capacity(64 * 1024, stdout.lock());
    let mut vm = Vm::new(program, Box::new(out));
    match vm.run_entry() {
        Ok(()) => exit(0),
        Err(e) => {
            eprintln!("{e}");
            exit(e.exit_code());
        }
    }
}

fn main() {
    if let Ok(exe) = std::env::current_exe() {
        if let Some(embedded) = read_embedded(&exe) {
            match embedded {
                Ok(p) => run_program(&p),
                Err(e) => {
                    eprintln!("error: {e}");
                    exit(1);
                }
            }
        }
    }
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(|s| s.as_str()) {
        Some("--version") | Some("-V") => {
            println!("jax-rt {}", env!("CARGO_PKG_VERSION"));
        }
        Some(path) if !path.starts_with('-') => {
            let bytes = match std::fs::read(path) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("error: cannot read {path}: {e}");
                    exit(1);
                }
            };
            match Program::decode(&bytes) {
                Ok(p) => run_program(&p),
                Err(e) => {
                    eprintln!("error: {path}: {e}");
                    exit(1);
                }
            }
        }
        _ => {
            eprintln!("jax-rt {}: runs Jax bundles.\nusage: jax-rt <program.jxb>", env!("CARGO_PKG_VERSION"));
            exit(2);
        }
    }
}
