//! End-to-end tests: compile each `tests/programs/*.jax`, run it on the VM and
//! compare stdout with the `// out: ...` comment lines in the file.
//! A file may instead declare `// error: <substring>` to assert that
//! compilation fails with a diagnostic containing that text, or
//! `// panic: <substring>` to assert a runtime panic.

use std::fs;
use std::path::PathBuf;

use jax_compiler::{compile, render, Options};
use jax_syntax::SourceFile;
use jax_vm::{RunError, Vm};

fn run_source(name: &str, src: &str) -> Result<String, Result<RunError, String>> {
    let file = SourceFile::new(name, src);
    let program = match compile(&file, &Options::default()) {
        Ok(p) => p,
        Err(diags) => {
            let text: Vec<String> = diags.iter().map(|d| render(&file, d)).collect();
            return Err(Err(text.join("\n")));
        }
    };
    // Round-trip through the binary format to exercise it as well.
    let program = jax_bytecode::Program::decode(&program.encode()).expect("bytecode round trip");
    let mut buf: Vec<u8> = Vec::new();
    let result = {
        let mut vm = Vm::new(&program, Box::new(&mut buf));
        vm.run_entry()
    };
    match result {
        Ok(()) => Ok(String::from_utf8(buf).unwrap()),
        Err(e) => Err(Ok(e)),
    }
}

#[test]
fn all_programs() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/programs");
    let mut entries: Vec<_> = fs::read_dir(&dir).unwrap().map(|e| e.unwrap().path()).collect();
    entries.sort();
    let mut ran = 0;
    for path in entries {
        if path.extension().and_then(|e| e.to_str()) != Some("jax") {
            continue;
        }
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let src = fs::read_to_string(&path).unwrap();
        let expected_out: Vec<&str> = src
            .lines()
            .filter_map(|l| l.trim_start().strip_prefix("// out:"))
            .map(|l| l.strip_prefix(' ').unwrap_or(l))
            .collect();
        let expect_error = src.lines().find_map(|l| l.trim_start().strip_prefix("// error:")).map(|s| s.trim());
        let expect_panic = src.lines().find_map(|l| l.trim_start().strip_prefix("// panic:")).map(|s| s.trim());
        let result = run_source(&name, &src);
        match (result, expect_error, expect_panic) {
            (Ok(out), None, None) => {
                let got: Vec<&str> = out.lines().collect();
                assert_eq!(got, expected_out, "stdout mismatch in {name}");
            }
            (Err(Err(msg)), Some(want), _) => {
                assert!(msg.contains(want), "{name}: expected an error containing {want:?}, got:\n{msg}");
            }
            (Err(Ok(RunError::Panic { message, .. })), _, Some(want)) => {
                assert!(message.contains(want), "{name}: expected panic containing {want:?}, got {message:?}");
            }
            (Ok(out), _, _) => panic!("{name}: expected failure but it ran and printed:\n{out}"),
            (Err(Err(msg)), None, _) => panic!("{name}: failed to compile:\n{msg}"),
            (Err(Ok(e)), _, _) => panic!("{name}: unexpected runtime error: {e}"),
        }
        ran += 1;
    }
    assert!(ran > 0, "no test programs found");
}
