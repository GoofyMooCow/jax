# Implementation status (stage 0, Jax 0.1.0)

## Works
* Launcher: version resolution (`+x.y.z`, `JAX_TOOLCHAIN`, `jax.toml`, `//! jax x.y.z` header,
  config default, newest installed, newest available), auto-download, `jax toolchain list|install|remove|default|which|latest|prune|path`.
* Toolchain installer: manifest + archive fetch (`file://`, plain paths, `http(s)://` via system `curl`),
  SHA-256 verification, file locking, atomic rename, `JAX_OFFLINE`, `JAX_DIST_URL`.
* `jax.toml`: full validation (exact versions, targets, dependencies, profiles, capsule, tasks); `jax.lock` read/write.
* Language: functions (named + default args), `let`/`var`/`const`, Int/Float/Bool/Text/Unit,
  string interpolation, if/else, `if let`, `match` (exhaustive for enums/Bool/optionals, guards, ranges, or-patterns),
  while/loop/for (ranges and lists), structs, `impl` methods, `mut self`, `with`, enums with payloads,
  optionals (`T?`, `none`, `??`, `?.`, `!`, `some(p)`), lists (`push`, `get`, `len`, indexing, assignment), `test` blocks.
* Diagnostics with source excerpts, error codes and "did you mean" help.
* Runtime: checked integer arithmetic, bounds checks, panics with stack traces (exit code 101).

## Parsed but rejected with `E0900 ... not supported yet`
Generics, trait bounds and default trait methods, closures, `Result` and `?`, `async`/`await`,
`Map`/`Set`, tuples, sized integer types, multi-file modules (`use` of non-`std` paths).
`impl Trait for Type` currently registers the methods but does not check the trait.

## Planned (commands print "planned")
fmt, lint, doc, bench, package manager (`add`, `vendor`, `publish`), capsules (`.jcap`), `jax dist`,
language server, registry.

## Known gaps vs. the specification
* **Execution is a bytecode VM**, not native machine code. Built executables are the VM runtime plus
  the program. The IR/bundle format is the stable seam where native backends will attach.
* No release signature verification yet (hashes only). The distribution URL is a placeholder.
* HTTPS downloads shell out to `curl`.
* Only `main()` with no arguments; no stdin or file APIs in the standard library yet.
