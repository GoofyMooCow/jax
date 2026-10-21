# Architecture

```
source ──jax-syntax──▶ AST ──jax-types──▶ AST + TypeInfo ──jax-codegen──▶ Program (jax-bytecode)
                                                                   │
                                          ┌────────────────────────┼───────────────────────┐
                                     jax-vm (run)          .jxb bundle (encode)    exe = jax-rt + payload
```

* **Side tables, not a second tree.** The checker records resolutions (call targets, argument
  order, field indices, enum tags) in `TypeInfo`, keyed by node id. Codegen consumes them.
* **Value semantics.** Structs and lists are `Rc` copy-on-write. Field/index assignment and
  `mut self` methods compile to "rebuild and store back into the place".
* **Statement termination.** The lexer tags tokens with `nl_before`; the parser ends expressions
  at a newline unless inside `()`/`[]` or the next line starts with `.`.
* **Toolchain vs launcher.** The launcher never changes behavior; everything is in
  `toolchains/<version>/` (`bin/jax-driver`, `runtimes/<target>/jax-rt`).
* **Zero dependencies.** TOML, SHA-256, tar and the bytecode codec are implemented in-repo.
