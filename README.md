# Jax

A portability-first programming language and toolchain, written in Rust.
This repository is **stage 0**: a working launcher, toolchain manager, manifest
parser, lexer, parser, type checker, bytecode compiler and VM, with **zero
third-party dependencies**.

```text
$ jax new hello && cd hello
$ jax run            # first run downloads the toolchain this project pins
Hello, Jax!
$ jax build          # one self-contained file; needs no Jax installed
$ ./build/linux-x64/hello
Hello, Jax!
```

## Install

```sh
# Linux / macOS
curl -fsSL https://goofymoocow.github.io/jax/install.sh | sh

# Windows (PowerShell)
irm https://goofymoocow.github.io/jax/install.ps1 | iex
```

Website and docs: <https://goofymoocow.github.io/jax/>. The installers put the
launcher in `~/.jax/bin` (`%LOCALAPPDATA%\Jax\bin` on Windows), add it to `PATH`
and install the latest toolchain. Releasing and hosting: [docs/RELEASING.md](docs/RELEASING.md).

## Quick start (building from source)

```sh
cargo test --workspace            # 80+ tests, including end-to-end programs
cargo xtask dist                  # builds dist/: launcher + toolchain archive
export JAX_HOME=/tmp/jax-home JAX_DIST_URL=file://$PWD/dist
./dist/bin/linux-x64/jax new hello && cd hello && ../dist/bin/linux-x64/jax run   # use your platform's folder
```

Rust 1.75+ is enough. Nothing is fetched from crates.io.

## Repository layout

| Path | Purpose |
|---|---|
| `crates/jax-launcher` | The tiny `jax` binary: picks, downloads and execs the pinned toolchain |
| `crates/jax-toolchain` | Toolchain store, installer, SHA-256, tar, config (`~/.jax`) |
| `crates/jax-manifest` | `jax.toml` / `jax.lock` parsing, exact versions, own TOML parser |
| `crates/jax-syntax` | Lexer, parser, AST, diagnostics |
| `crates/jax-types` | Name resolution and type checking |
| `crates/jax-codegen` | Typed AST to bytecode |
| `crates/jax-bytecode` | Instruction set, `.jxb` format, disassembler |
| `crates/jax-vm` | Bytecode interpreter |
| `crates/jax-rt` | Runtime stub embedded into built executables |
| `crates/jax-compiler` | Pipeline glue + end-to-end tests (`tests/programs/*.jax`) |
| `crates/jax-driver` | Toolchain-side CLI (`check`, `build`, `run`, `test`, `new`, `dump`) |
| `xtask` | `cargo xtask dist` / `merge` |
| `site/` | The website, `install.sh`, `install.ps1` (served by GitHub Pages) |
| `.github/workflows/` | CI, multi-platform release, site publishing |
| `scripts/` | `set-domain.sh` to move to a new domain |
| `examples/` | Sample programs and a sample project |
| `docs/` | `STATUS.md` (what works), `ARCHITECTURE.md` |

## How portability works today

* **Pinned toolchain.** `jax.toml` requires `jax = "x.y.z"` (ranges are rejected).
  The launcher reads it, installs that exact toolchain if missing (SHA-256
  verified, atomic install, multiple versions side by side) and runs it.
* **Standalone executables.** `jax build` copies the runtime stub and appends the
  compiled program: one file, no Jax needed on the target.
* **Portable bundles.** `jax build --bundle` writes an architecture-independent `.jxb`.
* **Cross-targets.** `--target windows-x64` works when the toolchain ships that runtime.

See [docs/STATUS.md](docs/STATUS.md) for the honest list of what is and is not implemented.
