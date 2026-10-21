#!/bin/sh
# Jax installer for Linux and macOS.
#
#   curl -fsSL https://goofymoocow.github.io/jax/install.sh | sh
#
# Options (pass after `sh -s --`):
#   --no-modify-path   do not edit shell startup files
#   --no-toolchain     install only the launcher; toolchains download on first use
#   --version X.Y.Z    install this toolchain instead of the latest
#   --help
#
# Environment: JAX_HOME (default ~/.jax), JAX_DIST_URL (override the download site).
set -eu

JAX_SITE="https://goofymoocow.github.io/jax"
DIST="${JAX_DIST_URL:-$JAX_SITE/dist}"
DIST="${DIST%/}"
JAX_HOME_DIR="${JAX_HOME:-$HOME/.jax}"
BIN_DIR="$JAX_HOME_DIR/bin"

MODIFY_PATH=1
INSTALL_TOOLCHAIN=1
WANT_VERSION=""

say() { printf 'jax-install: %s\n' "$*" >&2; }
err() { say "error: $*"; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --no-modify-path) MODIFY_PATH=0 ;;
    --no-toolchain) INSTALL_TOOLCHAIN=0 ;;
    --version) shift; [ $# -gt 0 ] || err "--version needs a value"; WANT_VERSION="$1" ;;
    --help|-h) sed -n '2,13p' "$0" 2>/dev/null | sed 's/^# \{0,1\}//' || true; exit 0 ;;
    *) err "unknown option: $1 (try --help)" ;;
  esac
  shift
done

command -v curl >/dev/null 2>&1 || err "curl is required but was not found"

# ---- detect the platform ------------------------------------------------------
os=$(uname -s)
arch=$(uname -m)
case "$os" in
  Linux) os=linux ;;
  Darwin) os=macos ;;
  *) err "unsupported operating system: $os (on Windows use install.ps1)" ;;
esac
case "$arch" in
  x86_64|amd64) arch=x64 ;;
  aarch64|arm64) arch=arm64 ;;
  *) err "unsupported CPU architecture: $arch" ;;
esac
# A shell running under Rosetta reports x86_64 on Apple Silicon.
if [ "$os" = macos ] && [ "$arch" = x64 ] && [ "$(sysctl -n sysctl.proc_translated 2>/dev/null || echo 0)" = 1 ]; then
  arch=arm64
fi
target="$os-$arch"

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  elif command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | cut -d' ' -f1
  else err "need sha256sum or shasum to verify the download"; fi
}

# ---- download and verify the launcher --------------------------------------------
tmp=$(mktemp -d 2>/dev/null || mktemp -d -t jax-install)
trap 'rm -rf "$tmp"' EXIT INT TERM

say "installing the Jax launcher for $target"
curl -fsSL --retry 3 -o "$tmp/jax" "$DIST/bin/$target/jax" \
  || err "could not download $DIST/bin/$target/jax (is $target published yet?)"
curl -fsSL --retry 3 -o "$tmp/jax.sha256" "$DIST/bin/$target/jax.sha256" \
  || err "could not download the checksum for the launcher"

expected=$(cut -d' ' -f1 "$tmp/jax.sha256" | tr 'A-F' 'a-f')
actual=$(sha256_of "$tmp/jax")
[ "$expected" = "$actual" ] || err "checksum mismatch for the launcher
  expected $expected
  actual   $actual
nothing was installed"

mkdir -p "$BIN_DIR"
chmod +x "$tmp/jax"
mv -f "$tmp/jax" "$BIN_DIR/jax"
say "installed $BIN_DIR/jax"

# ---- put it on PATH --------------------------------------------------------------
path_line="export PATH=\"$BIN_DIR:\$PATH\"  # added by the Jax installer"
if [ "$MODIFY_PATH" = 1 ]; then
  case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *)
      touched=0
      for rc in "$HOME/.profile" "$HOME/.bashrc" "$HOME/.zshrc"; do
        if [ -f "$rc" ] && ! grep -qF "$BIN_DIR" "$rc" 2>/dev/null; then
          printf '\n%s\n' "$path_line" >> "$rc"
          say "added $BIN_DIR to PATH in $rc"
          touched=1
        fi
      done
      if [ "$touched" = 0 ] && [ ! -f "$HOME/.profile" ]; then
        printf '%s\n' "$path_line" > "$HOME/.profile"
        say "created $HOME/.profile with the PATH entry"
      fi
      ;;
  esac
fi

# ---- install a toolchain ---------------------------------------------------------
export JAX_DIST_URL="$DIST"
export JAX_HOME="$JAX_HOME_DIR"
if [ "$INSTALL_TOOLCHAIN" = 1 ]; then
  if [ -z "$WANT_VERSION" ]; then
    WANT_VERSION=$("$BIN_DIR/jax" toolchain latest) || err "could not look up the latest toolchain"
  fi
  "$BIN_DIR/jax" toolchain install "$WANT_VERSION" --set-default
fi

cat >&2 <<MSG

Jax is installed.

  Launcher:  $BIN_DIR/jax
  Home:      $JAX_HOME_DIR   (toolchains live in $JAX_HOME_DIR/toolchains)

Open a new terminal (or run: export PATH="$BIN_DIR:\$PATH"), then try:

  jax new hello && cd hello && jax run

Docs: $JAX_SITE/docs/
MSG
