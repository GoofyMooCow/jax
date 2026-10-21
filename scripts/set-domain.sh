#!/bin/sh
# Point everything at a new home for Jax.
#
#   scripts/set-domain.sh https://jax-lang.org
#   scripts/set-domain.sh https://example.com/jax
#
# Rewrites the site URL in the installers, the website, the built-in default
# download location and the docs. Run it from the repository root, commit, and
# publish a new release. Old launchers keep working with their built-in URL, so
# keep the old address redirecting for a while.
set -eu

[ $# -eq 1 ] || { echo "usage: $0 <new site URL>" >&2; exit 2; }
new=$(printf '%s' "$1" | sed 's|/*$||')
old=$(grep -o 'JAX_SITE="[^"]*"' site/install.sh | head -1 | cut -d'"' -f2)
[ -n "$old" ] || { echo "cannot find the current site URL in site/install.sh" >&2; exit 1; }
[ "$old" != "$new" ] || { echo "already $new"; exit 0; }

# Escape for sed.
esc() { printf '%s' "$1" | sed 's/[|&\\]/\\&/g'; }
o=$(esc "$old"); n=$(esc "$new")

for f in $(grep -rlF "$old" site crates README.md docs scripts .github 2>/dev/null | grep -v '^scripts/set-domain.sh$'); do
  sed "s|$o|$n|g" "$f" > "$f.tmp" && cat "$f.tmp" > "$f" && rm -f "$f.tmp"
  echo "updated $f"
done
echo "done: $old -> $new"
echo "For a custom domain also add a site/CNAME file containing the bare host name."
