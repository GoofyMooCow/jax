# Releasing and hosting

The website, the installers and every released toolchain are static files served
by GitHub Pages from the `gh-pages` branch of `GoofyMooCow/jax`:

```
https://goofymoocow.github.io/jax/                 website            (site/)
https://goofymoocow.github.io/jax/install.sh       Linux/macOS installer
https://goofymoocow.github.io/jax/install.ps1      Windows installer
https://goofymoocow.github.io/jax/dist/...         launchers + toolchains (built by CI)
```

## One-time setup

1. Create the GitHub repo `GoofyMooCow/jax` and push this code to `main`.
2. Run the **Release** workflow once (Actions tab -> Release -> Run workflow). It creates the `gh-pages` branch.
3. Settings -> Pages -> Source: **Deploy from a branch**, branch `gh-pages`, folder `/ (root)`.
4. The site is live at `https://goofymoocow.github.io/jax/`.

## Publishing a release

1. Bump `version` in the workspace `Cargo.toml` (it becomes the toolchain version).
2. Commit, then tag and push: `git tag v0.1.1 && git push --tags`.
3. The **Release** workflow builds on Linux, macOS and Windows (x64 and arm64), runs the tests on each,
   merges the per-platform results with `cargo xtask merge`, and publishes `dist/` plus the site.
   `keep_files: true` keeps every older toolchain on the branch, because releases are permanent.

Runner labels (`macos-15-intel`, `windows-11-arm`, ...) change over time. If a job cannot find its runner,
edit the matrix in `.github/workflows/release.yml`. A platform that is missing from a release simply is not
offered in `manifest.toml`; the installers report which platforms are published.

Website-only changes under `site/` are published by the **Site** workflow on every push to `main`.

## Building a distribution by hand

```sh
cargo xtask dist --out dist-linux            # on each machine
cargo xtask merge --out public/dist dist-linux dist-mac dist-win
cp -r site/. public/ && touch public/.nojekyll
# upload ./public to any static host
```

## Moving to a new domain

```sh
scripts/set-domain.sh https://jax-lang.org
```

This rewrites the URL in the installers, the website, the built-in default download location
(`crates/jax-toolchain/src/home.rs`) and the docs. Then:

1. For a custom domain add `site/CNAME` containing the bare host name, and configure DNS (GitHub Pages docs).
2. Publish a new release so new launchers embed the new default.
3. **Keep the old address working** (a redirect or the old Pages site) for a long time. Already-installed
   launchers have the old URL built in until the user sets `JAX_DIST_URL` or `dist_url` in `config.toml`.
   A future launcher release should follow redirects and self-update.

## Checklist for a trustworthy release

- [ ] `cargo test --workspace` is green on all three OS jobs
- [ ] `install.sh` and `install.ps1` tested on a clean machine
- [ ] Manifest lists every platform you intend to support
- [ ] (Future) sign the manifest; today only SHA-256 checksums protect downloads
