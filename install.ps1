# Jax installer for Windows (PowerShell 5.1 or newer).
#
#   irm https://goofymoocow.github.io/jax/install.ps1 | iex
#
# Optional environment variables (set them before running):
#   JAX_NO_MODIFY_PATH=1   do not add the launcher to your user PATH
#   JAX_NO_TOOLCHAIN=1     install only the launcher
#   JAX_VERSION=X.Y.Z      install this toolchain instead of the latest
#   JAX_HOME               install location (default %LOCALAPPDATA%\Jax)
#   JAX_DIST_URL           override the download site

$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12

$JaxSite = 'https://goofymoocow.github.io/jax'
$Dist = if ($env:JAX_DIST_URL) { $env:JAX_DIST_URL.TrimEnd('/') } else { "$JaxSite/dist" }
$JaxHome = if ($env:JAX_HOME) { $env:JAX_HOME } else { Join-Path $env:LOCALAPPDATA 'Jax' }
$BinDir = Join-Path $JaxHome 'bin'

function Say($msg) { Write-Host "jax-install: $msg" }
function Fail($msg) { Write-Host "jax-install: error: $msg" -ForegroundColor Red; throw $msg }

function Get-Remote($url, $dest) {
    if ($url -like 'file://*') {
        $path = $url.Substring(7)
        if ($path -match '^/[A-Za-z]:') { $path = $path.Substring(1) }
        Copy-Item -LiteralPath $path -Destination $dest -Force
    } else {
        Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $dest
    }
}

# ---- detect the platform ------------------------------------------------------
$archRaw = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }
switch ($archRaw) {
    'AMD64' { $arch = 'x64' }
    'ARM64' { $arch = 'arm64' }
    default { Fail "unsupported CPU architecture: $archRaw" }
}
$target = "windows-$arch"

# ---- download and verify the launcher --------------------------------------------
$tmp = Join-Path ([IO.Path]::GetTempPath()) ("jax-install-" + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $tmp | Out-Null
try {
    Say "installing the Jax launcher for $target"
    $exe = Join-Path $tmp 'jax.exe'
    $sum = Join-Path $tmp 'jax.exe.sha256'
    try {
        Get-Remote "$Dist/bin/$target/jax.exe" $exe
        Get-Remote "$Dist/bin/$target/jax.exe.sha256" $sum
    } catch {
        Fail "could not download the launcher from $Dist/bin/$target/ (is $target published yet?)"
    }
    $expected = ((Get-Content -LiteralPath $sum -Raw).Trim() -split '\s+')[0].ToLower()
    $actual = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash.ToLower()
    if ($expected -ne $actual) {
        Fail "checksum mismatch for the launcher`n  expected $expected`n  actual   $actual`nnothing was installed"
    }

    New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
    $dest = Join-Path $BinDir 'jax.exe'
    if (Test-Path $dest) {
        # A running jax.exe cannot be overwritten but can be renamed.
        Move-Item -Force -LiteralPath $dest -Destination ($dest + '.old') -ErrorAction SilentlyContinue
    }
    Move-Item -Force -LiteralPath $exe -Destination $dest
    Remove-Item -Force -ErrorAction SilentlyContinue ($dest + '.old')
    Say "installed $dest"
} finally {
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue $tmp
}

# ---- put it on the user PATH ---------------------------------------------------------
if (-not $env:JAX_NO_MODIFY_PATH) {
    $userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
    $parts = if ($userPath) { $userPath -split ';' | Where-Object { $_ } } else { @() }
    if ($parts -notcontains $BinDir) {
        $newPath = (@($parts) + $BinDir) -join ';'
        [Environment]::SetEnvironmentVariable('Path', $newPath, 'User')
        Say "added $BinDir to your user PATH"
    }
    if (($env:Path -split ';') -notcontains $BinDir) { $env:Path = "$env:Path;$BinDir" }
}

# ---- install a toolchain ---------------------------------------------------------------
$env:JAX_DIST_URL = $Dist
$env:JAX_HOME = $JaxHome
if (-not $env:JAX_NO_TOOLCHAIN) {
    $version = $env:JAX_VERSION
    if (-not $version) {
        $version = (& (Join-Path $BinDir 'jax.exe') toolchain latest).Trim()
        if ($LASTEXITCODE -ne 0 -or -not $version) { Fail 'could not look up the latest toolchain' }
    }
    & (Join-Path $BinDir 'jax.exe') toolchain install $version --set-default
    if ($LASTEXITCODE -ne 0) { Fail "installing toolchain $version failed" }
}

Write-Host ""
Write-Host "Jax is installed."
Write-Host ""
Write-Host "  Launcher:  $BinDir\jax.exe"
Write-Host "  Home:      $JaxHome   (toolchains live in $JaxHome\toolchains)"
Write-Host ""
Write-Host "Open a new terminal, then try:"
Write-Host ""
Write-Host "  jax new hello; cd hello; jax run"
Write-Host ""
Write-Host "Docs: $JaxSite/docs/"
