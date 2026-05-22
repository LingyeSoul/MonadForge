# anima_lora bootstrap installer (Windows / PowerShell).
#
#   irm https://raw.githubusercontent.com/sorryhyun/anima_lora/main/install.ps1 | iex
#
# Installs uv if missing, downloads the latest release tarball (no git
# required), seeds the update baseline so the first `make update` is clean,
# and runs `uv sync`. Mirrors scripts/update.py — keep the two in sync.
#
# Options (env vars, since args don't pass through `irm | iex`):
#   $env:ANIMA_VERSION = 'v1.4.0'   install a specific tag   (default: latest)
#   $env:ANIMA_DIR     = 'C:\path'  target directory         (default: .\anima_lora)

$ErrorActionPreference = 'Stop'
$Repo    = 'sorryhyun/anima_lora'
$Version = $env:ANIMA_VERSION
$Dir     = if ($env:ANIMA_DIR) { $env:ANIMA_DIR } else { 'anima_lora' }

function Say($m)  { Write-Host "==> $m" -ForegroundColor Cyan }
function Die($m)  { Write-Host "error: $m" -ForegroundColor Red; exit 1 }

# 1. uv ----------------------------------------------------------------------
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Say 'installing uv (https://astral.sh/uv)'
  irm https://astral.sh/uv/install.ps1 | iex
  $env:Path = "$env:USERPROFILE\.local\bin;$env:USERPROFILE\.cargo\bin;$env:Path"
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Die 'uv install failed; open a new PowerShell and re-run'
}

# 2. resolve the release tag -------------------------------------------------
if (-not $Version) {
  Say "resolving latest release of $Repo"
  $rel = irm "https://api.github.com/repos/$Repo/releases/latest" `
            -Headers @{ Accept = 'application/vnd.github+json' }
  $Version = $rel.tag_name
  if (-not $Version) { Die 'could not resolve latest release tag from GitHub API' }
}
Say "installing $Repo @ $Version -> $Dir\"

if ((Test-Path $Dir) -and (Get-ChildItem -Force $Dir | Select-Object -First 1)) {
  Die "$Dir\ already exists and is not empty - set `$env:ANIMA_DIR to a different path"
}

# 3. download + extract ------------------------------------------------------
$Tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("anima-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Force -Path $Tmp | Out-Null
try {
  # Use the zipball + .NET unzip, NOT the bundled tar.exe: Windows' bsdtar
  # decodes archive entry names with the active ANSI code page and chokes on
  # the non-ASCII guidebook filenames under docs/guidelines/ (가이드북.md,
  # ガイドブック.md, 指南书.md) with "Invalid empty pathname". GitHub's zipball
  # flags entry names as UTF-8 and .NET's ZipFile honors that.
  $Zipball = "https://github.com/$Repo/archive/refs/tags/$Version.zip"
  Say "downloading $Zipball"
  $zip = Join-Path $Tmp 'release.zip'
  irm $Zipball -OutFile $zip
  Add-Type -AssemblyName System.IO.Compression.FileSystem
  [System.IO.Compression.ZipFile]::ExtractToDirectory($zip, $Tmp, [System.Text.Encoding]::UTF8)
  $top = Get-ChildItem -Directory $Tmp | Select-Object -First 1
  if (-not $top) { Die 'unexpected archive layout' }
  New-Item -ItemType Directory -Force -Path $Dir | Out-Null
  Copy-Item -Path (Join-Path $top.FullName '*') -Destination $Dir -Recurse -Force
} finally {
  Remove-Item -Recurse -Force $Tmp -ErrorAction SilentlyContinue
}

Set-Location $Dir

# 4. seed the update baseline (before uv sync, so .venv isn't hashed) --------
Say 'seeding update baseline (.anima_release.json)'
try {
  uv run --no-project python scripts/update.py --seed-manifest --version $Version
} catch {
  Say 'manifest seed skipped (first `make update` will back up instead - harmless)'
}

# 5. dependencies ------------------------------------------------------------
Say 'running uv sync (this resolves torch + flash-attn; may take a while)'
uv sync

# 6. desktop shortcut (best-effort — never abort the install over this) ------
Say 'creating desktop shortcut (Anima LoRA GUI)'
try {
  uv run python tasks.py gui-shortcut
  if ($LASTEXITCODE -ne 0) { throw "gui-shortcut exited $LASTEXITCODE" }
} catch {
  Say 'desktop shortcut skipped; create it later with: uv run python tasks.py gui-shortcut'
}

Write-Host ""
Write-Host "[OK] installed to $Dir\" -ForegroundColor Green
Write-Host @"

Next steps:
  cd $Dir
  hf auth login            # authenticate for gated model downloads
  python tasks.py download-models   # DiT + Qwen3 text encoder + VAE into models\

Then launch the GUI from the "Anima LoRA GUI" desktop shortcut,
or run:  python tasks.py gui

Update later with:  python tasks.py update
"@
