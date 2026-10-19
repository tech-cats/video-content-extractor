#requires -Version 5.1
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new()
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [System.Text.UTF8Encoding]::new()
$env:PYTHONUTF8 = "1"

function Test-PythonCommand {
    param([string]$Exe, [string[]]$Prefix)
    try {
        & $Exe @Prefix -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" *> $null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    }
}

$pythonExe = $null
$pythonPrefix = @()

if (Get-Command py -ErrorAction SilentlyContinue) {
    foreach ($version in @("3.12", "3.11", "3.13", "3.10", "3")) {
        if (Test-PythonCommand -Exe "py" -Prefix @("-$version")) {
            $pythonExe = "py"
            $pythonPrefix = @("-$version")
            break
        }
    }
}

if (-not $pythonExe) {
    $candidates = @(
        (Get-Command python -ErrorAction SilentlyContinue),
        (Get-Command python3 -ErrorAction SilentlyContinue)
    ) | Where-Object { $_ }
    foreach ($candidate in $candidates) {
        if (Test-PythonCommand -Exe $candidate.Source -Prefix @()) {
            $pythonExe = $candidate.Source
            break
        }
    }
}

if (-not $pythonExe) {
    Write-Host "Python 3.10+ was not found. Trying to install Python 3.12 with winget..."
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        & winget install --id Python.Python.3.12 --exact --scope user `
            --accept-source-agreements --accept-package-agreements
    }

    $fallbackPaths = @(
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python314\python.exe",
        "$env:ProgramFiles\Python312\python.exe"
    )
    foreach ($path in $fallbackPaths) {
        if ((Test-Path -LiteralPath $path) -and (Test-PythonCommand -Exe $path -Prefix @())) {
            $pythonExe = $path
            break
        }
    }
}

if (-not $pythonExe) {
    throw "Python 3.10 or newer could not be found or installed. Install Python, then rerun run.ps1."
}

& $pythonExe @pythonPrefix "bootstrap.py" @args
exit $LASTEXITCODE
