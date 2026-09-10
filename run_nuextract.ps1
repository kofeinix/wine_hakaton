param(
    [ValidateSet("auto", "cpu", "mps", "gpu")]
    [string]$Device = $env:NUEXTRACT_DEVICE,

    [string]$HostName = $(if ($env:NUEXTRACT_HOST) { $env:NUEXTRACT_HOST } else { "127.0.0.1" }),
    [int]$Port = $(if ($env:NUEXTRACT_PORT) { [int]$env:NUEXTRACT_PORT } else { 11434 }),
    [string]$Model = $(if ($env:OLLAMA_MODEL) { $env:OLLAMA_MODEL } else { "numind/nuextract3:q4_k_m" }),
    [string]$ServedModelName = $(if ($env:NUEXTRACT_SERVED_MODEL_NAME) { $env:NUEXTRACT_SERVED_MODEL_NAME } else { "nuextract3" }),
    [int]$MaxModelLen = $(if ($env:NUEXTRACT_MAX_MODEL_LEN) { [int]$env:NUEXTRACT_MAX_MODEL_LEN } else { 4000 }),
    [switch]$StopModelOnExit = $true
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$IsWindowsPlatform = if (Get-Variable IsWindows -ErrorAction SilentlyContinue) { $IsWindows } else { $env:OS -eq "Windows_NT" }

if ([string]::IsNullOrWhiteSpace($Device)) { $Device = "auto" }

$ServeProcess  = $null
$StartedByUs   = $false
$DesktopAppPid = $null

$script:LogTargets = @()

function Write-Log {
    param([string]$Message)
    Write-Host "[nuextract] $Message"
}

function Write-OllamaLog {
    param([string]$Line)
    Write-Host "[ollama]   $Line"
}

function Fail {
    param([string]$Message)
    Write-Error $Message
    exit 1
}

function Test-Command {
    param([string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Test-Http {
    param([string]$Url)
    try {
        Invoke-RestMethod -Uri $Url -Method Get -TimeoutSec 2 | Out-Null
        return $true
    } catch { return $false }
}

function Wait-Http {
    param([string]$Url, [int]$TimeoutSeconds = 60)
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-Http -Url $Url) { return $true }
        Start-Sleep -Seconds 1
    }
    return $false
}

function Add-LogTarget {
    param([string]$Path)
    $pos = 0
    if (Test-Path $Path) { $pos = (Get-Item $Path).Length }
    $script:LogTargets += [pscustomobject]@{ Path = $Path; Pos = $pos }
    Write-Log "log target added: $Path"
}

function Read-OllamaLogs {
    foreach ($t in $script:LogTargets) {
        if (-not (Test-Path $t.Path)) { continue }
        try {
            $fs = [System.IO.File]::Open(
                $t.Path,
                [System.IO.FileMode]::Open,
                [System.IO.FileAccess]::Read,
                [System.IO.FileShare]::ReadWrite)

            if ($t.Pos -gt $fs.Length) { $t.Pos = 0 }
            $fs.Seek($t.Pos, [System.IO.SeekOrigin]::Begin) | Out-Null

            $sr = New-Object System.IO.StreamReader($fs, [System.Text.Encoding]::UTF8)
            while (-not $sr.EndOfStream) {
                $line = $sr.ReadLine()
                if ($null -ne $line) { Write-OllamaLog $line }
            }
            $t.Pos = $fs.Position
            $sr.Close()
            $fs.Close()
        }
        catch { }
    }
}

if (-not $IsWindowsPlatform) {
    Fail "run_nuextract.ps1 is intended for Windows. Use ./run_nuextract.sh on macOS/Linux."
}

if ($Device -eq "auto") {
    if ((Test-Command "nvidia-smi")) {
        try { nvidia-smi -L | Out-Null; $Device = "gpu" }
        catch { $Device = "cpu" }
    } else { $Device = "cpu" }
}

if ($Device -eq "mps") {
    Fail "Windows launcher supports cpu and gpu through Ollama. Requested: mps"
}

if ($Device -eq "gpu") {
    if (-not (Test-Command "nvidia-smi")) {
        Fail "Windows gpu mode requires an NVIDIA GPU and nvidia-smi. Use -Device cpu."
    }
    try { nvidia-smi -L | Out-Null }
    catch { Fail "Windows gpu mode was requested, but no usable NVIDIA GPU was detected. Use -Device cpu." }
}
elseif ($Device -ne "cpu") {
    Fail "Windows launcher supports cpu and gpu through Ollama. Requested: $Device"
}

# --- NEW: choose num_gpu for Ollama based on requested device -------------
# 0  -> force CPU (no layers offloaded to GPU)
# -1 -> offload all layers to GPU
# $null -> leave Ollama default ("auto")
$NumGpu = switch ($Device) {
    "cpu" { 0 }
    "gpu" { -1 }
    default { $null }
}
# -------------------------------------------------------------------------

if (-not (Test-Command "ollama")) {
    Write-Host @"
Ollama is required on Windows.

Install it with one of these options:
  winget install Ollama.Ollama
  https://ollama.com/download

Then rerun:
  powershell -ExecutionPolicy Bypass -File .\run_nuextract.ps1 -Device cpu
"@
    exit 1
}

Write-Log "selected $Device on Windows -> using Ollama"

$ListenHost   = if ($HostName -eq "0.0.0.0") { "127.0.0.1" } else { $HostName }
$LocalBaseUrl = "http://${ListenHost}:$Port"
$TagsUrl      = "$LocalBaseUrl/api/tags"

$DesktopAppProc = Get-Process -Name "ollama app" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($null -ne $DesktopAppProc) { $DesktopAppPid = $DesktopAppProc.Id }

if (Test-Http -Url $TagsUrl) {
    if ($null -ne $DesktopAppPid) {
        Write-Log "Ollama is already serving $LocalBaseUrl (desktop app PID $DesktopAppPid). Reusing it, will not stop."
        Add-LogTarget (Join-Path $env:LOCALAPPDATA "Ollama\server.log")
    } else {
        Write-Log "Ollama is already serving $LocalBaseUrl (external process). Reusing it, will not stop."
        $maybeLog = Join-Path $env:LOCALAPPDATA "Ollama\server.log"
        if (Test-Path $maybeLog) { Add-LogTarget $maybeLog }
        else { Write-Log "No server.log found; external Ollama logs are not accessible." }
    }
} else {
    if ($null -ne $DesktopAppPid) {
        Write-Log "Port $Port is free, but 'ollama app' is running (PID $DesktopAppPid)."
        Write-Log "Stopping desktop app so we can start and later stop our own server."
        Stop-Process -Id $DesktopAppPid -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 1
    }

    $outFile = Join-Path $env:TEMP "ollama-serve.out.log"
    $errFile = Join-Path $env:TEMP "ollama-serve.err.log"
    Remove-Item $outFile, $errFile -Force -ErrorAction SilentlyContinue

    Write-Log "starting Ollama server (logging to $outFile and $errFile)"
    if (-not $env:OLLAMA_HOST) { $env:OLLAMA_HOST = "${HostName}:$Port" }

    # Best-effort: also nudge the server itself for CPU mode.
    if ($Device -eq "cpu") {
        $env:OLLAMA_NUM_GPU = "0"
    } else {
        Remove-Item Env:OLLAMA_NUM_GPU -ErrorAction SilentlyContinue
    }

    $ServeProcess = Start-Process -FilePath "ollama" `
        -ArgumentList "serve" `
        -RedirectStandardOutput $outFile `
        -RedirectStandardError  $errFile `
        -WindowStyle Hidden `
        -PassThru

    $StartedByUs = $true

    Add-LogTarget $outFile
    Add-LogTarget $errFile

    $deadline = (Get-Date).AddSeconds(60)
    $ready = $false
    while ((Get-Date) -lt $deadline) {
        Read-OllamaLogs
        if (Test-Http -Url $TagsUrl) { $ready = $true; break }
        Start-Sleep -Milliseconds 500
    }
    if (-not $ready) {
        Read-OllamaLogs
        Fail "Ollama did not become ready at $TagsUrl"
    }
}

Write-Log "pulling/checking Ollama model: $Model"
& ollama pull $Model

$Modelfile = New-TemporaryFile
try {
    $ModelfileContent = "FROM $Model`nPARAMETER num_ctx $MaxModelLen"
    if ($null -ne $NumGpu) {
        $ModelfileContent += "`nPARAMETER num_gpu $NumGpu"
        Write-Log "device=$Device -> setting num_gpu=$NumGpu in Modelfile"
    } else {
        Write-Log "device=$Device -> leaving num_gpu at Ollama default (auto)"
    }
    $ModelfileContent | Set-Content -Path $Modelfile -Encoding utf8

    Write-Log "creating Ollama alias: $ServedModelName -> $Model"
    & ollama create $ServedModelName -f $Modelfile
}
finally {
    Remove-Item -Path $Modelfile -Force -ErrorAction SilentlyContinue
}

Write-Host @"
[nuextract] Ollama OpenAI-compatible endpoint is ready:
  base_url: $LocalBaseUrl/v1
  model:    $ServedModelName
  device:   $Device (num_gpu=$(if ($null -eq $NumGpu) { 'auto' } else { $NumGpu }))

For Docker Desktop on Windows, use:
  LLM__BASE_URL=http://host.docker.internal:$Port/v1
  LLM__API_KEY=ollama
  LLM__MODEL_NAME=$ServedModelName
"@

if ($StartedByUs) {
    Write-Log "Ollama started by this script (PID $($ServeProcess.Id)). Ctrl+C will stop server and unload model."
} else {
    Write-Log "Ollama is not managed by this script. Ctrl+C will unload model, server left alone."
}
Write-Log "Console held. Ctrl+C to exit."

try {
    while ($true) {
        Read-OllamaLogs
        Start-Sleep -Milliseconds 300
    }
}
finally {
    if ($StopModelOnExit) {
        try {
            Write-Log "Unloading model '$ServedModelName' from memory (ollama stop)..."
            & ollama stop $ServedModelName 2>$null | Out-Null
        } catch {
            Write-Log "ollama stop failed: $($_.Exception.Message)"
        }
    }

    if ($StartedByUs -and $null -ne $ServeProcess -and -not $ServeProcess.HasExited) {
        Write-Log "Stopping Ollama server (PID $($ServeProcess.Id))..."
        try { Stop-Process -Id $ServeProcess.Id -Force -ErrorAction SilentlyContinue } catch { }
        Start-Sleep -Milliseconds 300

        Get-Process -Name "ollama" -ErrorAction SilentlyContinue |
            Where-Object { $null -eq $DesktopAppPid -or $_.Id -ne $DesktopAppPid } |
            ForEach-Object {
                Write-Log "Cleaning up ollama PID $($_.Id)..."
                Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
            }
    } else {
        Write-Log "Ollama server was not started by this script; not stopping it."
    }

    Write-Log "Exit."
}