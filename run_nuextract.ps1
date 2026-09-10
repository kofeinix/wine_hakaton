param(
    [ValidateSet("auto", "cpu", "mps", "gpu")]
    [string]$Device = $env:NUEXTRACT_DEVICE,

    [string]$HostName = $(if ($env:NUEXTRACT_HOST) { $env:NUEXTRACT_HOST } else { "127.0.0.1" }),
    [int]$Port = $(if ($env:NUEXTRACT_PORT) { [int]$env:NUEXTRACT_PORT } else { 11434 }),
    [string]$Model = $(if ($env:OLLAMA_MODEL) { $env:OLLAMA_MODEL } else { "numind/nuextract3:q4_k_m" }),
    [string]$ServedModelName = $(if ($env:NUEXTRACT_SERVED_MODEL_NAME) { $env:NUEXTRACT_SERVED_MODEL_NAME } else { "nuextract3" }),
    [int]$MaxModelLen = $(if ($env:NUEXTRACT_MAX_MODEL_LEN) { [int]$env:NUEXTRACT_MAX_MODEL_LEN } else { 4000 })
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$IsWindowsPlatform = if (Get-Variable IsWindows -ErrorAction SilentlyContinue) { $IsWindows } else { $env:OS -eq "Windows_NT" }

if ([string]::IsNullOrWhiteSpace($Device)) {
    $Device = "auto"
}

function Write-Log {
    param([string]$Message)
    Write-Host "[nuextract] $Message"
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
    }
    catch {
        return $false
    }
}

function Wait-Http {
    param(
        [string]$Url,
        [int]$TimeoutSeconds = 60
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-Http -Url $Url) {
            return $true
        }
        Start-Sleep -Seconds 1
    }
    return $false
}

if (-not $IsWindowsPlatform) {
    Fail "run_nuextract.ps1 is intended for Windows. Use ./run_nuextract.sh on macOS/Linux."
}

if ($Device -eq "auto") {
    if ((Test-Command "nvidia-smi")) {
        try {
            nvidia-smi -L | Out-Null
            $Device = "gpu"
        }
        catch {
            $Device = "cpu"
        }
    }
    else {
        $Device = "cpu"
    }
}

if ($Device -eq "mps") {
    Fail "Windows launcher supports cpu and gpu through Ollama. Requested: mps"
}

if ($Device -eq "gpu") {
    if (-not (Test-Command "nvidia-smi")) {
        Fail "Windows gpu mode requires an NVIDIA GPU and nvidia-smi. Use -Device cpu."
    }
    try {
        nvidia-smi -L | Out-Null
    }
    catch {
        Fail "Windows gpu mode was requested, but no usable NVIDIA GPU was detected. Use -Device cpu."
    }
}
elseif ($Device -ne "cpu") {
    Fail "Windows launcher supports cpu and gpu through Ollama. Requested: $Device"
}

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

$ListenHost = if ($HostName -eq "0.0.0.0") { "127.0.0.1" } else { $HostName }
$LocalBaseUrl = "http://${ListenHost}:$Port"
$TagsUrl = "$LocalBaseUrl/api/tags"
$StartedProcess = $null

if (-not (Test-Http -Url $TagsUrl)) {
    Write-Log "starting Ollama server"
    if (-not $env:OLLAMA_HOST) {
        $env:OLLAMA_HOST = "${HostName}:$Port"
    }
    $StartedProcess = Start-Process -FilePath "ollama" -ArgumentList "serve" -PassThru -WindowStyle Hidden
    if (-not (Wait-Http -Url $TagsUrl -TimeoutSeconds 60)) {
        Fail "Ollama did not become ready at $TagsUrl"
    }
}

Write-Log "pulling/checking Ollama model: $Model"
ollama pull $Model

$Modelfile = New-TemporaryFile
try {
    @"
FROM $Model
PARAMETER num_ctx $MaxModelLen
"@ | Set-Content -Path $Modelfile -Encoding utf8

    Write-Log "creating Ollama alias: $ServedModelName -> $Model"
    ollama create $ServedModelName -f $Modelfile
}
finally {
    Remove-Item -Path $Modelfile -Force -ErrorAction SilentlyContinue
}

Write-Host @"
[nuextract] Ollama OpenAI-compatible endpoint is ready:
  base_url: $LocalBaseUrl/v1
  model:    $ServedModelName

For Docker Desktop on Windows, use:
  LLM__BASE_URL=http://host.docker.internal:$Port/v1
  LLM__API_KEY=ollama
  LLM__MODEL_NAME=$ServedModelName
"@

if ($null -ne $StartedProcess) {
    Write-Log "Ollama was started by this launcher. Press Ctrl+C to stop this PowerShell session; stop Ollama separately if needed."
    Wait-Process -Id $StartedProcess.Id
}
