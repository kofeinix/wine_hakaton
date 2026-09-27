# Запуск vision-LLM Qwen3-VL-4B для распознавания текста этикеток (OCR) на Windows.
# LLM необязательна: без неё поиск работает только по картинке, без реранка по тексту.
#
#   .\run_llm.ps1            # LM Studio (по умолчанию)
#   .\run_llm.ps1 ollama     # Ollama
param([ValidateSet("lmstudio", "ollama")][string]$Backend = "lmstudio")
$ErrorActionPreference = "Stop"

function Show-Env($Port, $Model) {
    Write-Host ""
    Write-Host "LLM запущена. Пропишите в .env:"
    Write-Host "  LLM__BASE_URL=http://host.docker.internal:$Port/v1"
    Write-Host "  LLM__MODEL_NAME=$Model"
    Write-Host "и перезапустите приложение: docker compose up -d app"
}

if ($Backend -eq "lmstudio") {
    $lms = (Get-Command lms -ErrorAction SilentlyContinue).Source
    if (-not $lms) { $lms = Join-Path $env:USERPROFILE ".lmstudio\bin\lms.exe" }
    if (-not (Test-Path $lms)) { throw "Не найден lms: установите LM Studio (https://lmstudio.ai) и запустите его один раз" }
    $model = "qwen/qwen3-vl-4b"
    & $lms get $model --yes
    # уже загруженную модель не грузим повторно (иначе LM Studio поднимет вторую копию)
    if (-not ((& $lms ps) -match "^$([regex]::Escape($model)) ")) { & $lms load $model --context-length 8192 --yes }
    if (-not ((& $lms server status 2>&1) -match "running")) { & $lms server start --port 1234 }
    Show-Env 1234 $model
}
else {
    if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) { throw "Не найден ollama: https://ollama.com/download" }
    $model = "qwen3-vl:4b"
    try { Invoke-RestMethod http://127.0.0.1:11434/api/version | Out-Null }
    catch { Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden; Start-Sleep -Seconds 5 }
    ollama pull $model
    Show-Env 11434 $model
}
