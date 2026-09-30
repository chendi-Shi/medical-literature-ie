$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
Set-Location -LiteralPath $taskRoot
$runtimeFile = Join-Path $taskRoot '.runtime.json'
if (Test-Path -LiteralPath $runtimeFile) {
    $taskRuntime = Get-Content -LiteralPath $runtimeFile -Raw | ConvertFrom-Json
    $taskPython = $taskRuntime.python
} elseif (Test-Path -LiteralPath (Join-Path $taskRoot '.venv\Scripts\python.exe')) {
    $taskPython = Join-Path $taskRoot '.venv\Scripts\python.exe'
} else {
    $taskPython = 'python'
}
Write-Host 'NLP Lab: http://127.0.0.1:8778'
& $taskPython -m nlp_lab.cli serve --port 8778
