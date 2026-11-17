# Runs inside the dedicated demo console window that record_w03_screen_demo.py captures.
# Shows only the demo banner and the program's own output: no prompt, no paths, no warnings.
param([string]$Alert = 'A07')
$Host.UI.RawUI.WindowTitle = 'W03 live demo'
$env:PYTHONWARNINGS = 'ignore'
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_VERBOSITY = 'error'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
Clear-Host
Write-Host 'InGen Fleet Health - Week 3 live demo: fault alert -> explanation' -ForegroundColor Cyan
Write-Host 'Real run of scripts/w03_demo.py on the synthetic Aido Rover fleet (simulator v0.2); local CPU only' -ForegroundColor DarkGray
& '.venv\Scripts\python.exe' 'scripts\w03_demo.py' '--alert' $Alert '--events' 'media/w03_demo_events.json' 2>$null
Write-Host ''
Write-Host 'Demo finished. The explanations and checks above were generated during this recording.' -ForegroundColor Green
Start-Sleep -Seconds 3
