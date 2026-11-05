# Runs inside the dedicated demo console window that record_screen_demo.py captures.
# Shows only the demo banner and the program's own output: no prompt, no paths, no warnings.
$Host.UI.RawUI.WindowTitle = 'W02 live demo'
$env:PYTHONWARNINGS = 'ignore'
$env:MLFLOW_DISABLE_AGENT_HINT = '1'
$env:PYTHONIOENCODING = 'utf-8'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
Clear-Host
Write-Host 'InGen Fleet Health - Week 2 live demo (screen recording)' -ForegroundColor Cyan
Write-Host 'Real run of scripts/w02_demo.py: one walk-forward fold on unseen robots' -ForegroundColor Cyan
Write-Host 'Synthetic Aido Rover fleet (simulator v0.2)' -ForegroundColor DarkGray
Write-Host ''
Start-Sleep -Seconds 1
& '.venv\Scripts\python.exe' 'scripts\w02_demo.py' '--data-dir' 'data/w02' '--out' 'media/w02_screen_run_log.json' 2>$null
Write-Host ''
Write-Host 'Demo finished. Every number above was computed during this recording.' -ForegroundColor Green
Start-Sleep -Seconds 3
