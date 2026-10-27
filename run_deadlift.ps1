$ErrorActionPreference = 'Continue'
$env:CLIPCAFE_VOCAL_ONLY = '1'
$src = 'D:\caaapct\Edits\DEADLIFT_PRS_CARBS_ANAVAR.webm'
$out = 'D:\ai\clipcafe\out\gym_deadlift_vocal'
$log = 'D:\ai\clipcafe\out\gym_deadlift_vocal\run.log'
New-Item -ItemType Directory -Force -Path $out | Out-Null
Set-Location 'D:\ai\clipcafe'
"START $(Get-Date -Format o)" | Out-File -FilePath $log -Encoding utf8
python clipcafe.py export --video $src --outfile "$out\scenes.json" --force-whisper --scorer heuristic --top 400 *>> $log
"EXPORT DONE $(Get-Date -Format o)" | Out-File -FilePath $log -Append -Encoding utf8
