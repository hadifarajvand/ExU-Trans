# PowerShell script to run training with the venv
$projectRoot = "d:\GitHub\aysan\class-projects\1"
Set-Location $projectRoot

# Activate venv
& .\.venv\Scripts\Activate.ps1

# Run the main training script from scripts directory
Write-Host "Starting ExU-Trans training..."
Write-Host "Output will be saved to: $projectRoot\outputs\"
Write-Host ""

python .\scripts\main.py

Write-Host ""
Write-Host "Training complete. Check outputs\ for results."
