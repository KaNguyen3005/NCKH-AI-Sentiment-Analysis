$ErrorActionPreference = "Stop"

Write-Host "Running Experiment 1: Baseline (CrossEntropy)"
python src/train.py --config config.yaml --loss_type ce
Write-Host "---------------------------------------------------"

Write-Host "Running Experiment 2: Class Weight"
python src/train.py --config config.yaml --loss_type class_weight
Write-Host "---------------------------------------------------"

Write-Host "Running Experiment 3: Focal Loss"
python src/train.py --config config.yaml --loss_type focal
Write-Host "---------------------------------------------------"

Write-Host "All experiments finished."
