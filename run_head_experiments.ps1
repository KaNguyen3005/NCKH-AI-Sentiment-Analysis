$ErrorActionPreference = "Stop"

Write-Host "Running Experiment: CLS Head with Focal Loss"
python src/train.py --config config.yaml --loss_type focal --head_type cls
Write-Host "---------------------------------------------------"

Write-Host "Running Experiment: Mean-Pooling Head with Focal Loss"
python src/train.py --config config.yaml --loss_type focal --head_type mean_pooling
Write-Host "---------------------------------------------------"

Write-Host "Head experiments finished."
