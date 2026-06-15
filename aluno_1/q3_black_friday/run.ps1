# run.ps1 - Executa o fluxo COMPLETO da Q3 (Windows PowerShell).
# Treino dos 3 modelos -> outputs -> demo de inferencia.

$ErrorActionPreference = "Stop"

# Garante execucao a partir da pasta do projeto (onde esta este script).
Set-Location -Path $PSScriptRoot

Write-Host "==> [1/2] Treinando os 3 modelos (src/main.py)..."
python src/main.py
if ($LASTEXITCODE -ne 0) { throw "Falha no treino (main.py)." }

Write-Host "==> [2/2] Rodando a demo de inferencia (src/inferencia.py)..."
python src/inferencia.py
if ($LASTEXITCODE -ne 0) { throw "Falha na inferencia (inferencia.py)." }

Write-Host "==> Fluxo concluido. Veja a pasta outputs/."
