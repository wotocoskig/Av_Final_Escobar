#!/usr/bin/env bash
# run.sh - Executa o fluxo COMPLETO da Q2 (bash).
# Treino -> outputs -> demo de inferencia.

set -euo pipefail

# Garante execucao a partir da pasta do projeto (onde esta este script).
cd "$(dirname "$0")"

echo "==> [1/2] Treinando o modelo (src/main.py)..."
python src/main.py

echo "==> [2/2] Rodando a demo de inferencia (src/inferencia.py)..."
python src/inferencia.py

echo "==> Fluxo concluido. Veja a pasta outputs/."
