# TESTE - Q2 Wine Quality (Aluno 1)

Ambiente: Windows, Python 3.14.4, scikit-learn 1.8.0, pandas 3.0.2, numpy 2.4.4.
Executado pelo auditor a partir da raiz `aluno_1/q2_wine_quality`.

## Comandos

```
python src/main.py        # treino + avaliacao + salvar artefatos
python src/inferencia.py  # inferencia em vinho novo
```
(equivalente a `run.ps1`)

## EXIT CODES

| Etapa | Comando | EXIT CODE |
|-------|---------|-----------|
| Treino | `python src/main.py` | **0** |
| Inferencia | `python src/inferencia.py` | **0** |

## Metricas reproduzidas (conjunto de teste, 1300 amostras)

Comparacao dos 3 modelos (selecao por F1-macro):

| Modelo | Acuracia | F1-macro | F1-weighted |
|--------|----------|----------|-------------|
| **RandomForest (vencedor)** | **0.6923** | **0.3959** | **0.6792** |
| KNN | 0.6592 | 0.3713 | 0.6476 |
| GradientBoosting | 0.5838 | 0.3122 | 0.5706 |

Acuracia por classe (recall) do vencedor:

| quality | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---------|---|---|---|---|---|---|---|
| recall | 0.0000 | 0.1163 | 0.7103 | 0.8060 | 0.5602 | 0.3333 | 0.0000 |

Os numeros batem com `outputs/metricas.json` e com o README ate a 4a casa decimal.
Classes 3 e 9 com recall 0.00 (1 amostra de 9 no teste) — limitacao dos dados,
documentada.

## Artefatos confirmados (regenerados nesta execucao)

- `matriz_confusao_RandomForest.png` (matriz do vencedor)
- `comparacao_modelos.json` (3 modelos)
- `metricas.json` (vencedor)
- `modelo_vencedor.joblib` (pipeline + colunas + classes)

## Screenshot textual da inferencia

```
============================================================
Q2 - WINE QUALITY | INFERENCIA EM VINHO NOVO
============================================================
Modelo carregado: RandomForest
Colunas esperadas: ['fixed acidity', 'volatile acidity', 'citric acid',
'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide',
'density', 'pH', 'sulphates', 'alcohol', 'tipo']

Vinho de entrada (branco):
  fixed acidity         : 6.8
  volatile acidity      : 0.31
  citric acid           : 0.34
  residual sugar        : 6.1
  chlorides             : 0.047
  free sulfur dioxide   : 28.0
  total sulfur dioxide  : 122.0
  density               : 0.9952
  pH                    : 3.21
  sulphates             : 0.5
  alcohol               : 10.6
  tipo                  : 1

============================================================
RESULTADO DA INFERENCIA
============================================================
Tipo             : branco
Quality previsto : 6
Probabilidade    : 0.5900 (59.00%) | acaso ~ 0,14 (1/7 classes); quanto MAIOR, melhor
Obs.: as classes 3 e 9 quase nao sao aprendidas (poucas amostras), logo notas
extremas dificilmente serao previstas.
```

Inferencia coerente: vinho branco de exemplo (com valores FABRICADOS) classificado
como quality 6 (a classe majoritaria) com 59% de confianca.

## Conclusao do teste

Reprodutibilidade **CONFIRMADA**: exit 0 nas duas etapas, metricas identicas ao
relatado, inferencia funcional com probabilidade.
