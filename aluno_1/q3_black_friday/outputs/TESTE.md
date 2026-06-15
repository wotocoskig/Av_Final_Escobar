# TESTE — Q3 Black Friday (Projeto A / aluno_1)

Ambiente: Windows 10, Python 3.14.4, scikit-learn 1.8.0, pandas 3.0.2, numpy 2.4.4, matplotlib 3.11.0, seaborn 0.13.2. Sem venv.

## Comandos executados

```
python src/main.py          # treino dos 3 modelos + artefatos
python src/inferencia.py    # demo de inferência (venda nova)
```

(equivale a `run.ps1` / `run.sh`, inspecionados: ambos rodam `main.py` depois `inferencia.py` com checagem de exit code).

## EXIT CODES

| Etapa | Comando | EXIT CODE |
|-------|---------|-----------|
| Treino | `python src/main.py` | **0** |
| Inferência | `python src/inferencia.py` | **0** |

## Reprodutibilidade

Execução do zero (dataset sintético regerado, seed 42). Métricas idênticas às do `metricas.json` versionado.

## Métricas por alvo (conjunto de teste: 3000 treino / 1000 teste)

| Alvo | Acurácia global | Acaso (maj.) | F1 macro | F1 ponderado |
|------|-----------------|--------------|----------|--------------|
| product_category | 0,287 | 0,211 | 0,248 | 0,270 |
| payment_method   | 0,370 | 0,313 | 0,273 | 0,339 |
| age_group        | 0,417 | 0,207 | 0,406 | 0,416 |

Os 3 alvos ficam acima do acaso. Sensibilidade/especificidade por classe (one-vs-rest) presentes em `metricas.json` — recalculadas de forma independente a partir das matrizes de confusão: **0 divergências**.

Exemplos (one-vs-rest):
- age_group `55+`: sens=0,588 / espec=0,901 (n=194)
- payment `Cartao_Credito`: sens=0,677 / espec=0,702 (n=313)
- product `Esportes`: sens=0,545 / espec=0,728 (n=211); `Brinquedos`: sens=0,100 / espec=0,986 (n=40)

## "Screenshot textual" da inferência

```
============================================================
Q3 - BLACK FRIDAY | DEMO DE INFERENCIA
============================================================
Venda NOVA apresentada ao sistema:
            gender: F
        occupation: 12
     city_category: A
        stay_years: 3
    marital_status: 1
   purchase_amount: 850.0
          quantity: 2
============================================================
PREVISOES (com grau de certeza)
============================================================
    Categoria do produto:           Beleza  (certeza = 37.3%)
      Forma de pagamento:   Cartao_Credito  (certeza = 74.0%)
            Faixa etaria:              55+  (certeza = 53.3%)

Obs.: o grau de certeza vem de max(predict_proba) de cada modelo.
============================================================
```

Venda nova → categoria + pagamento + faixa etária, cada um com % de certeza. **Funciona.**

## Artefatos confirmados (regenerados na execução)

- `outputs/matriz_confusao_{product_category,payment_method,age_group}.png` (3 PNGs)
- `outputs/metricas.json` (espec + sens por classe nos 3 alvos)
- `outputs/pipeline_{product_category,payment_method,age_group}.joblib` (3 modelos; ~280 MB no total — pesados)
- `outputs/colunas_entrada.json`
- `data/black_friday_sintetico.csv` (4000 linhas)

## Verificação de vazamento

`feature_names_in_` dos 3 pipelines = `[gender, city_category, occupation, stay_years, marital_status, purchase_amount, quantity]` — **nenhum alvo** entra como feature. OK.
