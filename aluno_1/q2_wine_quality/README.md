# Q2 - Wine Quality (Classificacao supervisionada)

Aluno 1 — estilo metodico e didatico.

> Isto e classificacao **MULTICLASSE FINA** da nota inteira (3 a 9), NAO
> regressao nem agrupamento em faixas; o alvo `quality` **NUNCA** entra como
> feature de entrada. As classes 3 e 9 sao quase inaprendiveis — limitacao
> **DOS DADOS** (30 e 5 amostras), nao do algoritmo.

Previsao da qualidade (`quality`, nota inteira de 3 a 9) de vinhos a partir de
suas caracteristicas fisico-quimicas, unindo os conjuntos de **vinho tinto** e
**vinho branco** em um unico problema de classificacao multiclasse.

## Objetivo

- Unir tinto + branco preservando os nomes das colunas e adicionar a feature
  `tipo` (tinto=0 / branco=1).
- Avaliar **3 classificadores** (RandomForest, GradientBoosting, KNN), cada um
  em um `Pipeline` com `StandardScaler`.
- Selecionar o vencedor por **F1-macro** (desempate por acuracia global).
- Reportar acuracia global, acuracia por classe (matriz de confusao), F1-macro
  e F1-weighted.
- Disponibilizar um modulo de **inferencia** para um vinho novo.

## Estrutura

```
q2_wine_quality/
├── data/
│   ├── winequality-red.csv      # vinho tinto  (sep=';')
│   └── winequality-white.csv    # vinho branco (sep=';')
├── src/
│   ├── main.py                  # treina, avalia, seleciona e salva
│   └── inferencia.py            # carrega o vencedor e preve vinho novo
├── outputs/
│   ├── modelo_vencedor.joblib       # pipeline vencedor (fit so no treino)
│   ├── metricas.json                # metricas do vencedor
│   ├── comparacao_modelos.json      # metricas dos 3 modelos
│   ├── matriz_confusao_RandomForest.png
│   └── debate_Q2.md                 # decisoes de modelagem
├── requirements.txt
├── run.sh                       # fluxo completo (bash)
├── run.ps1                      # fluxo completo (PowerShell)
└── README.md
```

## Como executar

PowerShell (Windows):

```powershell
./run.ps1
```

Bash:

```bash
./run.sh
```

Ou manualmente:

```bash
python src/main.py        # treina, avalia e salva os artefatos
python src/inferencia.py  # preve a quality de um vinho novo
```

## Dados

| Conjunto      | Amostras |
|---------------|----------|
| Vinho tinto   | 1.599    |
| Vinho branco  | 4.898    |
| **Total**     | **6.497** |

Distribuicao do alvo `quality` (classes **muito desbalanceadas**):

| quality | 3  | 4   | 5    | 6    | 7    | 8   | 9 |
|---------|----|-----|------|------|------|-----|---|
| amostras| 30 | 216 | 2138 | 2836 | 1079 | 193 | 5 |

As classes 5 e 6 dominam (~76% dos dados); 3 e 9 sao raríssimas.

## Decisoes justificadas

| Decisao | Escolha | Por que |
|---|---|---|
| Tarefa | **Classificacao multiclasse (7 classes)** | O enunciado pede a nota inteira 3..9, com acuracia por classe — nao faixas nem regressao. |
| Modelos | **RandomForest / GradientBoosting / KNN** em `Pipeline` | Tres familias distintas sob a mesma condicao de pre-processamento, comparadas de forma justa. |
| Scaler | **StandardScaler** em todos | Indispensavel ao KNN (distancia), inofensivo as arvores; mantem o fluxo uniforme e sem vazamento (fit so no treino). |
| Desbalanceamento | **`class_weight="balanced"`** (RF) | Penaliza erros nas classes minoritarias sem criar dados sinteticos. |
| Criterio de selecao | **F1-macro** (desempate por acuracia) | Trata todas as classes igualmente, impede que 5 e 6 mascarem as raras. |
| Feature `tipo` | **tinto=0 / branco=1** | Separa os dois perfis fisico-quimicos distintos (branco costuma ter mais `residual sugar`). |
| Split | **Estratificado 80/20** (`random_state=42`) | 5.197 treino / 1.300 teste; a estratificacao preserva a proporcao das classes raras (3 e 9). |

### Sobre as linhas duplicadas (limitacao conhecida)

O conjunto unido tem **1.177 linhas fisico-quimicas totalmente duplicadas**
(6.497 -> 5.320 distintas). Como nao removemos essas duplicatas, ~27% das linhas
de teste possuem um vetor de features identico tambem no treino, o que torna as
metricas reportadas **otimistas por memorizacao** (um vazamento sutil). Mantemos
o comportamento de referencia desta entrega e registramos a limitacao
explicitamente: deduplicar antes do split (a correcao tecnica) reduziria a
acuracia global para ~0,56 e o F1-macro para ~0,26, numeros mais honestos sobre
a generalizacao real. O fit do `StandardScaler` continua ocorrendo **somente no
treino**, dentro do Pipeline.

## Resultados (reais)

Comparacao dos 3 modelos no conjunto de teste:

| Modelo            | Acuracia global | F1-macro   | F1-weighted |
|-------------------|-----------------|------------|-------------|
| **RandomForest**  | **0,6923**      | **0,3959** | **0,6792**  |
| KNN               | 0,6592          | 0,3713     | 0,6476      |
| GradientBoosting  | 0,5838          | 0,3122     | 0,5706      |

**Vencedor: RandomForest** (maior F1-macro e tambem maior acuracia global).

### Metricas detalhadas do vencedor (RandomForest)

- Acuracia global: **0,6923**
- F1-macro: **0,3959**
- F1-weighted: **0,6792**

Acuracia por classe (diagonal da matriz de confusao):

| quality | 3      | 4      | 5      | 6      | 7      | 8      | 9      |
|---------|--------|--------|--------|--------|--------|--------|--------|
| acuracia| 0,0000 | 0,1163 | 0,7103 | 0,8060 | 0,5602 | 0,3333 | 0,0000 |

Matriz de confusao: `outputs/matriz_confusao_RandomForest.png`.

### Leitura dos numeros

- O modelo acerta muito bem as classes dominantes (5 e 6) e razoavelmente as
  intermediarias (7). As classes **3 e 9 ficam em 0,0000**: com apenas 30 e 5
  amostras no total (1 da classe 9 no teste), e estatisticamente inviavel
  aprende-las — uma limitacao **dos dados**, nao do algoritmo.
- Por isso a acuracia global (0,69) e bem maior que o F1-macro (0,40): o
  F1-macro penaliza explicitamente o fracasso nas classes raras, sendo o
  criterio mais honesto para escolher o modelo neste cenario desbalanceado.
- A feature `tipo` (tinto/branco) entra no pipeline e ajuda o modelo a separar
  os dois perfis de vinho, que tem distribuicoes fisico-quimicas distintas
  (ex.: branco costuma ter mais `residual sugar` e `total sulfur dioxide`).

## Sem vazamento de dados

O `StandardScaler` vive **dentro** do `Pipeline`, e seu `fit` ocorre **somente
no treino** (`pipeline.fit(X_treino, ...)`). A inferencia apenas carrega o
pipeline ja ajustado e chama `predict`/`predict_proba` — **nada e re-ajustado**,
nenhuma estatistica do vinho novo influencia o pre-processamento. O alvo
`quality` **nunca** entra como entrada, e o vinho da demo e **fabricado** (nao
copiado de nenhuma linha dos CSVs). A unica fonte de otimismo conhecida sao as
1.177 duplicatas exatas descritas acima — limitacao **dos dados**, documentada e
nao silenciosa. `random_state=42` em todo split/modelo garante reprodutibilidade.

## O que cada saida significa (pasta `outputs/`)

| Arquivo | Significado |
|---|---|
| `modelo_vencedor.joblib` | Pipeline vencedor + colunas + classes (fit so no treino, `compress=3`). |
| `metricas.json` | Metricas do vencedor (acuracia global, F1-macro/weighted, acuracia por classe) + baseline de acaso. |
| `comparacao_modelos.json` | Metricas dos 3 modelos avaliados, lado a lado. |
| `matriz_confusao_RandomForest.png` | Acerto por classe do vencedor (com a ressalva metodologica no titulo). |
| `debate_Q2.md` | Debate das decisoes de modelagem (classes raras, F1-macro, faixas). |

> Os artefatos de `outputs/` sao **regenerados a cada `run`** (`run.ps1` /
> `run.sh`); nao precisam ser versionados.

## Inferencia

`src/inferencia.py` carrega `outputs/modelo_vencedor.joblib` (pipeline ja
ajustado **somente no treino**, sem vazamento) e preve a `quality` de um vinho
novo junto com a probabilidade da classe prevista.

**Contrato de entrada:** o vinho informado deve trazer exatamente as **12
features** na ordem do treino, **incluindo `tipo`** (tinto=0 / branco=1).
Colunas faltantes, nomes errados (ex.: `PH` em vez de `pH`) ou um `tipo` fora de
{0, 1} agora geram um **erro claro** (`ValueError`) — em vez de uma previsao
silenciosa sobre dados incompletos.

Exemplo executado (vinho branco de demonstracao, com valores **fabricados** que
nao existem em nenhuma linha dos CSVs — uma amostra genuinamente nova):

```
Quality previsto : 6
Probabilidade    : 0.5900 (59.00%)
```

## Decisoes de modelagem

Ver `outputs/debate_Q2.md` para o debate completo sobre: tratamento das classes
raras (3 e 9), escolha do F1-macro vs micro/weighted, e a possibilidade de
agrupar a `quality` em faixas.

## Ambiente

Python 3.14 — dependencias em `requirements.txt`. `random_state=42` em todo
split/modelo.
