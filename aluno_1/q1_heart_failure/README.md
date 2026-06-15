# Q1 — Heart Failure: Sistema de Agrupamento (Clustering)

Aluno 1 — estilo metodico e didatico.

## Objetivo

Construir um sistema de **agrupamento nao supervisionado** (clustering) que
organiza pacientes com insuficiencia cardiaca em grupos clinicamente similares.
Dado um **paciente novo e desconhecido**, o sistema indica a qual grupo ele e
mais parecido.

> Isto **NAO e classificacao**. A coluna `DEATH_EVENT` (obito) **nunca** entra
> no treino. Ela e usada **apenas depois**, para *caracterizar* os grupos
> encontrados, calculando a taxa de obito de cada cluster.

Dataset: `data/heart_failure_clinical_records_dataset.csv` — 299 pacientes,
13 colunas, sem valores ausentes.

## Algoritmo e decisoes de projeto (justificadas)

| Decisao | Escolha | Por que |
|---|---|---|
| Algoritmo | **KMeans** | Particiona por distancia euclidiana, simples e direto; permite atribuir um paciente novo ao centroide mais proximo via `predict`. |
| Escolha de k | **Maior silhouette**, varrendo k=2..8 | Criterio objetivo, unico e reprodutivel, sem a ambiguidade do "cotovelo". Grafico em `silhouette_por_k.png`. |
| Scaler (continuas) | **StandardScaler** | Equaliza a dispersao das variaveis; e o scaler natural para distancia euclidiana. |
| Binarias | **Passthrough** (NAO escaladas) | Numa binaria 0/1 a distancia entre categorias ja vale 1, estavel e interpretavel; escalar inflaria o peso de categorias raras. |
| Pre-processamento | **ColumnTransformer** | StandardScaler nas continuas + passthrough nas binarias, num so passo, dentro do Pipeline. |
| `time` | **Mantida** como continua escalada | E o tempo de acompanhamento (dias), uma variavel clinica legitima; o projeto exige inclui-la. Ressalva: relaciona-se com o desfecho, por isso a taxa de obito por cluster e lida como caracterizacao descritiva. |
| `DEATH_EVENT` | **Fora do treino** | Usada so na caracterizacao (taxa de obito por grupo). |

Discussao completa das decisoes polemicas em
[`outputs/debate_Q1.md`](outputs/debate_Q1.md).

### Colunas

- **Continuas (StandardScaler):** `age`, `creatinine_phosphokinase`,
  `ejection_fraction`, `platelets`, `serum_creatinine`, `serum_sodium`, `time`.
- **Binarias (passthrough):** `anaemia`, `diabetes`, `high_blood_pressure`,
  `sex`, `smoking`.
- **Alvo (fora do treino):** `DEATH_EVENT`.

### Sem vazamento de dados

Por se tratar de **agrupamento não supervisionado**, não há divisão
treino/teste: aqui "treino" significa a **coorte inteira** de 299 pacientes, e
o modelo é ajustado sobre ela. A fronteira de vazamento que de fato importa não
é treino/teste, e sim **coorte conhecida → paciente novo**: o `StandardScaler`
é ajustado (`fit`) uma única vez sobre os 299 pacientes, dentro do Pipeline, e o
paciente novo é **somente transformado** (sem re-ajuste) e atribuído a um grupo
com `predict` — ou seja, nenhuma estatística do paciente novo influencia o
pré-processamento. Além disso, `DEATH_EVENT` **nunca** entra como variável de
entrada; é usado apenas depois, para caracterizar os grupos. `random_state=42`
em todo o projeto garante reprodutibilidade.

## Como rodar

Requer Python 3 com as libs de `requirements.txt` (ja instaladas globalmente
no ambiente de avaliacao).

```powershell
# Windows / PowerShell
./run.ps1
```

```bash
# bash
bash run.sh
```

Ou manualmente:

```bash
python src/main.py        # treina, avalia, salva artefatos
python src/inferencia.py  # demo: paciente novo -> grupo
```

## O que cada saida significa (pasta `outputs/`)

| Arquivo | Significado |
|---|---|
| `silhouette_por_k.png` | Silhouette medio para k=2..8; destaca o k escolhido e anota seu valor. |
| `clusters_pca.png` | Dispersao 2D (PCA, **so visualizacao**) colorida por cluster, com os centroides marcados (`X`). |
| `taxa_obito_por_cluster.png` | Barras da taxa de obito por cluster (caracterizacao; `DEATH_EVENT` fora do treino). |
| `perfil_clusters.csv` / `.txt` | Medias de cada variavel por cluster + taxa de `DEATH_EVENT` + n de pacientes. |
| `metricas.json` | k escolhido, silhouette, davies_bouldin, calinski_harabasz, n por cluster, taxa de obito por cluster e bloco `clusters` com rotulo de risco. |
| `pipeline_kmeans.joblib` | Pipeline treinado (pre-processamento + KMeans) para a inferencia. |
| `colunas_entrada.json` | Ordem das colunas de entrada usada no treino. |
| `debate_Q1.md` | Debate das decisoes polemicas + decisao final. |

> Nota: a pasta `outputs/` tambem recebe arquivos de auditoria externos
> (`AUDITORIA.md`, `TESTE.md`), gerados pelo avaliador — nao sao saidas do
> `main.py` e por isso nao constam na tabela acima.

## Resultados reais (execucao com `random_state=42`)

- **k escolhido:** 2 (maior silhouette entre k=2..8).
- **Silhouette:** 0.1335 (maior = melhor).
- **Davies-Bouldin:** 2.5640 (menor = melhor).
- **Calinski-Harabasz:** 37.96 (maior = melhor).

Silhouette por k:

| k | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| silhouette | **0,1335** | 0,1235 | 0,1131 | 0,1310 | 0,1247 | 0,1185 | 0,1265 |

### Perfil dos grupos (caracterizacao com `DEATH_EVENT`)

| Cluster | n pacientes | Taxa de obito | Idade media | Ejection fraction | Serum creatinine | Time (dias) |
|---|---|---|---|---|---|---|
| 0 | 187 | **13,9%** | 55,9 | 39,3 | 1,06 | 159,0 |
| 1 | 112 | **62,5%** | 69,1 | 36,1 | 1,94 | 82,3 |

**Leitura clinica:** mesmo sem nunca ver o desfecho, o KMeans separou um grupo
de **maior risco** (Cluster 1: pacientes mais velhos, com creatinina serica
mais alta e menor tempo de acompanhamento, taxa de obito de 62,5%) de um grupo
de **menor risco** (Cluster 0, taxa de obito de 13,9%). Ressalva honesta: parte
dessa separacao por obito vem da inclusao de `time` (tempo de acompanhamento),
que e um **proxy do desfecho** — pacientes que morrem tendem a ter `time` menor
(ver `outputs/debate_Q1.md`, Tema 4). Por isso a diferenca `13,9% vs 62,5%`
deve ser lida como **caracterizacao descritiva** dos grupos, nunca como prova
de poder preditivo do agrupamento.

### Demo de inferencia

O paciente novo de exemplo (60 anos, ejection_fraction 35, serum_creatinine
1.2, time 115 dias, etc.) foi atribuido ao **Cluster 0** (grupo de menor risco,
taxa de obito de 13.9%), coerente com seu perfil clinico relativamente
favoravel.

## Estrutura

```
q1_heart_failure/
  data/    heart_failure_clinical_records_dataset.csv
  src/     main.py (treino+avaliacao+salvar) | inferencia.py (paciente novo -> grupo)
  outputs/ PNGs, perfis, metricas.json, pipeline .joblib, debate_Q1.md
  requirements.txt | run.ps1 | run.sh | README.md
```
