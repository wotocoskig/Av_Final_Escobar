# MANUAL DE ESTILO DEFINITIVO — ALUNO 1

Referencia unica para refinar os 3 projetos (`q1_heart_failure`, `q2_wine_quality`,
`q3_black_friday`) de modo que pareçam escritos pela MESMA pessoa, distinta do
aluno_2 (modular) e do aluno_3 (explorador/EDA).

> Este manual e descritivo do estilo que JA existe nos 3 projetos e prescritivo
> nos pontos onde ha divergencia a corrigir. Onde diz "PADRONIZAR", existe hoje
> uma inconsistencia real entre os projetos; siga a regra indicada.

---

## 1. Persona em uma frase

Engenheiro de ML **DIDATICO e METODICO**: escreve codigo que se le como uma AULA
— enxuto, reprodutivel e justificado, em que cada decisao vem acompanhada do seu
PORQUE, e a separacao "treino vs. inferencia" e a unica fronteira de modulo.

---

## 2. Estrutura de arquivos padrao (identica nos 3 projetos)

```
qN_nome/
  data/                 datasets de entrada (CSV)
  src/main.py           carrega -> pre-processa -> treina -> avalia -> salva artefatos
  src/inferencia.py     carrega artefato salvo -> monta UMA amostra nova fabricada -> predict/predict_proba
  outputs/              PNGs, *.json de metricas, *.joblib, perfis .csv/.txt
  requirements.txt
  run.ps1               fluxo completo (PowerShell)
  run.sh                fluxo completo (bash)
  README.md
```

**REGRA DE OURO — exatamente 2 modulos em `src/`.** Toda a logica vive em funcoes
pequenas dentro de `main.py` e `inferencia.py`. NUNCA criar `src/utils.py`,
`src/preprocess.py`, package com `__init__.py` ou qualquer camada extra. A
separacao "treino (main) vs. inferencia" e a UNICA fronteira de modulo permitida
— e o que distingue do aluno_2/modular.

- `main.py`: TREINA, AVALIA e PERSISTE. Termina salvando o(s) pipeline(s) e os
  metadados de colunas em `outputs/`.
- `inferencia.py`: SO CONSUME o que foi salvo. Carrega o artefato, fabrica UMA
  amostra nova realista (NUNCA copiada de uma linha do dataset), e chama
  `predict`/`predict_proba`. Sem refit, sem vazamento.

`banner()` e deliberadamente DUPLICADO nos 2 arquivos de cada projeto (a persona
e "2 arquivos autossuficientes", nao compartilha utils) — ver §6.

---

## 3. Convencao de nomes (bilingue: dominio em PT, padrao tecnico mantido)

- **Variaveis e funcoes:** `snake_case` em PORTUGUES sem acento. Funcoes sao
  verbos: `carregar_dados`, `construir_preprocessador`, `construir_pipeline`,
  `escolher_melhor_k`, `treinar_e_avaliar_alvo`, `avaliar_modelo`,
  `plotar_matriz_confusao`, `gerar_perfil_clusters`, `prever_grupo`,
  `prever_alvo`. Variaveis de dominio em PT: `caminho`, `pasta_saida`, `rotulos`,
  `metricas`, `vencedor`, `dominio`, `preprocessador`, `escalador`, `melhor_k`,
  `perfil`.
- **Mantidos em forma tecnica/inglesa** (convencao consagrada — NUNCA traduzir):
  `X`, `y`, `X_treino`, `X_teste`, `y_treino`, `y_teste`, `y_pred`/`y_predito`,
  `pipeline`, `modelo`, `scaler`, `classes`. Misturar PT e EN na mesma linha e
  PROPOSITAL e e assinatura da persona.
- **PROIBIDO:** `df1`/`df2`/`tmp`/`aux`/`data2`. DataFrames recebem nome de
  dominio (`dados`, `df`, `tinto`, `branco`, `df_paciente`, `df_venda`).
- **CONSTANTES** em `UPPER_SNAKE`, declaradas no topo do modulo logo apos os
  imports, agrupadas sob um banner de comentario com regua `# ---...---` e o
  titulo "Definicao das colunas ..." / "Configuracoes globais":
  `RANDOM_STATE = 42` (SEMPRE 42, com comentario "Reprodutibilidade
  obrigatoria"), `COLUNA_ALVO`, `COLUNAS_CONTINUAS`, `COLUNAS_BINARIAS`,
  `COLUNAS_CATEGORICAS`, `COLUNAS_NUMERICAS`, `COLUNAS_FEATURES`, `ALVOS`,
  `DOMINIO_*`, `N_LINHAS_SINTETICO`, e os caminhos `DIR_PROJETO`/`DIR_DADOS`/
  `DIR_SAIDA` (ver §3.1).
- **Arquivos de saida** em PT, descritivos: `pipeline_kmeans.joblib`,
  `modelo_vencedor.joblib`, `pipeline_<alvo>.joblib`, `perfil_clusters.csv/.txt`,
  `matriz_confusao_<nome>.png`, `metricas.json`, `comparacao_modelos.json`,
  `colunas_entrada.json`, `silhouette_por_k.png`, `clusters_pca.png`.

### 3.1 Caminhos — PADRONIZAR (constantes tipadas no topo)

Adotar em TODOS os modulos o idioma da Q2: constantes de modulo tipadas,
declaradas UMA vez sob as constantes, e nunca recalculadas dentro das funcoes.

```python
DIR_PROJETO: Path = Path(__file__).resolve().parents[1]
DIR_DADOS: Path = DIR_PROJETO / "data"
DIR_SAIDA: Path = DIR_PROJETO / "outputs"
```

> Corrigir Q1/Q3, que hoje recalculam `Path(__file__).resolve().parents[1]` dentro
> de `carregar_dados`, `main`, etc. Em `inferencia.py` o equivalente e
> `PASTA_SAIDA`/`CAMINHO_MODELO` — usar o mesmo nome `DIR_SAIDA` para uniformidade.

### 3.2 joblib — PADRONIZAR (um unico idioma)

Adotar `from joblib import dump, load` (estilo Q1/Q3) em TODOS os arquivos.
ABANDONAR o `import joblib; joblib.dump(...)` da Q2. Manter `compress=3` ao salvar
modelos de arvore, com o comentario que explica o porque (ver §4).

---

## 4. Type hints e organizacao interna de cada modulo

**Type hints obrigatorios, 100% de cobertura.**

- Primeira linha de codigo de TODO modulo (apos a docstring): `from __future__
  import annotations`.
- TODA funcao anota parametros e retorno, inclusive helpers:
  `def banner(titulo: str) -> None:`. Sintaxe nativa de generics:
  `dict[int, float]`, `list[str]`, `tuple[int, dict[int, float]]`,
  `dict[str, Pipeline]`. Anotar tambem as constantes-caminho: `DIR_PROJETO: Path = ...`.

**Ordem canonica de cada modulo:**

1. Docstring de modulo extensa (§5.1) terminando na linha-assinatura.
2. `from __future__ import annotations`.
3. Imports: stdlib (`json`, `pathlib.Path`) -> `import matplotlib` ->
   `matplotlib.use("Agg")` com o comentario fixo -> demais terceiros
   (`matplotlib.pyplot as plt`, `numpy as np`, `pandas as pd`, `seaborn as sns`
   so onde ha heatmap) -> imports sklearn agrupados (cluster/compose/ensemble/
   metrics/model_selection/pipeline/preprocessing).
4. Constantes `UPPER_SNAKE` sob banner de comentario (inclui `RANDOM_STATE = 42`
   e os `DIR_*`).
5. `def banner(...)` -> funcoes auxiliares -> funcoes de etapa
   (carregar/construir/treinar/avaliar/plotar/salvar) -> `def main() -> None`.
6. `if __name__ == "__main__": main()`.

**Narrativa do `main()`:** fases delimitadas por reguas de comentario
`# ---...---` com cabecalhos numerados (`# 1) Carregamento ...`, `# 2) ...`) E por
`banner("TITULO EM CAIXA ALTA")` a cada fase.

**I/O:** sempre `pathlib`; toda leitura/escrita de texto com `encoding="utf-8"`;
todo `json.dump` com `indent=2, ensure_ascii=False`.

---

## 5. Voz dos comentarios e docstrings (a marca central)

**REGRA DE OURO: todo comentario explica o PORQUE, nunca o O-QUE.** O codigo ja
diz o que. Quando a decisao for polemica, o comentario cita o EFEITO COLATERAL
evitado.

- PROIBIDO: `# aplica StandardScaler`
- ASSINATURA: `# StandardScaler nas continuas para que nenhuma variavel domine a
  distancia euclidiana so por causa da sua unidade/escala.`
- Outros exemplos reais a preservar como tom: "escalar inflaria o peso de
  categorias raras", "evita erro silencioso", "compress=3 ... sem alterar as
  previsoes nem as metricas", "stratify ... mantem a proporcao das classes raras".

**Refrao anti-vazamento (leitmotiv consciente — manter SEMPRE):** marcar as
fronteiras de vazamento em prosa curta ancorada ao codigo: `(sem vazamento)`,
"fit so no treino", "nada e re-ajustado", "os alvos NUNCA entram como entrada".

**Docstring de funcao:** 1 linha imperativa-descritiva ("Imprime um banner...",
"Carrega e une..."). So vira multi-linha quando ha decisao NAO-obvia a justificar
(ver `escolher_melhor_k`, `metricas_por_classe`, `gerar_dataset_sintetico`).

### 5.1 Docstring de modulo — formato e PADRONIZACAO

Abre com `Q{n} - {Dataset}: {TAREFA em maiusculas}` + 1 linha de objetivo. Sempre
que houver armadilha conceitual, inclui um bloco em prosa que ATACA a armadilha
antes de qualquer codigo: "IMPORTANTE: isto NAO e classificacao", "Sem
vazamento:". Esse aviso conceitual no topo e assinatura — replicar SEMPRE.

- PADRONIZAR a regua: a Q2 usa `===` sob o titulo (estilo banner); Q1/Q3 nao.
  Adotar a regua `=`*49 sob o titulo do modulo nos 3, para a docstring "ecoar" o
  banner de runtime.
- PADRONIZAR a linha-assinatura: fechar TODA docstring de modulo com
  `Aluno 1 - estilo metodico e didatico.` (Q1/Q3 ja fazem; Q2 precisa adotar).
  Usar travessao simples `-` (nao `—`) para consistencia ASCII.

---

## 6. Estilo de prints e banners (a coreografia da saida)

```python
def banner(titulo: str) -> None:
    """Imprime um banner de secao padronizado."""
    print("=" * 60)
    print(titulo)
    print("=" * 60)
```

- `banner()` **byte-a-byte identico nos 6 arquivos** (duplicado de proposito; sem
  utils compartilhado). Largura **60** e a regua da casa — NUNCA inventar outra.
- Titulos SEMPRE em CAIXA ALTA, numerados pelo fluxo: `PRE-PROCESSAMENTO`,
  `ESCOLHA DO NUMERO DE CLUSTERS (k)`, `TREINO DO MODELO FINAL`, `AVALIACAO ...`,
  `SELECAO DO MODELO VENCEDOR (F1-macro)`, `PERSISTENCIA DO MODELO`. Banner final
  SEMPRE `CONCLUIDO`.
- Banner de abertura no formato `Q{n} - {DATASET} | {TAREFA}`
  (ex.: `Q1 - HEART FAILURE | CLUSTERING COM KMeans`,
  `Q3 - BLACK FRIDAY | MULTI-CLASSIFICACAO (3 ALVOS)`).
- **Metrica SEMPRE impressa com a direcao/baseline anexada ao numero**, nunca o
  numero pelado: `Silhouette : 0.1335 (quanto MAIOR, melhor)`,
  `Davies-Bouldin : 2.5640 (quanto MENOR, melhor)`,
  `Acuracia GLOBAL : 0.287 (acaso ~ 0.211)`.
- **Alinhamento de colunas com f-string** para tabelas no terminal
  (`:>26`, `:<18`, `:>22`, `:>14`, `:.4f`, `:.3f`) — manter como habito.
- **Aviso de risco gritado** com `print("!" * 60)` em bloco (ver
  `aviso_placeholder` na Q3). Reservar `!` EXCLUSIVAMENTE para resultados que NAO
  valem (placeholder/invalidos); nunca como enfeite. E a "voz de alerta" da
  persona.
- **Gancho pedagogico de fim:** a ultima linha do `main` orienta o proximo passo:
  `print("Execute agora: python src/inferencia.py (demo de inferencia).")`.

---

## 7. Estilo de visualizacao / EDA

**A visualizacao aqui e MINIMA, FINALISTICA e EXPLICADA** — gera SO o grafico que
comprova UMA decisao. NAO fazer EDA exploratoria ampla (histogramas de tudo,
correlacao, pairplots) — isso e territorio do aluno_3 e descaracteriza a persona.

- Cada projeto gera apenas o(s) grafico(s) que sustenta(m) uma decisao:
  `silhouette_por_k.png` (justifica o k), `clusters_pca.png` (so para "ver" os
  grupos), `matriz_confusao_*.png` (acerto por classe).
- **O titulo do grafico carrega a ressalva metodologica** como legenda didatica:
  "(PCA, apenas visualizacao)", "(acuracia global = 0.287)".
- **Eixos sempre rotulados em portugues com unidade/sentido:**
  "Componente 1 (X% da variancia)", "Classe PREVISTA", "Classe VERDADEIRA",
  "k (numero de clusters)", "Silhouette medio".
- `matplotlib.use("Agg")` SEMPRE antes do `pyplot`, com o comentario fixo
  `# backend nao-interativo: precisa vir ANTES do pyplot` (didatico sobre uma
  pegadinha real). `fig.savefig(..., dpi=120)` e `plt.close(fig)` ao final.
- Paleta **"Blues"** para matriz de confusao — ja consistente, manter.
- **PADRONIZAR a renderizacao da matriz de confusao:** Q3 usa `seaborn.heatmap`,
  Q2 desenha "na mao" com `imshow` + anotacao. Adotar `seaborn.heatmap`
  (`annot=True, fmt="d", cmap="Blues"`) nos dois, e listar `seaborn` no
  `requirements.txt` de Q2 (alem de Q3). Assim a "assinatura visual" fica uniforme.

---

## 8. requirements.txt e run scripts

### 8.1 requirements.txt — PADRONIZAR pin exato

Uma lib por linha, ordem fixa, **pin exato `==`** com as versoes do ambiente de
avaliacao (a Q2 ja faz; Q1/Q3 usam `>=` e devem migrar). `seaborn` em TODOS,
porque a matriz de confusao passa a usar `seaborn` em todo projeto (§7).

```
scikit-learn==1.8.0
pandas==3.0.2
numpy==2.4.4
scipy==1.17.1
matplotlib==3.11.0
seaborn==0.13.2
joblib==1.5.3
```

### 8.2 run.ps1 + run.sh — PADRONIZAR formato robusto (Q1/Q3)

Sempre o PAR `run.ps1` + `run.sh`, executando `main.py` depois `inferencia.py`,
com mensagem final "Veja a pasta outputs/.". Adotar em TODOS o formato da Q1/Q3
(a Q2 deve abandonar os banners `===` e adotar guardas + numeracao):

`run.ps1`:
```powershell
# run.ps1 - Executa o fluxo COMPLETO da QN (Windows PowerShell).
# Treino -> outputs -> demo de inferencia.

$ErrorActionPreference = "Stop"

# Garante execucao a partir da pasta do projeto (onde esta este script).
Set-Location -Path $PSScriptRoot

Write-Host "==> [1/2] Treinando o modelo (src/main.py)..."
python src/main.py
if ($LASTEXITCODE -ne 0) { throw "Falha no treino (main.py)." }

Write-Host "==> [2/2] Rodando a demo de inferencia (src/inferencia.py)..."
python src/inferencia.py
if ($LASTEXITCODE -ne 0) { throw "Falha na inferencia (inferencia.py)." }

Write-Host "==> Fluxo concluido. Veja a pasta outputs/."
```

`run.sh`:
```bash
#!/usr/bin/env bash
# run.sh - Executa o fluxo COMPLETO da QN (bash).
# Treino -> outputs -> demo de inferencia.

set -euo pipefail

# Garante execucao a partir da pasta do projeto (onde esta este script).
cd "$(dirname "$0")"

echo "==> [1/2] Treinando o modelo (src/main.py)..."
python src/main.py

echo "==> [2/2] Rodando a demo de inferencia (src/inferencia.py)..."
python src/inferencia.py

echo "==> Fluxo concluido. Veja a pasta outputs/."
```

> Regras nao-negociaveis dos scripts: `set -euo pipefail` (bash) e
> `$ErrorActionPreference = "Stop"` + guardas `if ($LASTEXITCODE -ne 0) { throw ... }`
> apos cada chamada (PowerShell); `cd "$(dirname "$0")"` / `Set-Location $PSScriptRoot`;
> passos rotulados `==> [1/2]` / `[2/2]`.

---

## 9. Tom do README (tom de aula escrita) — PADRONIZAR pelo molde da Q1

- Abre com `# Q{n} — {Dataset}: {Tarefa}` e, LOGO ABAIXO do H1, a linha-assinatura
  `Aluno 1 — estilo metodico e didatico.` (hoje so a Q1 tem; Q2/Q3 adotam).
- `> blockquote` no topo para o aviso conceitual de maior risco de mal-entendido
  ("Isto NAO e classificacao"). Um por README.
- Secao obrigatoria **"decisoes justificadas"** em TABELA
  `| Decisao | Escolha | Por que |` (hoje so a Q1). A coluna "Por que" e a alma
  didatica da persona — promover a padrao dos 3.
- Secao **"Sem vazamento de dados"** em PROSA explicativa (nao bullet seco),
  capitulo recorrente.
- Secao **"O que cada saida significa"** mapeando cada arquivo de `outputs/` ao
  seu sentido (hoje so a Q1 tem a tabela completa) — replicar nos 3.
- Secao **"Leitura clinica" / "Leitura dos numeros"**: SEMPRE traduzir a metrica
  para linguagem de dominio ("grupo de maior risco") com HONESTIDADE sobre
  limitacoes ("limitacao dos dados, nao do algoritmo"; admitir classes em
  0,0000). Honestidade metodologica e parte do tom.
- **Numeros em formato pt-BR nas TABELAS de resultado** (virgula decimal:
  "0,6923"). PADRONIZAR: Q1 usa ponto, Q2 usa virgula — adotar virgula nas tabelas.

---

## 10. O que evitar (guarda-corpos da persona)

- Sem girias, sem emojis, sem exclamacao decorativa (so em `aviso_placeholder`).
- Sem comentarios redundantes que repetem o codigo (§5).
- Sem jargao sem glosa: todo termo tecnico ganha uma traducao curta na primeira
  aparicao.
- Sem EDA exploratoria difusa (territorio do aluno_3).
- Sem arquitetura modular / utils compartilhado / package (territorio do aluno_2):
  manter os 2 arquivos autossuficientes com `banner()` duplicado de proposito.

---

## 11. MARCAS REGISTRADAS (idiomas-assinatura) — checklist

1. **Exatamente 2 modulos em `src/`** (`main.py` treina+persiste, `inferencia.py`
   so consome) — unica fronteira de modulo, sem package/utils/camadas extras.
2. **`from __future__ import annotations` como 1a linha de codigo** + type hints
   nativos (`dict[int,float]`, `tuple[...]`) em 100% das funcoes, inclusive helpers.
3. **Nomes bilingues deliberados:** dominio em PT sem acento (`carregar_dados`,
   `preprocessador`, `rotulos`, `pasta_saida`, `melhor_k`) convivendo com
   convencoes tecnicas intocadas (`X_treino`, `y_teste`, `pipeline`, `modelo`).
4. **`RANDOM_STATE = 42`** e blocos de constantes `UPPER_SNAKE` (`COLUNAS_*`,
   `DOMINIO_*`, `ALVOS`, `DIR_*`) no topo, sob banner de comentario explicando a
   decisao.
5. **`def banner(titulo: str) -> None` com `"="*60` replicado byte-a-byte nos 6
   arquivos**; `main()` narrado em fases numeradas com reguas `# ---...---` e
   banner final SEMPRE `CONCLUIDO`.
6. **Comentario explica o PORQUE e o efeito colateral evitado, nunca o O-QUE**
   ("escalar inflaria o peso de categorias raras", "evita erro silencioso").
7. **Metrica sempre impressa com a direcao/baseline anexada**
   ("(quanto MAIOR, melhor)", "(acaso ~ 0.211)").
8. **Refrao anti-vazamento como leitmotiv** ("(sem vazamento)", "fit so no
   treino", "nada e re-ajustado", "os alvos NUNCA entram como entrada").
9. **Aviso conceitual no topo da docstring de modulo** (bloco "IMPORTANTE:" /
   "Sem vazamento:") fechado pela linha-assinatura
   `Aluno 1 - estilo metodico e didatico.`
10. **`matplotlib.use("Agg")` antes do `pyplot`** com o comentario fixo "precisa
    vir ANTES do pyplot"; visualizacao minima e finalistica com ressalva
    metodologica no titulo do grafico ("PCA, apenas visualizacao"), nunca EDA ampla.
11. **Aviso de risco gritado em bloco `"!"*60`** (`aviso_placeholder`), reservado
    so para resultados que nao valem; e **gancho pedagogico de fim** ("Execute
    agora: python src/inferencia.py").
12. **I/O 100% `pathlib`** via constantes `DIR_PROJETO`/`DIR_DADOS`/`DIR_SAIDA`
    tipadas, `encoding="utf-8"`, `json.dump(indent=2, ensure_ascii=False)`; joblib
    sempre via `from joblib import dump, load`.
13. **Par `run.ps1` + `run.sh`** (main -> inferencia) com `set -euo pipefail`,
    guardas `$LASTEXITCODE`/`throw` e passos `==> [1/2]`/`[2/2]`; `requirements.txt`
    de uma lib por linha em ordem fixa com pin exato `==`.
14. **README em tom de aula:** linha-assinatura sob o H1, blockquote conceitual,
    tabela `| Decisao | Escolha | Por que |`, tabela "O que cada saida significa",
    "Leitura clinica/dos numeros" com honestidade sobre limitacoes, numeros pt-BR
    (virgula) nas tabelas de resultado.
