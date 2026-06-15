# Q3 — Black Friday: Multi-classificação (3 alvos independentes)

Aluno 1 — estilo metodico e didatico.

> **Atenção:** os três alvos são previstos por **modelos independentes** — este
> projeto assume que `product_category`, `payment_method` e `age_group` são
> **condicionalmente independentes** dadas as features; não há cadeia entre eles.

Projeto de Machine Learning que, para **uma venda da Black Friday**, prevê
**três alvos ao mesmo tempo**, cada um com seu próprio **grau de certeza**:

1. `product_category` — categoria do produto;
2. `payment_method` — forma de pagamento;
3. `age_group` — faixa etária do comprador.

A estratégia de modelagem deste aluno é **três `RandomForestClassifier`
independentes** (um por alvo). Cada modelo é um `Pipeline` com `ColumnTransformer`
(one-hot nas categóricas, `StandardScaler` nas numéricas) e split **estratificado**
(`random_state=42`). O grau de certeza de cada previsão é o `max(predict_proba)`
do respectivo modelo.

---

## AVISO IMPORTANTE — RESULTADOS COM DADOS SINTÉTICOS (PLACEHOLDER)

```
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!  ATENCAO: DADOS SINTETICOS (PLACEHOLDER)             !!!
!!!  O arquivo data/black_friday.csv NAO foi encontrado. !!!
!!!  Foi gerado um dataset SINTETICO (seed 42) APENAS    !!!
!!!  para provar que o pipeline roda de ponta a ponta.   !!!
!!!  Os numeros NAO valem como resultado final / real.   !!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

Como **não havia** o arquivo `data/black_friday.csv`, o `src/main.py` **gerou um
dataset sintético** de ~4000 linhas (`numpy` com seed 42) e o salvou em
`data/black_friday_sintetico.csv`. Esse dataset foi construído com **dependências
probabilísticas features→alvos** de propósito, para que os três modelos fiquem
**acima do acaso** e o pipeline possa ser demonstrado de ponta a ponta.

**Os números abaixo são PLACEHOLDERS.** Para obter resultados reais, basta colocar
o CSV verdadeiro em `data/black_friday.csv` (com as mesmas colunas do esquema) e
rodar de novo: o código passa a usá-lo automaticamente, sem gerar o sintético.

---

## Esquema dos dados

**Features de entrada** (as mesmas para os 3 modelos; nenhum alvo entra como feature):

| coluna            | tipo        | domínio                         |
|-------------------|-------------|---------------------------------|
| `gender`          | categórica  | `M` / `F`                       |
| `occupation`      | numérica    | inteiro 0–20                    |
| `city_category`   | categórica  | `A` / `B` / `C`                 |
| `stay_years`      | numérica    | inteiro 0–4                     |
| `marital_status`  | numérica    | 0 / 1                           |
| `purchase_amount` | numérica    | float (R$)                      |
| `quantity`        | numérica    | inteiro 1–5                     |

**Alvos:**

- `age_group` ∈ {`0-17`, `18-25`, `26-35`, `36-45`, `46-50`, `51-55`, `55+`}
- `product_category` ∈ {`Eletronicos`, `Roupas`, `Casa`, `Alimentos`, `Beleza`, `Brinquedos`, `Esportes`}
- `payment_method` ∈ {`Cartao_Credito`, `Cartao_Debito`, `PIX`, `Dinheiro`, `Boleto`}

**Dependências sintéticas (placeholder)** embutidas para gerar sinal aprendível:

- `age_group` ← `occupation`, `marital_status`;
- `payment_method` ← `city_category`, `purchase_amount`;
- `product_category` ← `gender`, `age_group` (+ reforço direto de `occupation`/`quantity`,
  que o modelo consegue observar).

---

## Decisões justificadas

| Decisão | Escolha | Por que |
|---|---|---|
| Estratégia | **3 `RandomForestClassifier` independentes** (um por alvo) | Os três alvos são tratados como **condicionalmente independentes** dadas as features; cada modelo tem seu próprio `predict_proba`, dando um **grau de certeza por alvo**. Um `MultiOutputClassifier`/cadeia exploraria correlação entre alvos, ao custo de mais complexidade e do risco de propagar o erro de um alvo para o outro (ver `outputs/debate_Q3.md`). |
| Pré-processamento | **`OneHotEncoder` (categóricas) + `StandardScaler` (numéricas)** num `ColumnTransformer` | One-hot evita impor **ordem espúria** a `gender`/`city_category`; o `StandardScaler` impede que uma feature numérica domine só pela unidade/escala. Tudo dentro do `Pipeline`, com `fit` **só no treino** (sem vazamento). |
| Split | **Estratificado, `test_size=0.25`** | Preserva a proporção das classes do alvo no teste — essencial com classes raras (`Brinquedos` n=40, `Boleto` n=96), que de outro modo poderiam sumir do teste. |
| Floresta | **`n_estimators=300`** + **`compress=3`** no `.joblib` | 300 árvores dão uma floresta robusta e estável; `compress=3` só **reduz o tamanho do arquivo** salvo (RandomForest comprime muito bem), **sem alterar as previsões nem as métricas**. |
| Grau de certeza | **`max(predict_proba)`** do respectivo modelo | Escore de confiança **relativo** para ranquear a classe mais provável. É proba **crua, NÃO calibrada** — a média das árvores tende a ser mal calibrada (ver `outputs/debate_Q3.md`). |
| Dados ausentes | **Fallback sintético (seed 42)**, salvo em `data/black_friday_sintetico.csv` | Sem o CSV real, prova o pipeline **ponta a ponta** de forma reprodutível, com banner `!!!` BEM VISÍVEL avisando que os números são **placeholder**. |

---

## Como executar

Windows (PowerShell):

```powershell
.\run.ps1
```

Linux / macOS / Git Bash:

```bash
bash run.sh
```

Ou manualmente:

```bash
python src/main.py        # treina os 3 modelos, gera outputs/
python src/inferencia.py  # demo: venda nova -> 3 previsões com % de certeza
```

Dependências em `requirements.txt`. Tudo usa `random_state=42`.

---

## Resultados REAIS desta execução (dados sintéticos / placeholder)

Split estratificado: **3000 linhas de treino / 1000 de teste** por alvo.
"Acaso" = acurácia de prever sempre a **classe majoritária** (baseline).

| Alvo                | Acurácia global | Acaso (maj.) | F1 macro | F1 ponderado |
|---------------------|-----------------|--------------|----------|--------------|
| `product_category`  | **0,287**       | 0,211        | 0,248    | 0,270        |
| `payment_method`    | **0,370**       | 0,313        | 0,273    | 0,339        |
| `age_group`         | **0,417**       | 0,207        | 0,406    | 0,416        |

Os três modelos ficam **acima do acaso**, como esperado pelas dependências
embutidas. `age_group` é o mais previsível (sinal `occupation`/`marital_status`
mais limpo); `product_category` é o mais difícil (sete classes desbalanceadas,
parte do sinal passa por `age_group`, que o modelo não observa diretamente).

### Sensibilidade e especificidade por classe (one-vs-rest)

Calculadas a partir da matriz de confusão: `sensibilidade = TP/(TP+FN)`,
`especificidade = TN/(TN+FP)`. (Valores completos em `outputs/metricas.json`.)

**`age_group`** (destaques):

| classe  | sensibilidade | especificidade | suporte |
|---------|---------------|----------------|---------|
| `0-17`  | 0,439         | 0,969          | 57      |
| `55+`   | 0,588         | 0,901          | 194     |
| `51-55` | 0,430         | 0,842          | 207     |

**`payment_method`** (destaques):

| classe           | sensibilidade | especificidade | suporte |
|------------------|---------------|----------------|---------|
| `Cartao_Credito` | 0,677         | 0,702          | 313     |
| `PIX`            | 0,412         | 0,714          | 255     |
| `Boleto`         | 0,042         | 0,960          | 96      |

**`product_category`** (destaques):

| classe       | sensibilidade | especificidade | suporte |
|--------------|---------------|----------------|---------|
| `Esportes`   | 0,545         | 0,728          | 211     |
| `Beleza`     | 0,403         | 0,880          | 149     |
| `Brinquedos` | 0,100         | 0,986          | 40      |

Padrão típico de problema multiclasse desbalanceado: classes raras têm
**especificidade alta** (raramente acusadas por engano) mas **sensibilidade baixa**
(o modelo erra em recuperá-las), enquanto as classes frequentes têm sensibilidade
mais alta.

### Demo de inferência (`src/inferencia.py`)

Para uma venda nova (`gender=F`, `occupation=12`, `city_category=A`,
`stay_years=3`, `marital_status=1`, `purchase_amount=850.0`, `quantity=2`):

| Alvo previsto         | Classe           | Grau de certeza |
|-----------------------|------------------|-----------------|
| Categoria do produto  | `Beleza`         | 37,3%           |
| Forma de pagamento    | `Cartao_Credito` | 74,0%           |
| Faixa etária          | `55+`            | 53,3%           |

O grau de certeza é o `max(predict_proba)` de cada modelo.

---

## Estrutura do projeto

```
q3_black_friday/
├── data/
│   └── black_friday_sintetico.csv      # gerado (placeholder); o real seria black_friday.csv
├── src/
│   ├── main.py                          # gera/carrega dados, treina e avalia os 3 modelos
│   └── inferencia.py                    # demo: venda nova -> 3 previsões + % de certeza
├── outputs/
│   ├── matriz_confusao_product_category.png
│   ├── matriz_confusao_payment_method.png
│   ├── matriz_confusao_age_group.png
│   ├── metricas.json                    # acurácia, F1, sens./espec. por classe (3 alvos)
│   ├── pipeline_product_category.joblib
│   ├── pipeline_payment_method.joblib
│   ├── pipeline_age_group.joblib
│   ├── colunas_entrada.json
│   └── debate_Q3.md                     # discussão técnica
├── requirements.txt
├── run.ps1
├── run.sh
└── README.md
```

### O que cada saída significa (pasta `outputs/`)

| Arquivo | Significado |
|---|---|
| `pipeline_product_category.joblib` | `Pipeline` treinado (pré-processamento + RandomForest) do alvo `product_category`; `compress=3`; consumido pela inferência. |
| `pipeline_payment_method.joblib` | Idem para `payment_method`. |
| `pipeline_age_group.joblib` | Idem para `age_group`. |
| `colunas_entrada.json` | Ordem das **7 features** de entrada; a inferência monta a venda nessa ordem para não gerar erro silencioso. |
| `metricas.json` | Acurácia global, acaso (baseline majoritária), F1 macro/ponderado, matriz de confusão e **sensibilidade/especificidade one-vs-rest por classe** dos 3 alvos + flag `dados_sinteticos_placeholder`. |
| `matriz_confusao_product_category.png` | Acerto por classe (heatmap `Blues`); o título carrega a **acurácia global** como ressalva e, sob placeholder, a marca `DADOS SINTETICOS`. |
| `matriz_confusao_payment_method.png` | Idem para `payment_method`. |
| `matriz_confusao_age_group.png` | Idem para `age_group`. |
| `debate_Q3.md` | Discussão técnica (proba cru vs. calibrado, sens./espec. multiclasse, 3 modelos vs. multi-saída). |

---

## Leitura dos números

Tradução das métricas para a linguagem do domínio, com honestidade sobre as
limitações (que aqui são **do sinal sintético, não do algoritmo**):

- **`age_group` é o mais previsível** (acurácia 0,417): `occupation` e
  `marital_status` dão o sinal mais limpo e diretamente observável pelo modelo.
- **`product_category` é o mais difícil** (acurácia 0,287): sete classes
  desbalanceadas e parte do sinal passa por `age_group`, que esse modelo **não
  observa diretamente**.
- **Classes raras quase nunca são recuperadas:** `Boleto` tem sensibilidade
  **0,042** e `Brinquedos` **0,100** — limitação do sinal sintético, não defeito
  do modelo. Em compensação, têm **especificidade alta** (raramente acusadas por
  engano), o padrão clássico do multiclasse desbalanceado.

---

## Sem vazamento de dados

Cada modelo recebe **apenas as features** como entrada; os outros dois alvos
**nunca** entram como variável de entrada — é justamente o que cada modelo
precisa prever. O `OneHotEncoder` e o `StandardScaler` vivem **dentro do
`Pipeline`** e são ajustados (`fit`) **somente no conjunto de treino**; na
inferência apenas se aplica (`transform`/`predict`), de modo que nenhuma
estatística da venda nova influencia o pré-processamento — **nada é reajustado**.
O split é **estratificado** por alvo, preservando a proporção das classes (e
protegendo as raras), e `random_state=42` garante reprodutibilidade ponta a
ponta.

Uma ressalva honesta sobre o **grau de certeza**: ele é o `max(predict_proba)`
**cru** de cada RandomForest — um escore de confiança **relativo**, e **não uma
probabilidade calibrada**. Um "74,0% de certeza" não significa que, em 74% dos
casos parecidos, a classe se confirma; a calibração (via `CalibratedClassifierCV`)
fica como próximo passo quando houver o dataset real (não se calibra sobre dados
sintéticos). Discussão completa em `outputs/debate_Q3.md`.
