# Debate Q1 — Decisoes polemicas do clustering

Tres perspectivas discutem as decisoes mais controversas deste projeto de
agrupamento de pacientes com insuficiencia cardiaca. Ao final de cada tema ha
uma **DECISAO + PORQUE**.

---

## Tema 1 — Como tratar as binarias (0/1) na distancia euclidiana?

As colunas `anaemia`, `diabetes`, `high_blood_pressure`, `sex` e `smoking` sao
indicadores 0/1. O KMeans mede distancia euclidiana, entao a forma de
representar essas colunas muda o resultado.

**Perspectiva A — "Escalar tudo junto" (StandardScaler em todas):**
defende padronizar tambem as binarias, "para colocar todas as variaveis na
mesma escala". O problema: ao padronizar uma binaria, a distancia entre as duas
categorias passa a depender da *frequencia* da categoria. Uma condicao rara
(poucos 1) vira um desvio-padrao pequeno, e a diferenca 0->1 escalada fica
**enorme**, dominando a distancia. Isso da peso desproporcional a categorias
raras, sem justificativa clinica.

**Perspectiva B — "Passthrough" (deixar 0/1 cru):**
argumenta que numa binaria 0/1 a distancia euclidiana entre as categorias ja
vale exatamente 1 — um valor estavel, interpretavel e independente da
frequencia. Mantendo as continuas padronizadas (desvio 1), cada binaria
contribui com peso comparavel ao de uma continua, sem distorcao. E a abordagem
mais simples e robusta.

**Perspectiva C — "K-prototypes / Gower":**
sugere abandonar o KMeans puro e usar um metodo proprio para dados mistos
(K-prototypes, ou distancia de Gower). E teoricamente o mais correto, mas
adiciona dependencia externa e complexidade, e o enunciado exige KMeans.

> **DECISAO:** binarias como **passthrough** (NAO escaladas), continuas com
> **StandardScaler**, via `ColumnTransformer`.
> **PORQUE:** numa binaria 0/1 a distancia entre categorias ja e 1, estavel e
> interpretavel; escalar inflaria o peso de categorias raras. Padronizando so
> as continuas, todas as variaveis contribuem com peso comparavel, mantendo o
> KMeans simples e sem dependencias externas.

---

## Tema 2 — Qual scaler para as variaveis continuas?

**Perspectiva A — StandardScaler (z-score):**
centra em media 0 e desvio 1. Coloca todas as continuas na mesma escala de
dispersao, que e exatamente o que a distancia euclidiana do KMeans precisa.
Padrao da area para clustering baseado em distancia.

**Perspectiva B — MinMaxScaler (0..1):**
comprime tudo para [0,1]. Funciona, mas e muito sensivel a outliers: um unico
`creatinine_phosphokinase` ou `platelets` extremo achata todos os demais
valores num cantinho da faixa, perdendo resolucao.

**Perspectiva C — RobustScaler (mediana/IQR):**
resistente a outliers, atraente porque o dataset tem caudas longas (CPK,
plaquetas). Porem deixa as escalas menos comparaveis entre si e foge do scaler
exigido para este aluno.

> **DECISAO:** **StandardScaler** nas continuas.
> **PORQUE:** e o scaler natural para distancia euclidiana (equaliza a
> dispersao), e o padrao para KMeans e atende a exigencia do projeto. Os
> outliers existem, mas a padronizacao por z-score ja reduz bastante a
> dominancia de unidades distintas sem introduzir as fragilidades do MinMax.

---

## Tema 3 — Como escolher k (numero de clusters)?

**Perspectiva A — Metodo do cotovelo (inercia):**
plota a inercia (WCSS) por k e procura o "cotovelo". E visual e popular, mas
muitas vezes ambiguo — o cotovelo nao e claro e a leitura vira subjetiva.

**Perspectiva B — Silhouette:**
mede, para cada ponto, quao bem ele cabe no proprio cluster versus o vizinho
mais proximo. Da um numero unico e comparavel por k; basta varrer k=2..8 e
pegar o maior. Objetivo e reprodutivel.

**Perspectiva C — Conhecimento de dominio (fixar k=2):**
"so importa quem morre e quem nao morre, entao k=2". Mas isso e classificacao
disfarcada e usa o rotulo que deve ficar FORA do treino. O agrupamento deve
emergir dos dados clinicos, nao do desfecho.

> **DECISAO:** escolher k por **maior silhouette**, varrendo **k=2..8**, com o
> grafico `silhouette_por_k.png` como evidencia.
> **PORQUE:** o silhouette da um criterio objetivo, unico e reprodutivel,
> evitando a ambiguidade do cotovelo e o vazamento de fixar k pelo desfecho.

---

## Tema 4 (bonus) — O que fazer com `time`?

`time` e o tempo de acompanhamento (dias). Sabe-se que ele se correlaciona
fortemente com o desfecho (acompanhamentos curtos costumam coincidir com
obitos precoces). Ha duas posturas:

- **Remover `time`:** evita "contaminar" os grupos com um proxy do desfecho.
- **Manter `time` como continua escalada:** e uma variavel clinica legitima
  (intensidade/duracao do seguimento) e o enunciado pede para inclui-la.

> **DECISAO:** **manter `time`** como continua escalada (StandardScaler).
> **PORQUE:** o projeto exige inclui-la, e ela e uma medida real de
> acompanhamento. Registramos a ressalva de que `time` se relaciona com o
> desfecho; por isso a taxa de obito por cluster e lida como *caracterizacao*
> descritiva, e nao como prova de poder preditivo do agrupamento.
