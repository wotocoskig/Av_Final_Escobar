# Debate Q2 - Wine Quality

Este documento registra, de forma metodica, as principais decisoes de modelagem
para a classificacao da qualidade de vinhos (alvo `quality`, inteiro de 3 a 9).
O conjunto une vinho tinto (1.599) e branco (4.898), totalizando 6.497 amostras,
com a feature adicional `tipo` (tinto=0 / branco=1).

O problema central e o **desbalanceamento severo** das classes:

| quality | 3  | 4   | 5    | 6    | 7    | 8   | 9 |
|---------|----|-----|------|------|------|-----|---|
| amostras| 30 | 216 | 2138 | 2836 | 1079 | 193 | 5 |

As classes 5 e 6 concentram ~76% dos dados; as classes 3 e 9 sao raríssimas
(30 e 5 amostras no total).

---

## Tema 1 - Como tratar as classes raras (3 e 9)?

**Perspectiva A - Manter todas as classes como estao.**
Preserva o problema original (7 classes) e a interpretabilidade da nota. A
estratificacao no split garante presenca minima de cada classe em treino e
teste. Desvantagem: com apenas 5 amostras da classe 9 (4 treino / 1 teste),
nenhum modelo consegue aprende-la — de fato, a acuracia das classes 3 e 9 ficou
em 0,0000 no vencedor.

**Perspectiva B - Usar pesos de classe (`class_weight="balanced"`).**
Em vez de descartar dados, penaliza mais os erros nas classes minoritarias. Foi
a opcao adotada no RandomForest. Melhora o recall das classes intermediarias
(7 e 8) sem remover informacao. Limitacao: nao cria amostras novas, entao
classes com 4-5 exemplos continuam praticamente inalcancaveis.

**Perspectiva C - Reamostragem (SMOTE/undersampling).**
Poderia equilibrar artificialmente as classes. Rejeitada aqui porque: (1) exige
dependencia externa (imbalanced-learn) fora do escopo; (2) sintetizar vizinhos
para uma classe de 5 amostras gera ruido pouco confiavel; (3) aumenta o risco de
vazamento se mal aplicada (deve ficar dentro do pipeline, so no treino).

**Decisao.** Manter as 7 classes (Perspectiva A) combinado com `class_weight`
no modelo de arvore (Perspectiva B). E a abordagem que respeita os dados reais,
nao introduz dependencias nem dados sinteticos, e ainda assim atenua o
desbalanceamento. Documentamos explicitamente que as classes 3 e 9 sao
estatisticamente inviaveis de aprender com tao poucas amostras.

---

## Tema 2 - F1-macro vs micro/weighted: qual guia a selecao?

**Perspectiva A - Acuracia / F1-micro.** Faceis de ler, mas dominadas pelas
classes 5 e 6. Um modelo que so acerta 5 e 6 ja passa de ~70% de acuracia
"parecendo bom", mascarando o fracasso nas demais classes. Inadequado como
criterio unico de selecao aqui.

**Perspectiva B - F1-weighted.** Pondera o F1 de cada classe pelo seu suporte.
E util para reportar o desempenho "medio que o usuario sente", mas continua
dando pouco peso as classes raras — justamente as que mais nos preocupam.

**Perspectiva C - F1-macro.** Media simples do F1 por classe, tratando todas
igualmente. E o criterio mais exigente neste cenario desbalanceado: forca o
modelo a nao ignorar as classes minoritarias.

**Decisao.** Selecionar o vencedor por **F1-macro** (desempate por acuracia
global), mas **reportar as tres** metricas (acuracia, F1-macro e F1-weighted)
mais a acuracia por classe via matriz de confusao. Assim a escolha e justa com
as classes raras, e o relatorio continua transparente sobre o desempenho global.

---

## Tema 3 - Agrupar `quality` em faixas (ex.: ruim/medio/bom)?

**Perspectiva A - Manter as 7 classes (multiclasse fina).**
Mantem a granularidade original pedida no enunciado. Custo: classes raras
permanecem praticamente impossiveis de prever.

**Perspectiva B - Agrupar em faixas (ex.: 3-5 = "baixa", 6 = "media",
7-9 = "alta").** Reduziria o desbalanceamento e quase certamente elevaria a
acuracia e o F1. E uma simplificacao legitima para uma aplicacao pratica de
"triagem" de qualidade.

**Decisao.** Para esta entrega, **mantemos as 7 classes** porque o enunciado
define o alvo como a nota inteira de 3 a 9 e exige acuracia por classe. O
agrupamento em faixas fica registrado como evolucao recomendada: se o objetivo
de negocio fosse apenas separar vinhos bons de ruins, a faixificacao tornaria o
modelo mais robusto e implantavel.

---

## Tema 4 - Linhas duplicadas no conjunto unido

Ao unir tinto e branco, o conjunto tem **1.177 linhas fisico-quimicas totalmente
duplicadas** (6.497 -> 5.320 distintas). Apos o split estratificado (seed=42),
~27% das linhas de teste tem um vetor de features identico tambem no treino —
ou seja, parte da metrica e medida sobre amostras efetivamente memorizadas
(vazamento sutil que **infla** os numeros).

**Perspectiva A - Manter as duplicatas.** Preserva exatamente as metricas de
referencia desta entrega (acuracia 0,6923 / F1-macro 0,3959). Custo: os numeros
sao otimistas quanto a generalizacao real.

**Perspectiva B - Deduplicar antes do split.** Correcao tecnica do vazamento;
mede a generalizacao de forma honesta. Verificado: derruba a acuracia global
para ~0,56 e o F1-macro para ~0,26 (o vencedor segue RandomForest).

**Decisao.** Para esta entrega **mantemos as duplicatas** (Perspectiva A),
preservando as metricas de referencia, mas **documentamos a limitacao
explicitamente** aqui e no README — nunca de forma silenciosa. Fica registrado
que a deduplicacao e a evolucao recomendada para medir a generalizacao real.

---

## Conclusao para implantacao

O **RandomForest** foi escolhido por apresentar o maior F1-macro entre os tres
modelos avaliados, sendo tambem o de maior acuracia global. O `class_weight`
ajudou nas classes intermediarias, mas reconhecemos abertamente que as classes
3 e 9 nao sao aprendidas — uma limitacao dos dados, nao do algoritmo. O acerto
por classe esta materializado em `outputs/matriz_confusao_RandomForest.png`
(diagonal = recall por classe), evidencia direta do que se discutiu acima. Para
producao, recomenda-se monitorar o desempenho por classe, deduplicar o conjunto
para medir a generalizacao real e considerar o agrupamento em faixas caso o caso
de uso permita.

Aluno 1 - estilo metodico e didatico.
