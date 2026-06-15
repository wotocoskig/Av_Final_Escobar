# Debate técnico — Q3 (Black Friday, multi-classificação)

Documento de discussão das principais decisões de projeto. Para cada tema,
listo 2–3 perspectivas e fecho com a **decisão adotada** neste trabalho.

> Lembrete: os números desta entrega vêm de **dados sintéticos (placeholder)**.
> As decisões metodológicas abaixo, porém, valem igualmente para o dataset real.

---

## 1. `predict_proba` cru vs. calibrado

O "grau de certeza" exibido na inferência é o `max(predict_proba)` de cada
RandomForest. Mas essa probabilidade é confiável como **probabilidade de fato**?

### Perspectiva A — usar a probabilidade crua (o que foi feito)
- **Prós:** simples, sem etapa extra, sem gastar dados num conjunto de
  calibração. Para **ranquear** classes (qual é a mais provável) e dar uma
  noção de confiança relativa, o `predict_proba` cru já serve.
- **Contras:** a média das árvores de um RandomForest tende a ser **mal
  calibrada** — costuma ser "tímida" nos extremos (empurra probabilidades para
  longe de 0 e de 1). Um "74% de certeza" não significa que, em 74% dos casos
  parecidos, a classe se confirma.

### Perspectiva B — calibrar (Platt/sigmoid ou isotônica)
- **Prós:** `CalibratedClassifierCV` (com `cv` interno) faz a probabilidade
  reportada bater melhor com a frequência observada. Importante quando o número
  vai virar **decisão de negócio** (ex.: oferecer parcelamento só se a certeza
  da forma de pagamento > X%).
- **Contras:** custo computacional maior, mais um hiperparâmetro (sigmoid vs.
  isotônica), e a isotônica pode **sobreajustar** com poucos dados por classe —
  e aqui há classes raras (`Brinquedos` n=40, `Boleto` n=96).

### Perspectiva C — multiclasse muda o jogo
- Calibração é teoricamente mais limpa no caso **binário**. No multiclasse
  (5 e 7 classes aqui), calibra-se **one-vs-rest** e depois re-normaliza, o que
  introduz aproximações. Avaliar a calibração exige curvas de confiabilidade /
  Brier score **por classe**.

### Decisão
Mantemos o **`predict_proba` cru** nesta entrega, deixando claro que ele serve
como **score de confiança relativo**, não como probabilidade calibrada. A
calibração (via `CalibratedClassifierCV`, provavelmente **sigmoid** pelo tamanho
das classes raras) fica como **próximo passo** assim que houver o dataset real e
um conjunto dedicado a medir Brier score — **não se calibra sobre dados
sintéticos** (a calibração aprenderia o ruído do gerador, não a realidade).

---

## 2. Como medir especificidade e sensibilidade em multiclasse

Sensibilidade e especificidade são definidas para problema **binário**. Com 5–7
classes, é preciso decidir como estendê-las.

### Perspectiva A — one-vs-rest por classe (o que foi feito)
Para cada classe `i`, a partir da matriz de confusão:
- `TP` = acertos da classe `i`;
- `FN` = exemplos da classe `i` previstos como outra;
- `FP` = exemplos de outras classes previstos como `i`;
- `TN` = todo o resto.
- `sensibilidade_i = TP/(TP+FN)` (= recall da classe `i`);
- `especificidade_i = TN/(TN+FP)`.

- **Prós:** dá um valor **por classe**, expondo onde o modelo falha. Revela o
  padrão clássico — classe rara com **especificidade altíssima** (quase nunca é
  acusada por engano) mas **sensibilidade baixa** (raramente recuperada), como
  `Brinquedos` (espec. 0,986 / sens. 0,100) e `Boleto` (espec. 0,960 / sens.
  0,042) nesta execução.
- **Contras:** são 5–7 pares de números por alvo; não há um valor único de
  "especificidade do modelo".

### Perspectiva B — médias (macro vs. ponderada)
- **Macro:** média simples entre classes — trata todas as classes como
  igualmente importantes, então **penaliza** bem o desempenho ruim nas raras.
- **Ponderada (weighted):** pondera pelo suporte — reflete melhor a acurácia
  "sentida" no conjunto, mas **mascara** o fracasso nas classes raras.
- **Prós:** resumem tudo num número. **Contras:** escondem exatamente a
  informação mais útil (onde o modelo erra). Reportamos F1 **macro e ponderado**
  no JSON justamente para mostrar essa diferença.

### Perspectiva C — micro / acurácia global
A média **micro** de recall em multiclasse de rótulo único colapsa na
**acurácia global** — por isso reportamos a acurácia global separadamente e não
duplicamos como "recall micro".

### Decisão
Adotamos **one-vs-rest por classe** como métrica principal de sensibilidade e
especificidade (gravadas em `outputs/metricas.json`, com a matriz de confusão
completa), complementadas por **F1 macro e ponderado** para uma visão resumida.
Essa combinação mostra tanto o desempenho agregado quanto **onde** cada um dos
três modelos acerta e erra.

---

## 3. (Bônus) Três modelos independentes vs. um classificador multi-saída

- **Independentes (decisão adotada):** três `RandomForestClassifier`, um por
  alvo. Cada um pode ter seu próprio comportamento e seu `predict_proba`
  próprio; é simples de treinar, avaliar e salvar separadamente. Assume que os
  três alvos são **condicionalmente independentes** dadas as features.
- **`MultiOutputClassifier` / cadeia de classificadores:** trataria os três
  alvos de forma conjunta; uma *classifier chain* poderia explorar correlação
  entre alvos (ex.: `age_group` → `product_category`). Custo: mais complexidade
  e risco de propagar erro de um alvo para o outro.

**Decisão:** três modelos independentes, alinhado ao requisito do projeto e mais
transparente para extrair um **grau de certeza por alvo** via `max(predict_proba)`.

---

*Aluno 1 - estilo metodico e didatico.*
