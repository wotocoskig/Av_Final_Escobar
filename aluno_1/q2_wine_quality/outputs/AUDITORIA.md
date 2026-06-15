# AUDITORIA - Q2 Wine Quality (Aluno 1)

Auditor: avaliacao semestral. Projeto: `aluno_1/q2_wine_quality`.
Modelos esperados: RandomForest, GradientBoosting, KNN. **Confirmado.**

## Pontos positivos

- **Reproduz exatamente.** Treino e inferencia rodam com EXIT CODE 0; as metricas
  da execucao batem com as do README e do `metricas.json` ate a 4a casa
  (acuracia 0.6923, F1-macro 0.3959, F1-weighted 0.6792).
- **3 classificadores distintos e exigidos** (RandomForest, GradientBoosting, KNN),
  cada um em `Pipeline(StandardScaler -> modelo)`. Selecao do vencedor por
  **F1-macro com desempate por acuracia global**, e o desempate esta de fato
  implementado na chave de ordenacao (`key=lambda r: (r["f1_macro"], r["acuracia_global"])`).
- **Sem vazamento.** `StandardScaler` vive dentro do Pipeline; `fit` so ocorre no
  treino via `pipeline.fit(X_treino, ...)`. A inferencia apenas carrega o pipeline
  ja ajustado e chama `predict`/`predict_proba`.
- **Split estratificado** (`stratify=y`, `test_size=0.20`, `random_state=42`).
- **Metricas completas:** acuracia global, acuracia por classe (recall lido da
  diagonal da matriz de confusao), F1-macro e F1-weighted. Funcao
  `acuracia_por_classe` correta (acertos da classe / total da classe).
- **Artefatos todos presentes e regenerados:** `matriz_confusao_RandomForest.png`,
  `comparacao_modelos.json`, `metricas.json`, `modelo_vencedor.joblib` (com
  metadados de colunas/classes para a inferencia).
- **Inferencia robusta:** o artefato salva `colunas` e a entrada e remontada com
  `pd.DataFrame([vinho], columns=colunas)`, garantindo a mesma ordem de features.
- **Justificativa numerica do vencedor** explicita no README e no debate.
- **Classes raras 3 e 9 (recall 0.00) documentadas honestamente** como limitacao
  dos DADOS (30 e 5 amostras totais), nao do codigo. Debate discute SMOTE,
  class_weight e faixas com argumentos.

## Pontos negativos / riscos

- **Usa a feature `tipo` (tinto=0 / branco=1).** E uma escolha legitima e bem
  justificada no README, mas significa que a inferencia EXIGE que o usuario
  informe `tipo`. Nao e defeito; e uma decisao de design (diferencia o projeto de
  B e C). Apenas registrar.
- **`class_weight="balanced"` so no RandomForest**; GradientBoosting e KNN nao tem
  tratamento de desbalanceamento equivalente (GB nao aceita class_weight; KNN nao
  o expoe). A comparacao continua valida, mas nao e 100% "mesma condicao" entre os
  tres. Risco baixo — o vencedor e justamente o que recebeu o tratamento.
- **Vinho de inferencia com `residual sugar` = 6.1** (valor moderado e plausivel
  para um branco; FABRICADO a mao, nao copiado de nenhuma linha dos CSVs).
  Realista o suficiente. OK.
- Tamanho do `.joblib` (~14 MB) por causa de `n_estimators=200` + compress=3 —
  aceitavel.

## Checklist Q2

| # | Criterio | Status | Observacao |
|---|----------|--------|------------|
| 1 | Une tinto+branco preservando colunas; alvo=quality | OK | `pd.concat`, cabecalhos identicos; 6497x13 (inclui `tipo`) |
| 2 | >=3 classificadores + justificativa numerica do vencedor | OK | RF/GB/KNN; vencedor por F1-macro com numeros no README/debate |
| 3 | Acuracia global + por classe + F1 (macro/weighted) | OK | Todas reportadas no stdout, metricas.json e README |
| 4 | Matriz de confusao presente + F1 reportado | OK | `matriz_confusao_RandomForest.png` regenerada; F1 macro+weighted |
| 5 | Split estratificado + random_state fixo | OK | `stratify=y`, `random_state=42` |
| 6 | Sem vazamento (fit so no treino, via Pipeline) | OK | StandardScaler dentro do Pipeline |
| 7 | Inferencia roda com vinho novo realista | OK | EXIT 0; quality=6, prob=59.00% |
| 8 | README explica execucao e saidas; numeros reais | OK | Tabelas com numeros reais conferem com a execucao |
| 9 | (Projeto C) faixas | N/A | Nao se aplica ao Aluno 1 |

## VEREDITO: **APROVADO**

Projeto correto, reprodutivel e honesto. Sem correcoes obrigatorias.

### Sugestoes opcionais (nao bloqueiam aprovacao)
- ~~Mencionar no README que a inferencia exige a feature `tipo` na entrada.~~
  **Feito:** o README documenta o contrato de entrada (12 features, incluindo
  `tipo`) e a inferencia agora valida a entrada (erro claro em vez de previsao
  silenciosa).
- (Opcional) reportar tambem precision/recall por classe via `classification_report`
  para alinhar o nivel de detalhe ao dos colegas.
