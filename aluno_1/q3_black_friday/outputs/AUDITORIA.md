# AUDITORIA — Q3 Black Friday (Projeto A / aluno_1)

**Estratégia esperada:** 3 RandomForest independentes + `predict_proba`.
**Estratégia entregue:** 3 `RandomForestClassifier` (n_estimators=300) independentes, um por alvo, cada um em `Pipeline(ColumnTransformer + RF)`. **Confere.**

Data da auditoria: 2026-06-14 · Avaliador: auditor/testador Q3.

---

## PONTOS POSITIVOS

- **Arquitetura correta e fiel ao esperado.** Três modelos independentes, cada um com seu próprio `Pipeline` (one-hot nas categóricas + `StandardScaler` nas numéricas + RandomForest). Verificado nos `.joblib`: cada pipeline tem `feature_names_in_` = as 7 features, **sem nenhum alvo**.
- **Sem vazamento, comprovado.** `X = df[COLUNAS_FEATURES]`; os outros dois alvos nunca entram como feature. Pré-processamento (`fit`) ocorre só no treino, dentro do Pipeline; na inferência só há `predict`/`predict_proba`. Confirmado carregando os artefatos.
- **Especificidade e sensibilidade por classe corretas.** Função `metricas_por_classe` aplica one-vs-rest a partir da matriz de confusão: `sens=TP/(TP+FN)`, `espec=TN/(TN+FP)`. Recalculei de forma independente a partir das matrizes salvas em `metricas.json`: **0 divergências**.
- **Grau de certeza via `predict_proba`.** `grau_certeza = max(predict_proba)` em `inferencia.py`. Correto.
- **Fallback de dados implementado e bem sinalizado.** Lê `data/black_friday.csv` se existir; senão gera sintético (seed 42), salva em `data/black_friday_sintetico.csv` e imprime banner `!!!` BEM VISÍVEL (impresso 2x: após gerar e no fim). README documenta o placeholder com destaque. `metricas.json` carrega flag `dados_sinteticos_placeholder: true`.
- **Reprodutibilidade total.** `random_state=42` em geração, split (estratificado) e modelo. Reexecutei do zero: métricas batem **exatamente** com o `metricas.json` versionado (acc product=0.287, payment=0.370, age=0.417).
- **Split estratificado** por alvo (`test_size=0.25`, stratify=y).
- **README completo:** esquema, dependências sintéticas, tabela de métricas reais, sens/espec por classe, demo de inferência — todos com números reais preenchidos e coerentes com a execução.
- **debate_Q3.md** discute proba cru vs calibrado, sens/espec multiclasse (one-vs-rest), e 3 modelos vs multi-saída, com decisão justificada.
- **Sistema funcional:** dada uma venda nova, retorna os 3 alvos com % de certeza (Beleza 37,3% / Cartao_Credito 74,0% / 55+ 53,3%), idêntico ao README.

## PONTOS NEGATIVOS / RISCOS

- **Tamanho dos modelos (risco operacional, mitigado).** Os 3 `.joblib` somam ~38 MB (product=14,0 MB, payment=12,2 MB, age=11,9 MB) numa reexecução limpa no ambiente atual — o `compress=3` já resolve o risco de versionar/distribuir (o número anterior de ~280 MB referia-se a artefatos salvos sem compressão e não reproduz). Sugestão opcional remanescente: limitar `max_depth` ou reduzir árvores caso se queira reduzir ainda mais.
- **Métricas modestas — esperado e aceitável.** Acurácias 0,29–0,42, acima do acaso (baseline majoritária 0,21–0,31). Como os dados são sintéticos placeholder, isso é aceitável; o pipeline e as métricas estão corretos.
- **Classes raras com sensibilidade baixa** (ex.: Brinquedos 0,10; Boleto 0,042) — comportamento típico e corretamente discutido no README/debate, não é defeito.
- **Observação menor:** `metricas.json` reordena `features` como `[gender, city_category, occupation, ...]` (categóricas + numéricas), que é exatamente `COLUNAS_FEATURES`; consistente, sem problema.

---

## CHECKLIST

| Item | Status | Observação |
|------|--------|------------|
| Trata os 3 alvos (product_category, payment_method, age_group) | OK | Loop sobre `ALVOS`, 3 modelos |
| Por alvo: pipeline, acurácia global, acurácia por classe (matriz), F1 | OK | acc global, acc_ovr por classe, F1 macro+ponderado, matriz PNG |
| Especificidade E sensibilidade por classe (one-vs-rest correto) | OK | Fórmula verificada; 0 divergências vs recálculo independente |
| Grau de certeza via `predict_proba` | OK | `max(predict_proba)` |
| Sem vazamento (alvos não viram feature; pré-proc só no treino) | OK | `feature_names_in_` confirma só 7 features |
| Fallback CSV real → sintético com aviso visível + README | OK | Banner `!!!` 2x + flag no JSON + README |
| Sistema funcionando (venda → 3 previsões com %) | OK | Inferência reproduz o README |
| README explica execução e saídas; números reais | OK | Completo e coerente |
| Reprodutibilidade (seed 42) | OK | Reexecução bate exatamente |
| Matriz de confusão PNG dos 3 alvos | OK | 3 PNGs regenerados |
| metricas.json com espec E sens por classe | OK | Presente para os 3 alvos |
| Modelos salvos (.joblib) | OK | 3 pipelines (~38 MB no total com `compress=3`) |
| Dado sintético gerado em data/ | OK | `black_friday_sintetico.csv` |

---

## VEREDITO: **APROVADO**

Projeto fiel à estratégia esperada (3 RandomForest independentes), sem vazamento, com especificidade/sensibilidade por classe corretas, fallback sinalizado e reprodutibilidade comprovada. Sistema de inferência funcional.

**Correções obrigatórias:** nenhuma.

**Sugestões opcionais (não bloqueiam):** o `compress=3` já mantém os `.joblib` em ~38 MB no total; reduzir ainda mais (ex.: `max_depth` ou menos árvores) é possível, mas não necessário para distribuição.
