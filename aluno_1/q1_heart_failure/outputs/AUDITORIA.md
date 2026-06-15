# AUDITORIA — Q1 Heart Failure (Aluno 1: KMeans + StandardScaler, binárias passthrough)

Auditor: avaliação semestral (não corretiva — apenas avalia e testa).
Data: 2026-06-14.

## Resumo do projeto
- **Algoritmo:** KMeans (`n_init=10`, `random_state=42`), k escolhido por maior silhouette em k=2..8.
- **Scaler:** StandardScaler nas 7 contínuas; binárias (`anaemia`, `diabetes`, `high_blood_pressure`, `sex`, `smoking`) como `passthrough` num `ColumnTransformer` dentro de um `Pipeline`.
- **Alvo:** `DEATH_EVENT` removido das features; usado só para caracterizar (taxa de óbito por cluster).
- **`time`:** mantido como contínua escalada, com ressalva explícita de que se relaciona com o desfecho.

## PONTOS POSITIVOS
1. **Clustering legítimo e sem alvo no treino.** `X_treino = df.drop(columns=["DEATH_EVENT"])`; o alvo só entra em `gerar_perfil_clusters` para descrever os grupos. Conceito de não supervisionado respeitado.
2. **Pipeline encapsulado.** Pré-processamento + KMeans num único `Pipeline`; a inferência só chama `predict`, garantindo que o `StandardScaler` aplicado ao paciente novo é o mesmo do treino (sem refit).
3. **Tratamento das binárias justificado e correto para a abordagem.** A escolha de passthrough é coerente: numa binária 0/1 a distância euclidiana entre categorias já vale 1; isso é argumentado no README e no `debate_Q1.md` (Tema 1) contra escalar tudo / usar Gower.
4. **Seleção de k objetiva e reprodutível.** Silhouette varrido em k=2..8, gráfico `silhouette_por_k.png` gerado, k=2 escolhido pelo argmax. Decisão registrada e reproduzível (`random_state=42`).
5. **Melhor separação clínica das três versões.** Os grupos separam óbito de forma nítida (13.9% vs 62.5%), mostrando que a estrutura não supervisionada captura sinal clínico real.
6. **Inferência realista.** Paciente novo sem `DEATH_EVENT`, montado na ordem correta de colunas (`colunas_entrada.json`), retorna cluster + caracterização do grupo.
7. **Artefatos completos e métricas reais.** Todos os entregáveis presentes; os números do README batem exatamente com a execução.
8. **Decisão sobre `time` explícita.** README (tabela) e `debate_Q1.md` (Tema 4) declaram a manutenção de `time` e a ressalva de correlação com o desfecho.

## PONTOS NEGATIVOS / RISCOS
1. **Framing de "sem vazamento" levemente impreciso (menor).** O README e o cabeçalho de `inferencia.py` afirmam que o `StandardScaler` é ajustado "apenas nos dados de treino". Na prática, **não há split treino/teste**: o `fit` ocorre sobre os 299 pacientes. Para um pipeline de clustering demonstrativo isso é aceitável (a separação real que importa é treino → paciente novo, que está correta: o paciente novo só é `transform`ado). Porém a redação dá a entender uma separação treino/teste que não existe. **Não compromete a tarefa**, mas é um deslize de redação a corrigir para precisão conceitual.
2. **`time` mantido inflaciona a separação por óbito (risco didático, mitigado).** Como `time` correlaciona fortemente com o desfecho, parte da forte separação 13.9% vs 62.5% vem de um proxy do alvo. O aluno **reconhece isso explicitamente** e lê a taxa de óbito como caracterização descritiva, não como prova preditiva — o que neutraliza o risco. Registrado apenas como ponto de atenção.
3. **Silhouette baixo em termos absolutos (0.1335).** Esperado para esse dataset com muitas binárias no espaço; não é defeito, mas a estrutura geométrica é fraca. Bem contextualizado.

## Checklist de critérios

| Item | Status | Observação |
|------|--------|------------|
| Clustering de verdade; DEATH_EVENT fora do treino, só caracteriza | OK | `drop(columns=["DEATH_EVENT"])`; alvo só no perfil. |
| Binárias tratadas e justificadas | OK | Passthrough; justificado no README e debate (Tema 1). |
| Sem vazamento (fit só no treino; paciente novo só transformado) | OK (com ressalva) | Paciente novo só `transform`ado via Pipeline. Não há split treino/teste; a frase "fit só no treino" é imprecisa pois treina nos 299. Não quebra a tarefa. |
| Todo o pré-processamento demonstrado | OK | ColumnTransformer explícito; etapas impressas no console. |
| Inferência roda, paciente novo sem DEATH_EVENT, retorna grupo | OK | Cluster 0, 13.9% óbito; exit 0. |
| README explica como rodar e o que cada saída significa; números reais | OK | Tabela de saídas + resultados reais batem com a execução. |
| Decisão sobre 'time' explícita e justificada | OK | Mantido como contínua escalada, com ressalva de correlação (debate Tema 4). |

## VEREDITO: **APROVADO**

Projeto correto, reprodutível e bem justificado; a melhor separação clínica das três versões. Não há defeito que comprometa a tarefa.

### Correções recomendadas (não bloqueantes)
1. Ajustar a redação de "sem vazamento": ou (a) introduzir um split treino/teste real para que a frase "fit só no treino" seja literal, ou (b) reformular para "o scaler é ajustado no conjunto de treino (todos os 299 pacientes) e o paciente novo é apenas transformado", deixando claro que não há split.
2. (Opcional) Reforçar no README, ao lado da taxa 13.9% vs 62.5%, que parte da separação por óbito é influenciada pela inclusão de `time` (proxy do desfecho) — o debate já diz isso; basta amarrar ao número.
