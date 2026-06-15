# TESTE — Q1 Heart Failure (Aluno 1)

Ambiente: Windows, PowerShell, `python` 3.14.4, scikit-learn 1.8.0, pandas 3.0.2, numpy 2.4.4.
Pontos de entrada (de `run.ps1`): `python src/main.py` e depois `python src/inferencia.py`.

## Comando 1 — Treino
```
cd C:\Users\cezar\escobar_avaliacao_semestral\aluno_1\q1_heart_failure
python src/main.py
```
**EXIT CODE: 0**

Métricas obtidas (idênticas ao README e ao `metricas.json`):
- k escolhido: **2** (maior silhouette em k=2..8)
- Silhouette: **0.1335**
- Davies-Bouldin: **2.5640**
- Calinski-Harabasz: **37.96**
- n por cluster: {0: 187, 1: 112}
- Taxa de óbito (caracterização): Cluster 0 = **13.9%**, Cluster 1 = **62.5%**

Silhouette por k: k2=0.1335, k3=0.1235, k4=0.1131, k5=0.1310, k6=0.1247, k7=0.1185, k8=0.1265.

Artefatos (re)gerados: silhouette_por_k.png, clusters_pca.png, perfil_clusters.csv/.txt, metricas.json, pipeline_kmeans.joblib, colunas_entrada.json. Todos presentes.

## Comando 2 — Inferência
```
python src/inferencia.py
```
**EXIT CODE: 0**

### Screenshot textual da saída da inferência
```
============================================================
Q1 - HEART FAILURE | DEMO DE INFERENCIA
============================================================
Paciente NOVO (desconhecido) apresentado ao sistema:
                         age: 60.0
                     anaemia: 0
    creatinine_phosphokinase: 250
                    diabetes: 1
           ejection_fraction: 35
         high_blood_pressure: 1
                   platelets: 262000.0
            serum_creatinine: 1.2
                serum_sodium: 137
                         sex: 1
                     smoking: 0
                        time: 115
============================================================
RESULTADO DO AGRUPAMENTO
============================================================
Paciente atribuido ao CLUSTER 0.
Esse grupo tem 187 pacientes de treino.
Taxa de obito (caracterizacao) do grupo: 13.9%
Perfil medio do grupo (algumas variaveis):
  media                  age: 55.86
  media    ejection_fraction: 39.27
  media     serum_creatinine: 1.06
  media                 time: 158.96
============================================================
CONCLUIDO
============================================================
```

## Conclusão do teste
Fluxo completo reprodutível, exit code 0 nos dois scripts. Métricas e saída da inferência batem com o README. Paciente novo (sem DEATH_EVENT) atribuído ao Cluster 0 (grupo de menor risco, 13.9%), coerente com o perfil clínico favorável.
