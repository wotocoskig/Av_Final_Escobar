"""
Q1 - HEART FAILURE: AGRUPAMENTO (CLUSTERING NAO SUPERVISIONADO)
=================================================

Objetivo: agrupar pacientes clinicamente similares usando KMeans. Dado um
paciente novo (ver src/inferencia.py), o sistema indica a qual grupo ele e
mais parecido.

IMPORTANTE: isto NAO e classificacao. A coluna DEATH_EVENT NUNCA entra no
treino do modelo; ela e usada apenas DEPOIS, para caracterizar (descrever) os
grupos encontrados - calculando a taxa de obito por cluster.

Este script: carrega os dados, faz o pre-processamento, escolhe o melhor k,
treina o KMeans, avalia, gera os graficos/artefatos e salva o pipeline para a
inferencia.

Aluno 1 - estilo metodico e didatico.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend nao-interativo: precisa vir ANTES do pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Reprodutibilidade obrigatoria em todo o projeto.
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Caminhos do projeto (declarados UMA vez; nunca recalculados nas funcoes)
# ---------------------------------------------------------------------------
# Centralizar aqui evita recomputar parents[1] em cada funcao (DRY) e mantem
# todos os artefatos saindo da MESMA pasta, sem divergencia silenciosa.
DIR_PROJETO: Path = Path(__file__).resolve().parents[1]
DIR_DADOS: Path = DIR_PROJETO / "data"
DIR_SAIDA: Path = DIR_PROJETO / "outputs"
ARQUIVO_DADOS: str = "heart_failure_clinical_records_dataset.csv"

# ---------------------------------------------------------------------------
# Definicao das colunas (decisao de projeto justificada no README/debate)
# ---------------------------------------------------------------------------
# Coluna-alvo que fica FORA do treino e so caracteriza os grupos depois.
COLUNA_ALVO = "DEATH_EVENT"

# Binarias (0/1): NAO sao escaladas. Em uma binaria 0/1 a distancia euclidiana
# entre as duas categorias ja vale exatamente 1; aplicar StandardScaler iria
# distorcer essa distancia e inflar artificialmente o peso de categorias raras.
# Por isso vao como 'passthrough' no ColumnTransformer.
COLUNAS_BINARIAS = ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking"]

# Continuas: escaladas com StandardScaler (media 0, desvio 1) para que nenhuma
# variavel domine a distancia euclidiana so por causa da sua unidade/escala.
# 'time' = tempo de acompanhamento (dias). E incluido como continua escalada
# (decisao justificada no debate).
COLUNAS_CONTINUAS = [
    "age",
    "creatinine_phosphokinase",
    "ejection_fraction",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
    "time",
]


def banner(titulo: str) -> None:
    """Imprime um banner de secao padronizado."""
    print("=" * 60)
    print(titulo)
    print("=" * 60)


def carregar_dados() -> pd.DataFrame:
    """Carrega o CSV de pacientes com insuficiencia cardiaca."""
    caminho = DIR_DADOS / ARQUIVO_DADOS
    df = pd.read_csv(caminho)
    return df


def construir_preprocessador() -> ColumnTransformer:
    """Monta o ColumnTransformer, deixando as binarias como passthrough para nao inflar o peso de categorias raras na distancia euclidiana."""
    preprocessador = ColumnTransformer(
        transformers=[
            ("continuas", StandardScaler(), COLUNAS_CONTINUAS),
            ("binarias", "passthrough", COLUNAS_BINARIAS),
        ]
    )
    return preprocessador


def escolher_melhor_k(
    X_treino: pd.DataFrame,
    preprocessador: ColumnTransformer,
    k_min: int = 2,
    k_max: int = 8,
) -> tuple[int, dict[int, float]]:
    """
    Testa k=2..8 e escolhe o k com melhor silhouette.

    Para cada k, ajusta um pipeline (preprocessamento + KMeans) e calcula o
    silhouette no espaco JA transformado (o mesmo espaco onde o KMeans mede
    distancias). Retorna o melhor k e o dicionario k -> silhouette.
    """
    # Transforma uma unica vez; o espaco e o mesmo para todos os k. Usamos um
    # clone descartavel: a varredura de k NAO deve deixar o preprocessador
    # recebido ja fitado (o pipeline final re-ajusta o seu proprio, fit so no
    # treino, sem reaproveitar transformer ja fitado).
    X_transformado = clone(preprocessador).fit_transform(X_treino)

    silhouette_por_k: dict[int, float] = {}
    for k in range(k_min, k_max + 1):
        modelo = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        rotulos = modelo.fit_predict(X_transformado)
        sil = float(silhouette_score(X_transformado, rotulos))
        silhouette_por_k[k] = sil
        print(f"  k={k}: silhouette={sil:.4f}")

    melhor_k = max(silhouette_por_k, key=silhouette_por_k.get)
    return melhor_k, silhouette_por_k


def plotar_silhouette_por_k(
    silhouette_por_k: dict[int, float], melhor_k: int, caminho_saida: Path
) -> None:
    """Gera o grafico silhouette x k, destacando o k escolhido."""
    ks = list(silhouette_por_k.keys())
    sils = list(silhouette_por_k.values())

    melhor_sil = silhouette_por_k[melhor_k]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, sils, marker="o", color="#1f77b4", label="Silhouette")
    ax.axvline(
        melhor_k,
        color="#d62728",
        linestyle="--",
        label=f"melhor k = {melhor_k}",
    )
    # Anota o valor vencedor ao lado do ponto: torna o grafico autossuficiente
    # como evidencia da escolha de k (numero formatado em pt-BR).
    ax.annotate(
        f"{melhor_sil:.4f}".replace(".", ","),
        xy=(melhor_k, melhor_sil),
        xytext=(8, 8),
        textcoords="offset points",
        color="#d62728",
        fontweight="bold",
    )
    ax.set_title("Coeficiente de Silhouette por numero de clusters (k)")
    ax.set_xlabel("k (numero de clusters)")
    ax.set_ylabel("Silhouette medio (quanto MAIOR, melhor)")
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=120)
    plt.close(fig)


def plotar_clusters_pca(
    pipeline: Pipeline, X_treino: pd.DataFrame, caminho_saida: Path
) -> None:
    """
    Projeta os dados em 2D via PCA (apenas para visualizacao) e colore por cluster.

    O PCA e usado SO para desenhar; o KMeans foi treinado no espaco completo.
    """
    preprocessador: ColumnTransformer = pipeline.named_steps["preprocessador"]
    modelo: KMeans = pipeline.named_steps["modelo"]

    X_transformado = preprocessador.transform(X_treino)
    rotulos = modelo.predict(X_transformado)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_transformado)
    var_explicada = pca.explained_variance_ratio_ * 100

    # Projeta os centroides no MESMO espaco PCA: torna visivel a regra
    # "centroide mais proximo" que a inferencia executa via predict.
    centroides = pca.transform(modelo.cluster_centers_)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        x=coords[:, 0],
        y=coords[:, 1],
        hue=rotulos,
        palette="tab10",
        s=40,
        edgecolor="k",
        linewidth=0.3,
        ax=ax,
    )
    ax.scatter(
        centroides[:, 0],
        centroides[:, 1],
        marker="X",
        s=200,
        c="black",
        edgecolor="white",
        linewidths=1.0,
        label="centroide",
        zorder=5,
    )
    ax.set_title("Clusters projetados em 2D (PCA, apenas visualizacao)")
    ax.set_xlabel(f"Componente 1 ({var_explicada[0]:.1f}% da variancia)")
    ax.set_ylabel(f"Componente 2 ({var_explicada[1]:.1f}% da variancia)")
    ax.legend(title="Cluster")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=120)
    plt.close(fig)


def gerar_perfil_clusters(
    df_completo: pd.DataFrame,
    rotulos: np.ndarray,
    caminho_csv: Path,
    caminho_txt: Path,
) -> pd.DataFrame:
    """
    Calcula o perfil de cada cluster: media das variaveis + taxa de DEATH_EVENT.

    DEATH_EVENT entra AQUI (caracterizacao), nunca no treino.
    """
    df_perfil = df_completo.copy()
    df_perfil["cluster"] = rotulos

    # Medias por cluster de TODAS as variaveis (inclui DEATH_EVENT como taxa).
    perfil = df_perfil.groupby("cluster").mean(numeric_only=True)
    perfil["n_pacientes"] = df_perfil.groupby("cluster").size()
    # Renomeia para deixar explicito que e a taxa de obito do grupo.
    perfil = perfil.rename(columns={COLUNA_ALVO: "taxa_death_event"})

    perfil.to_csv(caminho_csv, encoding="utf-8")

    # Versao .txt legivel para leitura rapida.
    with caminho_txt.open("w", encoding="utf-8") as f:
        f.write("PERFIL DOS CLUSTERS (medias por grupo)\n")
        f.write("=" * 60 + "\n")
        f.write("Obs.: taxa_death_event = fracao de obitos no grupo (caracterizacao).\n")
        f.write("DEATH_EVENT NAO foi usado no treino do clustering.\n\n")
        f.write(perfil.to_string())
        f.write("\n")

    return perfil


def plotar_taxa_obito_por_cluster(
    taxa_por_cluster: dict[int, float],
    n_por_cluster: dict[int, float],
    caminho_saida: Path,
) -> None:
    """Plota a taxa de obito ja calculada por cluster (caracterizacao, nao treino)."""
    clusters = sorted(taxa_por_cluster)
    taxas = [taxa_por_cluster[c] for c in clusters]

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x=clusters, y=taxas, hue=clusters, palette="Blues", legend=False, ax=ax)

    # Anota o percentual (pt-BR) e o n de pacientes no topo de cada barra, para
    # o grafico ser autossuficiente como evidencia da caracterizacao.
    for c, taxa in zip(clusters, taxas):
        rotulo = f"{taxa * 100:.1f}%".replace(".", ",")
        ax.annotate(
            f"{rotulo}\n(n={int(n_por_cluster[c])})",
            xy=(c, taxa),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_title(
        "Taxa de obito por cluster (caracterizacao - DEATH_EVENT fora do treino)"
    )
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Taxa de obito")
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=120)
    plt.close(fig)


def main() -> None:
    DIR_SAIDA.mkdir(parents=True, exist_ok=True)

    banner("Q1 - HEART FAILURE | CLUSTERING COM KMeans")

    # -----------------------------------------------------------------------
    # 1) Carregamento dos dados
    # -----------------------------------------------------------------------
    df = carregar_dados()
    print(f"Dataset carregado: {df.shape[0]} pacientes, {df.shape[1]} colunas.")
    print(f"Valores ausentes no dataset: {int(df.isna().sum().sum())}")

    # -----------------------------------------------------------------------
    # 2) Separacao do alvo (fora do treino) e das features
    # -----------------------------------------------------------------------
    banner("PRE-PROCESSAMENTO")
    # X_treino = tudo MENOS DEATH_EVENT. y_alvo guarda DEATH_EVENT so para depois.
    X_treino = df.drop(columns=[COLUNA_ALVO]).copy()
    y_alvo = df[COLUNA_ALVO].copy()
    print(f"Coluna-alvo '{COLUNA_ALVO}' REMOVIDA do treino (uso so na caracterizacao).")
    print(f"Continuas (StandardScaler): {COLUNAS_CONTINUAS}")
    print(f"Binarias  (passthrough)   : {COLUNAS_BINARIAS}")

    preprocessador = construir_preprocessador()

    # -----------------------------------------------------------------------
    # 3) Escolha do melhor k (k=2..8) pelo silhouette
    # -----------------------------------------------------------------------
    banner("ESCOLHA DO NUMERO DE CLUSTERS (k)")
    melhor_k, silhouette_por_k = escolher_melhor_k(X_treino, preprocessador)
    print(f"\nMelhor k pelo silhouette: k={melhor_k} "
          f"(silhouette={silhouette_por_k[melhor_k]:.4f})")

    plotar_silhouette_por_k(
        silhouette_por_k, melhor_k, DIR_SAIDA / "silhouette_por_k.png"
    )
    print("Grafico salvo: outputs/silhouette_por_k.png")

    # -----------------------------------------------------------------------
    # 4) Treino do pipeline final com o melhor k
    # -----------------------------------------------------------------------
    banner("TREINO DO MODELO FINAL")
    # Pipeline = pre-processamento + KMeans. Tudo aprendido aqui (fit) fica
    # encapsulado; a inferencia apenas transforma e prediz (sem vazamento).
    pipeline = Pipeline(
        steps=[
            ("preprocessador", construir_preprocessador()),
            (
                "modelo",
                KMeans(n_clusters=melhor_k, n_init=10, random_state=RANDOM_STATE),
            ),
        ]
    )
    rotulos = pipeline.fit_predict(X_treino)
    print(f"KMeans treinado com k={melhor_k}.")

    # -----------------------------------------------------------------------
    # 5) Avaliacao (metricas internas de clustering)
    # -----------------------------------------------------------------------
    banner("AVALIACAO DO CLUSTERING")
    X_transformado = pipeline.named_steps["preprocessador"].transform(X_treino)
    sil = float(silhouette_score(X_transformado, rotulos))
    db = float(davies_bouldin_score(X_transformado, rotulos))
    ch = float(calinski_harabasz_score(X_transformado, rotulos))
    print(f"Silhouette        : {sil:.4f} (quanto MAIOR, melhor)")
    print(f"Davies-Bouldin    : {db:.4f} (quanto MENOR, melhor)")
    print(f"Calinski-Harabasz : {ch:.2f} (quanto MAIOR, melhor)")

    # Contagem por cluster.
    valores, contagens = np.unique(rotulos, return_counts=True)
    n_por_cluster = {int(c): int(n) for c, n in zip(valores, contagens)}
    print(f"Pacientes por cluster: {n_por_cluster}")

    # -----------------------------------------------------------------------
    # 6) Caracterizacao dos grupos com DEATH_EVENT (so agora!)
    # -----------------------------------------------------------------------
    banner("CARACTERIZACAO DOS GRUPOS (DEATH_EVENT)")
    perfil = gerar_perfil_clusters(
        df,
        rotulos,
        DIR_SAIDA / "perfil_clusters.csv",
        DIR_SAIDA / "perfil_clusters.txt",
    )
    taxa_death_por_cluster = {
        int(c): float(t) for c, t in perfil["taxa_death_event"].items()
    }
    for c in sorted(taxa_death_por_cluster):
        print(
            f"  Cluster {c}: {n_por_cluster[c]:>3} pacientes | "
            f"taxa de obito = {taxa_death_por_cluster[c]*100:.1f}%"
        )
    print("Perfis salvos: outputs/perfil_clusters.csv e .txt")

    # Grafico da taxa de obito por grupo: plota numeros JA calculados (nao
    # recalcula nada) e e o "minimo finalistico" que comprova a caracterizacao.
    plotar_taxa_obito_por_cluster(
        taxa_death_por_cluster,
        n_por_cluster,
        DIR_SAIDA / "taxa_obito_por_cluster.png",
    )
    print("Grafico salvo: outputs/taxa_obito_por_cluster.png")

    # -----------------------------------------------------------------------
    # 7) Visualizacao 2D via PCA
    # -----------------------------------------------------------------------
    plotar_clusters_pca(pipeline, X_treino, DIR_SAIDA / "clusters_pca.png")
    print("Grafico salvo: outputs/clusters_pca.png")

    # -----------------------------------------------------------------------
    # 8) Metricas em JSON
    # -----------------------------------------------------------------------
    # Bloco por cluster autoexplicativo: junta n_pacientes + taxa_death_event +
    # um rotulo de risco (o cluster com MAIOR taxa de obito e o "maior risco").
    # Apenas reorganiza numeros JA calculados - nada e recomputado.
    cluster_maior_risco = max(taxa_death_por_cluster, key=taxa_death_por_cluster.get)
    clusters = {
        str(c): {
            "n_pacientes": n_por_cluster[c],
            "taxa_death_event": taxa_death_por_cluster[c],
            "rotulo_risco": (
                "maior risco" if c == cluster_maior_risco else "menor risco"
            ),
        }
        for c in sorted(taxa_death_por_cluster)
    }
    metricas = {
        "descricao": (
            "Clustering KMeans nao supervisionado; DEATH_EVENT usado apenas "
            "para caracterizar grupos (fora do treino)."
        ),
        "k_escolhido": int(melhor_k),
        "silhouette": sil,
        "davies_bouldin": db,
        "calinski_harabasz": ch,
        "n_por_cluster": n_por_cluster,
        "taxa_death_event_por_cluster": taxa_death_por_cluster,
        "clusters": clusters,
        "silhouette_por_k": {int(k): float(v) for k, v in silhouette_por_k.items()},
        "colunas_continuas": COLUNAS_CONTINUAS,
        "colunas_binarias": COLUNAS_BINARIAS,
        "random_state": RANDOM_STATE,
    }
    caminho_metricas = DIR_SAIDA / "metricas.json"
    with caminho_metricas.open("w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)
    print("Metricas salvas: outputs/metricas.json")

    # -----------------------------------------------------------------------
    # 9) Persistencia do pipeline treinado (para a inferencia)
    # -----------------------------------------------------------------------
    banner("PERSISTENCIA DO MODELO")
    caminho_modelo = DIR_SAIDA / "pipeline_kmeans.joblib"
    dump(pipeline, caminho_modelo)
    print(f"Pipeline salvo: {caminho_modelo.relative_to(DIR_PROJETO)}")

    # Salva tambem a ordem das colunas de entrada, util para a inferencia
    # montar o paciente novo na ordem correta.
    caminho_colunas = DIR_SAIDA / "colunas_entrada.json"
    with caminho_colunas.open("w", encoding="utf-8") as f:
        json.dump(list(X_treino.columns), f, indent=2, ensure_ascii=False)
    print("Ordem das colunas salva: outputs/colunas_entrada.json")

    banner("CONCLUIDO")
    print("Execute agora: python src/inferencia.py (demo de inferencia).")


if __name__ == "__main__":
    main()
