"""
Q2 - Wine Quality: CLASSIFICACAO MULTICLASSE FINA da nota (3 a 9).
=================================================

Objetivo: prever a 'quality' (nota inteira de 3 a 9) de vinhos a partir de suas
caracteristicas fisico-quimicas, unindo os conjuntos de vinho tinto e branco.

IMPORTANTE: isto e classificacao multiclasse FINA da nota inteira (3 a 9), NAO
regressao nem agrupamento em faixas; o alvo 'quality' NUNCA entra como feature.
Sem vazamento: o StandardScaler e ajustado (fit) SOMENTE no treino, dentro do
Pipeline.

Fluxo (metodico e didatico):
    1. Carregar os dois CSV (tinto e branco) e uni-los preservando as colunas.
    2. Adicionar a feature 'tipo' (tinto=0 / branco=1).
    3. Split treino/teste ESTRATIFICADO (random_state=42).
    4. Avaliar 3 classificadores em Pipeline:
       RandomForest, GradientBoosting e KNN.
    5. Selecionar o vencedor por F1-macro (desempate por acuracia).
    6. Reportar acuracia global, acuracia por classe (matriz de confusao),
       F1-macro e F1-weighted.
    7. Salvar o pipeline vencedor (joblib) e os artefatos em outputs/.

Este modulo TREINA, AVALIA e SALVA. A inferencia fica em src/inferencia.py.

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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# Configuracoes globais
# --------------------------------------------------------------------------- #
RANDOM_STATE: int = 42  # Reprodutibilidade obrigatoria

# Nomes de colunas usados em pontos quentes (evita strings literais repetidas).
COLUNA_ALVO: str = "quality"
COLUNA_TIPO: str = "tipo"

# Hiperparametros que definem as metricas (centralizados, sem alterar os VALORES).
N_ESTIMADORES_RF: int = 200
N_VIZINHOS_KNN: int = 15
TEST_SIZE: float = 0.20

# Caminhos RELATIVOS ao diretorio do projeto (pathlib).
DIR_PROJETO: Path = Path(__file__).resolve().parents[1]
DIR_DADOS: Path = DIR_PROJETO / "data"
DIR_SAIDA: Path = DIR_PROJETO / "outputs"


def banner(titulo: str) -> None:
    """Imprime um banner de secao padronizado."""
    print("=" * 60)
    print(titulo)
    print("=" * 60)


def carregar_dados() -> pd.DataFrame:
    """Carrega e une os conjuntos de vinho tinto e branco.

    A coluna 'tipo' identifica a origem (tinto=0 / branco=1). Os nomes das
    colunas fisico-quimicas sao preservados exatamente como no arquivo.
    """
    caminho_tinto = DIR_DADOS / "winequality-red.csv"
    caminho_branco = DIR_DADOS / "winequality-white.csv"

    tinto = pd.read_csv(caminho_tinto, sep=";").copy()
    branco = pd.read_csv(caminho_branco, sep=";").copy()

    # Feature 'tipo': permite ao modelo distinguir tinto de branco.
    tinto[COLUNA_TIPO] = 0
    branco[COLUNA_TIPO] = 1

    dados = pd.concat([tinto, branco], ignore_index=True)

    # NOTA (sem vazamento): o conjunto unido tem 1177 linhas fisico-quimicas
    # totalmente duplicadas. NAO as removemos aqui para preservar as metricas de
    # referencia desta entrega; o efeito (otimismo por memorizacao de ~27% do
    # teste) esta documentado no README e no debate_Q2.md como limitacao conhecida.
    n_duplicadas = int(dados.duplicated().sum())

    print(f"Vinho tinto : {tinto.shape[0]} amostras")
    print(f"Vinho branco: {branco.shape[0]} amostras")
    print(f"Conjunto unido: {dados.shape[0]} amostras x {dados.shape[1]} colunas")
    print(f"Linhas totalmente duplicadas (mantidas, ver README): {n_duplicadas}")
    print()
    print("Distribuicao do alvo 'quality' (classes desbalanceadas):")
    print(dados[COLUNA_ALVO].value_counts().sort_index().to_string())
    print()
    return dados


def construir_modelos() -> dict[str, Pipeline]:
    """Define os 3 classificadores, cada um dentro de um Pipeline.

    O StandardScaler e incluido sempre: e indispensavel para o KNN (baseado em
    distancia) e inofensivo para os modelos de arvore, mantendo o pipeline
    uniforme e sem vazamento (o fit do scaler ocorre apenas no treino).
    """
    modelos: dict[str, Pipeline] = {
        "RandomForest": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "modelo",
                    RandomForestClassifier(
                        n_estimators=N_ESTIMADORES_RF,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "GradientBoosting": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "modelo",
                    GradientBoostingClassifier(random_state=RANDOM_STATE),
                ),
            ]
        ),
        "KNN": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                "modelo",
                KNeighborsClassifier(
                    n_neighbors=N_VIZINHOS_KNN, weights="distance"
                ),
            ),
            ]
        ),
    }
    return modelos


Metricas = dict[str, object]


def acuracia_por_classe(
    y_verdadeiro: pd.Series, y_predito: np.ndarray, classes: list[int]
) -> dict[str, float]:
    """Calcula a acuracia por classe a partir da matriz de confusao.

    Para cada classe, acuracia = acertos da classe / total de amostras da
    classe (equivale ao recall por classe lido na diagonal da matriz).
    """
    matriz = confusion_matrix(y_verdadeiro, y_predito, labels=classes)
    resultado: dict[str, float] = {}
    for i, classe in enumerate(classes):
        total_classe = matriz[i, :].sum()
        acertos = matriz[i, i]
        resultado[str(classe)] = (
            float(acertos / total_classe) if total_classe > 0 else 0.0
        )
    return resultado


def avaliar_modelo(
    nome: str,
    pipeline: Pipeline,
    X_treino: pd.DataFrame,
    X_teste: pd.DataFrame,
    y_treino: pd.Series,
    y_teste: pd.Series,
    classes: list[int],
) -> Metricas:
    """Treina um pipeline e devolve suas metricas no conjunto de teste."""
    pipeline.fit(X_treino, y_treino)
    y_predito = pipeline.predict(X_teste)

    # Acaso = prever sempre a classe majoritaria (6), so para comparar a acuracia.
    acaso = float(y_teste.value_counts(normalize=True).max())

    acuracia = float(accuracy_score(y_teste, y_predito))
    f1_macro = float(
        f1_score(
            y_teste, y_predito, labels=classes, average="macro", zero_division=0
        )
    )
    f1_weighted = float(
        f1_score(
            y_teste, y_predito, labels=classes, average="weighted", zero_division=0
        )
    )
    acc_classe = acuracia_por_classe(y_teste, y_predito, classes)

    print(f"[{nome}]")
    print(f"  Acuracia global : {acuracia:.4f} (acaso ~ {acaso:.4f} = classe majoritaria 6)")
    print(f"  F1-macro        : {f1_macro:.4f} (quanto MAIOR, melhor; trata as 7 classes igual)")
    print(f"  F1-weighted     : {f1_weighted:.4f} (quanto MAIOR, melhor)")
    print()

    return {
        "nome": nome,
        "acuracia_global": acuracia,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "acuracia_por_classe": acc_classe,
    }


def salvar_matriz_confusao(
    nome_vencedor: str,
    pipeline: Pipeline,
    X_teste: pd.DataFrame,
    y_teste: pd.Series,
    classes: list[int],
    acuracia: float,
    f1_macro: float,
) -> Path:
    """Gera e salva a matriz de confusao do modelo vencedor em PNG."""
    y_predito = pipeline.predict(X_teste)
    matriz = confusion_matrix(y_teste, y_predito, labels=classes)

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        matriz,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={"label": "Numero de amostras"},
        ax=ax,
    )
    # O titulo carrega a ressalva metodologica: as classes 3 e 9 sao quase
    # inaprendiveis (poucas amostras), logo a acuracia global engana sem o F1-macro.
    ax.set_title(
        f"Matriz de Confusao - {nome_vencedor}\n"
        f"(acuracia global = {acuracia:.2f} | F1-macro = {f1_macro:.2f}; "
        "classes 3 e 9 quase sem suporte)"
    )
    ax.set_xlabel("Classe PREVISTA (quality)")
    ax.set_ylabel("Classe VERDADEIRA (quality)")

    fig.tight_layout()
    caminho = DIR_SAIDA / f"matriz_confusao_{nome_vencedor}.png"
    fig.savefig(caminho, dpi=120)
    plt.close(fig)
    return caminho


def main() -> None:
    """Orquestra o fluxo: carrega, treina os 3 modelos, seleciona por F1-macro e
    persiste os artefatos."""
    DIR_SAIDA.mkdir(parents=True, exist_ok=True)

    banner("Q2 - WINE QUALITY | CARGA E UNIAO DOS DADOS")
    dados = carregar_dados()

    # Separacao features / alvo.
    X = dados.drop(columns=[COLUNA_ALVO]).copy()
    y = dados[COLUNA_ALVO].copy()
    classes = sorted(y.unique().tolist())

    banner("SPLIT TREINO/TESTE ESTRATIFICADO (80/20)")
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,  # mantem a proporcao das classes raras (3 e 9)
        random_state=RANDOM_STATE,
    )
    print(f"Treino: {X_treino.shape[0]} amostras | Teste: {X_teste.shape[0]} amostras")
    print(f"Classes presentes: {classes}")
    print()

    banner("AVALIACAO DOS 3 CLASSIFICADORES")
    modelos = construir_modelos()
    resultados: list[Metricas] = []
    pipelines_treinados: dict[str, Pipeline] = {}

    for nome, pipeline in modelos.items():
        metricas = avaliar_modelo(
            nome, pipeline, X_treino, X_teste, y_treino, y_teste, classes
        )
        resultados.append(metricas)
        pipelines_treinados[nome] = pipeline

    # ----------------------------------------------------------------------- #
    # Selecao do vencedor: F1-macro (desempate por acuracia global).
    # F1-macro trata todas as classes com o mesmo peso, evitando que as classes
    # dominantes (5 e 6) mascarem o desempenho nas classes raras.
    # ----------------------------------------------------------------------- #
    banner("SELECAO DO MODELO VENCEDOR (F1-macro)")
    vencedor = max(
        resultados, key=lambda r: (r["f1_macro"], r["acuracia_global"])
    )
    nome_vencedor = vencedor["nome"]
    pipeline_vencedor = pipelines_treinados[nome_vencedor]

    for r in sorted(resultados, key=lambda r: r["f1_macro"], reverse=True):
        marca = "  <== VENCEDOR" if r["nome"] == nome_vencedor else ""
        print(
            f"  {r['nome']:<18} F1-macro={r['f1_macro']:.4f} "
            f"acuracia={r['acuracia_global']:.4f}{marca}"
        )
    print()
    print(f"Modelo escolhido para implantacao: {nome_vencedor}")
    print()

    banner("METRICAS DETALHADAS DO VENCEDOR")
    acaso = float(y_teste.value_counts(normalize=True).max())
    print(f"Acuracia global : {vencedor['acuracia_global']:.4f} (acaso ~ {acaso:.4f} = classe majoritaria 6)")
    print(f"F1-macro        : {vencedor['f1_macro']:.4f} (quanto MAIOR, melhor; trata as 7 classes igual)")
    print(f"F1-weighted     : {vencedor['f1_weighted']:.4f} (quanto MAIOR, melhor)")
    print("Acuracia por classe (diagonal da matriz de confusao):")
    for classe, acc in vencedor["acuracia_por_classe"].items():
        print(f"  quality {classe}: {acc:.4f}")
    print()

    # ----------------------------------------------------------------------- #
    # Salvar artefatos.
    # ----------------------------------------------------------------------- #
    banner("SALVANDO ARTEFATOS EM outputs/")

    # 1) Comparacao dos 3 modelos.
    caminho_comparacao = DIR_SAIDA / "comparacao_modelos.json"
    with caminho_comparacao.open("w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    print(f"Comparacao dos modelos -> {caminho_comparacao.name}")

    # 2) Metricas do vencedor.
    metricas_vencedor = {
        "modelo_vencedor": nome_vencedor,
        "criterio_selecao": "F1-macro (desempate por acuracia global)",
        "acuracia_global": vencedor["acuracia_global"],
        "acuracia_acaso_majoritaria": acaso,
        "f1_macro": vencedor["f1_macro"],
        "f1_weighted": vencedor["f1_weighted"],
        "acuracia_por_classe": vencedor["acuracia_por_classe"],
    }
    caminho_metricas = DIR_SAIDA / "metricas.json"
    with caminho_metricas.open("w", encoding="utf-8") as f:
        json.dump(metricas_vencedor, f, indent=2, ensure_ascii=False)
    print(f"Metricas do vencedor   -> {caminho_metricas.name}")

    # 3) Matriz de confusao do vencedor (PNG).
    caminho_matriz = salvar_matriz_confusao(
        nome_vencedor,
        pipeline_vencedor,
        X_teste,
        y_teste,
        classes,
        float(vencedor["acuracia_global"]),
        float(vencedor["f1_macro"]),
    )
    print(f"Matriz de confusao     -> {caminho_matriz.name}")

    # 4) Pipeline vencedor (joblib) + metadados das colunas para a inferencia.
    artefato = {
        "pipeline": pipeline_vencedor,
        "colunas": list(X.columns),
        "classes": classes,
        "nome_modelo": nome_vencedor,
    }
    caminho_modelo = DIR_SAIDA / "modelo_vencedor.joblib"
    # compress=3: reduz drasticamente o tamanho do .joblib (RandomForest comprime
    # muito bem) sem alterar as previsoes nem as metricas.
    dump(artefato, caminho_modelo, compress=3)
    print(f"Pipeline vencedor      -> {caminho_modelo.name}")
    print()

    banner("CONCLUIDO")
    print(f"Vencedor: {nome_vencedor} | "
          f"acuracia={vencedor['acuracia_global']:.4f} | "
          f"F1-macro={vencedor['f1_macro']:.4f} | "
          f"F1-weighted={vencedor['f1_weighted']:.4f}")
    print("Execute agora: python src/inferencia.py (demo de inferencia).")


if __name__ == "__main__":
    main()
