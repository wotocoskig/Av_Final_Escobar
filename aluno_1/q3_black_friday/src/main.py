"""
Q3 - Black Friday: MULTI-CLASSIFICACAO com 3 alvos INDEPENDENTES.
=================================================

Objetivo: para uma venda da Black Friday, prever SIMULTANEAMENTE tres coisas,
cada uma com seu proprio GRAU DE CERTEZA (vindo de predict_proba):
  1) product_category  (categoria do produto)
  2) payment_method    (forma de pagamento)
  3) age_group         (faixa etaria do comprador)

Estrategia de modelagem (decisao deste aluno): TRES RandomForestClassifier
INDEPENDENTES, um por alvo. Cada modelo tem seu proprio pipeline com
ColumnTransformer (one-hot nas categoricas, scaling nas numericas) e e treinado
com split ESTRATIFICADO. O grau de certeza de cada previsao e o max(predict_proba).

Sem vazamento: cada alvo so usa as FEATURES como entrada; os outros dois alvos
NUNCA entram como feature. Todo pre-processamento (one-hot, scaling) e aprendido
DENTRO do pipeline, apenas no conjunto de treino.

DADOS: tenta ler data/black_friday.csv. Se nao existir, GERA um dataset
SINTETICO (seed 42) com dependencias probabilisticas features->alvos, salva em
data/black_friday_sintetico.csv e imprime um AVISO BEM VISIVEL de que os numeros
sao PLACEHOLDERS (nao valem como resultado final).

Este script: carrega/gera os dados, treina os 3 modelos, avalia (acuracia global,
acuracia por classe via matriz de confusao, F1, sensibilidade e especificidade
por classe one-vs-rest), gera os graficos/artefatos e salva os 3 pipelines.

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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Reprodutibilidade obrigatoria em todo o projeto.
RANDOM_STATE = 42

# Caminhos do projeto: calculados UMA vez e tipados (nunca recalculados dentro
# das funcoes), porque sao fixos e nunca variam por chamada.
DIR_PROJETO: Path = Path(__file__).resolve().parents[1]
DIR_DADOS: Path = DIR_PROJETO / "data"
DIR_SAIDA: Path = DIR_PROJETO / "outputs"

# ---------------------------------------------------------------------------
# Definicao das colunas (esquema fixo do projeto)
# ---------------------------------------------------------------------------
# Os TRES alvos. Cada um sera previsto por um modelo independente.
ALVOS = ["product_category", "payment_method", "age_group"]

# Features de ENTRADA (as mesmas para os tres modelos). Nenhum alvo entra aqui.
COLUNAS_CATEGORICAS = ["gender", "city_category"]
COLUNAS_NUMERICAS = [
    "occupation",
    "stay_years",
    "marital_status",
    "purchase_amount",
    "quantity",
]
COLUNAS_FEATURES = COLUNAS_CATEGORICAS + COLUNAS_NUMERICAS

# Dominios (categorias possiveis) de cada alvo, conforme o esquema do projeto.
DOMINIO_AGE_GROUP = ["0-17", "18-25", "26-35", "36-45", "46-50", "51-55", "55+"]
DOMINIO_PRODUCT_CATEGORY = [
    "Eletronicos",
    "Roupas",
    "Casa",
    "Alimentos",
    "Beleza",
    "Brinquedos",
    "Esportes",
]
DOMINIO_PAYMENT_METHOD = [
    "Cartao_Credito",
    "Cartao_Debito",
    "PIX",
    "Dinheiro",
    "Boleto",
]

# Mapa alvo -> dominio, agrupado no topo (em vez de reconstruido a cada chamada
# de treinar_e_avaliar_alvo), deixando a funcao focada em treinar/avaliar.
DOMINIO_POR_ALVO: dict[str, list[str]] = {
    "product_category": DOMINIO_PRODUCT_CATEGORY,
    "payment_method": DOMINIO_PAYMENT_METHOD,
    "age_group": DOMINIO_AGE_GROUP,
}

# Rotulos amigaveis em PT para os titulos dos graficos (o NOME do alvo cru segue
# em ingles no nome do arquivo). DUPLICADO de proposito em inferencia.py - a
# persona mantem os 2 arquivos autossuficientes, sem utils compartilhado.
ROTULO_ALVO: dict[str, str] = {
    "product_category": "Categoria do produto",
    "payment_method": "Forma de pagamento",
    "age_group": "Faixa etaria",
}

N_LINHAS_SINTETICO = 4000


def banner(titulo: str) -> None:
    """Imprime um banner de secao padronizado."""
    print("=" * 60)
    print(titulo)
    print("=" * 60)


def aviso_placeholder() -> None:
    """Imprime um AVISO BEM VISIVEL de que os dados sao sinteticos (placeholders)."""
    print("!" * 60)
    print("!!!  ATENCAO: DADOS SINTETICOS (PLACEHOLDER)             !!!")
    print("!!!  O arquivo data/black_friday.csv NAO foi encontrado. !!!")
    print("!!!  Foi gerado um dataset SINTETICO (seed 42) APENAS    !!!")
    print("!!!  para provar que o pipeline roda de ponta a ponta.   !!!")
    print("!!!  Os numeros NAO valem como resultado final / real.   !!!")
    print("!" * 60)


# ---------------------------------------------------------------------------
# Geracao do dataset sintetico (so quando o CSV real nao existe)
# ---------------------------------------------------------------------------
def gerar_dataset_sintetico(n_linhas: int = N_LINHAS_SINTETICO) -> pd.DataFrame:
    """
    Gera um dataset sintetico com DEPENDENCIAS probabilisticas features->alvos.

    As dependencias sao deliberadas para que os modelos fiquem ACIMA do acaso:
      - age_group     depende de occupation e marital_status;
      - payment_method depende de city_category e purchase_amount;
      - product_category depende de gender e age_group.

    Usa um unico Generator com seed 42 (reprodutibilidade total).
    """
    rng = np.random.default_rng(RANDOM_STATE)

    # --- Features independentes (distribuicoes simples e realistas) ---
    gender = rng.choice(["M", "F"], size=n_linhas, p=[0.55, 0.45])
    occupation = rng.integers(0, 21, size=n_linhas)  # 0..20
    city_category = rng.choice(["A", "B", "C"], size=n_linhas, p=[0.3, 0.4, 0.3])
    stay_years = rng.integers(0, 5, size=n_linhas)  # 0..4
    marital_status = rng.integers(0, 2, size=n_linhas)  # 0/1
    # purchase_amount: valor de compra (R$), lognormal para ter cauda a direita.
    purchase_amount = np.round(rng.lognormal(mean=6.0, sigma=0.6, size=n_linhas), 2)
    quantity = rng.integers(1, 6, size=n_linhas)  # 1..5

    df = pd.DataFrame(
        {
            "gender": gender,
            "occupation": occupation,
            "city_category": city_category,
            "stay_years": stay_years,
            "marital_status": marital_status,
            "purchase_amount": purchase_amount,
            "quantity": quantity,
        }
    )

    # -----------------------------------------------------------------------
    # ALVO 1: age_group depende de occupation e marital_status
    # -----------------------------------------------------------------------
    # Ideia: occupation baixa -> mais jovem; occupation alta -> mais velho.
    # marital_status=1 (casado) empurra para faixas mais adultas. A gaussiana
    # estreita (sigma=0.9) concentra a massa perto do centro, deixando o sinal
    # forte o suficiente para o modelo ficar bem acima do acaso.
    age_group = np.empty(n_linhas, dtype=object)
    n_idades = len(DOMINIO_AGE_GROUP)
    idx_idade = np.arange(n_idades)
    for i in range(n_linhas):
        score = occupation[i] / 20.0 + 0.30 * marital_status[i]  # 0 .. ~1.30
        score = min(score, 1.0)
        # Centro da distribuicao etaria desliza com o score.
        centro = score * (n_idades - 1)
        pesos = np.exp(-0.5 * ((idx_idade - centro) / 0.9) ** 2)  # gaussiana discreta
        pesos = pesos / pesos.sum()
        age_group[i] = rng.choice(DOMINIO_AGE_GROUP, p=pesos)

    df["age_group"] = age_group

    # -----------------------------------------------------------------------
    # ALVO 2: payment_method depende de city_category e purchase_amount
    # -----------------------------------------------------------------------
    # Ideia: cidade A (grande) usa mais cartao de credito; cidade C usa mais
    # dinheiro/boleto. Compras altas puxam para cartao de credito; baixas, para
    # PIX/dinheiro.
    base_por_cidade = {
        # ordem: Cartao_Credito, Cartao_Debito, PIX, Dinheiro, Boleto
        "A": np.array([0.50, 0.20, 0.16, 0.07, 0.07]),
        "B": np.array([0.28, 0.24, 0.27, 0.11, 0.10]),
        "C": np.array([0.12, 0.16, 0.30, 0.27, 0.15]),
    }
    limiar_alto = float(np.quantile(purchase_amount, 0.66))
    limiar_baixo = float(np.quantile(purchase_amount, 0.33))

    payment_method = np.empty(n_linhas, dtype=object)
    for i in range(n_linhas):
        pesos = base_por_cidade[city_category[i]].copy()
        if purchase_amount[i] >= limiar_alto:
            pesos[0] += 0.30  # mais Cartao_Credito
            pesos[3] -= 0.12  # menos Dinheiro
            pesos[2] -= 0.12  # menos PIX
        elif purchase_amount[i] <= limiar_baixo:
            pesos[2] += 0.22  # mais PIX
            pesos[3] += 0.12  # mais Dinheiro
            pesos[0] -= 0.22  # menos Cartao_Credito
        pesos = np.clip(pesos, 0.01, None)
        pesos = pesos / pesos.sum()
        payment_method[i] = rng.choice(DOMINIO_PAYMENT_METHOD, p=pesos)

    df["payment_method"] = payment_method

    # -----------------------------------------------------------------------
    # ALVO 3: product_category depende de gender e age_group
    # -----------------------------------------------------------------------
    # Ideia: genero e faixa etaria deslocam as preferencias de categoria.
    # Importante: como o modelo de product_category enxerga SOMENTE as features
    # (e nao o rotulo age_group), reforcamos tambem a dependencia direta de
    # occupation e quantity - que o modelo CONSEGUE observar - para que o sinal
    # seja aprendivel e fique acima do acaso. Os pesos sao bem contrastados.
    # ordem: Eletronicos, Roupas, Casa, Alimentos, Beleza, Brinquedos, Esportes
    # gender e o driver DOMINANTE e diretamente observavel pelo modelo: M e F
    # tem preferencias bem separadas (Esportes/Eletronicos vs Roupas/Beleza),
    # o que garante acuracia acima do acaso mesmo sem ver age_group.
    base_genero = {
        "M": np.array([0.26, 0.04, 0.08, 0.08, 0.02, 0.06, 0.46]),
        "F": np.array([0.06, 0.34, 0.10, 0.08, 0.36, 0.04, 0.02]),
    }
    # Ajustes por faixa etaria (somados ao vetor base e depois normalizados),
    # mais sutis para nao apagar a separacao por genero.
    ajuste_idade = {
        "0-17": np.array([0.06, 0.00, -0.04, -0.03, 0.00, 0.25, 0.00]),
        "18-25": np.array([0.10, 0.08, -0.04, -0.03, 0.06, 0.00, 0.06]),
        "26-35": np.array([0.06, 0.05, 0.03, 0.00, 0.05, -0.02, 0.03]),
        "36-45": np.array([0.00, 0.00, 0.10, 0.06, 0.00, 0.02, -0.03]),
        "46-50": np.array([-0.04, -0.04, 0.14, 0.10, -0.01, -0.03, -0.04]),
        "51-55": np.array([-0.05, -0.05, 0.16, 0.12, -0.02, -0.04, -0.05]),
        "55+": np.array([-0.05, -0.05, 0.14, 0.20, -0.03, -0.05, -0.05]),
    }
    product_category = np.empty(n_linhas, dtype=object)
    for i in range(n_linhas):
        pesos = base_genero[gender[i]] + ajuste_idade[age_group[i]]
        # Dependencia direta e observavel: occupation alta -> mais Casa/Esportes;
        # quantity alta -> mais Alimentos/Casa (compra de abastecimento).
        if occupation[i] >= 14:
            pesos[2] += 0.06  # Casa
            pesos[6] += 0.05  # Esportes
        if quantity[i] >= 4:
            pesos[3] += 0.08  # Alimentos
            pesos[2] += 0.04  # Casa
        pesos = np.clip(pesos, 0.01, None)
        pesos = pesos / pesos.sum()
        product_category[i] = rng.choice(DOMINIO_PRODUCT_CATEGORY, p=pesos)

    df["product_category"] = product_category

    # Reordena colunas: features primeiro, alvos depois (legibilidade).
    df = df[COLUNAS_FEATURES + ALVOS].copy()
    return df


def carregar_ou_gerar_dados() -> tuple[pd.DataFrame, bool]:
    """
    Carrega data/black_friday.csv se existir; caso contrario, gera o sintetico.

    Retorna (df, usou_sintetico). Quando gera o sintetico, salva uma copia em
    data/black_friday_sintetico.csv e sinaliza usou_sintetico=True.
    """
    caminho_real = DIR_DADOS / "black_friday.csv"
    if caminho_real.exists():
        df = pd.read_csv(caminho_real)
        print(f"Dataset REAL carregado: {caminho_real}")
        return df, False

    # Nao existe CSV real -> gera sintetico (placeholder).
    df = gerar_dataset_sintetico()
    DIR_DADOS.mkdir(parents=True, exist_ok=True)
    caminho_sintetico = DIR_DADOS / "black_friday_sintetico.csv"
    df.to_csv(caminho_sintetico, index=False, encoding="utf-8")
    print(f"Dataset SINTETICO gerado e salvo em: {caminho_sintetico}")
    return df, True


def validar_esquema_real(df: pd.DataFrame) -> None:
    """
    Valida o esquema do CSV REAL antes de treinar (falha cedo e legivel).

    Confere que todas as features e alvos esperados estao presentes (levanta
    excecao com a coluna que falta, em vez de um KeyError cru la na frente) e
    AVISA quando um alvo traz rotulos fora do dominio fixado - esses rotulos
    serao descartados pela matriz de confusao (labels=dominio), entao melhor
    expor o problema do que deixar passar mudo. So roda no caminho do CSV real;
    o sintetico ja nasce no esquema correto.
    """
    faltando = [c for c in COLUNAS_FEATURES + ALVOS if c not in df.columns]
    if faltando:
        raise ValueError(
            "CSV real fora do esquema: faltam as colunas "
            f"{faltando}. Esperado: {COLUNAS_FEATURES + ALVOS}."
        )
    for alvo in ALVOS:
        fora = sorted(set(df[alvo].unique()) - set(DOMINIO_POR_ALVO[alvo]))
        if fora:
            print(
                f"AVISO: '{alvo}' traz rotulos fora do dominio {fora}; serao "
                "ignorados na matriz de confusao (avaliados como erro)."
            )


# ---------------------------------------------------------------------------
# Pipeline por alvo
# ---------------------------------------------------------------------------
def construir_preprocessador() -> ColumnTransformer:
    """
    Monta o ColumnTransformer: one-hot nas categoricas, StandardScaler nas
    numericas. As features de entrada sao as MESMAS para os tres alvos.
    """
    preprocessador = ColumnTransformer(
        transformers=[
            (
                "categoricas",
                OneHotEncoder(handle_unknown="ignore"),
                COLUNAS_CATEGORICAS,
            ),
            ("numericas", StandardScaler(), COLUNAS_NUMERICAS),
        ]
    )
    return preprocessador


def construir_pipeline() -> Pipeline:
    """
    Pipeline = pre-processamento + RandomForestClassifier.

    Encapsular tudo no Pipeline garante que one-hot e scaling sejam aprendidos
    apenas no treino (fit) e apenas APLICADOS na inferencia (transform) -> sem
    vazamento.
    """
    pipeline = Pipeline(
        steps=[
            ("preprocessador", construir_preprocessador()),
            (
                "modelo",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return pipeline


# ---------------------------------------------------------------------------
# Metricas one-vs-rest a partir da matriz de confusao
# ---------------------------------------------------------------------------
def metricas_por_classe(
    matriz: np.ndarray, classes: list[str], total: int
) -> dict[str, dict[str, float]]:
    """
    Calcula, para CADA classe (one-vs-rest), a partir da matriz de confusao:
      - sensibilidade (recall) = TP / (TP + FN)
      - especificidade          = TN / (TN + FP)
      - acuracia_da_classe      = (TP + TN) / total   (acuracia binaria ovr)

    A matriz vem no formato sklearn: linhas = classe verdadeira, colunas = prevista.
    'total' e o tamanho REAL do teste (len(y_teste)), nao matriz.sum(): se o CSV
    real trouxer um rotulo fora do dominio fixado em labels=, a matriz descarta
    essa linha e matriz.sum() < len(y_teste), o que subcontaria os TN e
    distorceria a especificidade (evita esse erro silencioso).
    """
    resultado: dict[str, dict[str, float]] = {}
    for i, classe in enumerate(classes):
        tp = matriz[i, i]
        fn = matriz[i, :].sum() - tp  # verdadeiros da classe i previstos errado
        fp = matriz[:, i].sum() - tp  # outros previstos como classe i
        tn = total - tp - fn - fp

        sensibilidade = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        especificidade = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        acuracia_classe = (tp + tn) / total if total > 0 else 0.0

        resultado[classe] = {
            "sensibilidade": float(sensibilidade),
            "especificidade": float(especificidade),
            "acuracia_classe": float(acuracia_classe),
            "suporte": int(tp + fn),
        }
    return resultado


def plotar_matriz_confusao(
    matriz: np.ndarray, classes: list[str], titulo: str, caminho_saida: Path
) -> None:
    """Desenha a matriz de confusao como heatmap anotado e salva em PNG."""
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        matriz,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=True,
        cbar_kws={"label": "Numero de amostras"},  # a colorbar e um eixo de leitura
        ax=ax,
    )
    ax.set_title(titulo)
    ax.set_xlabel("Classe PREVISTA")
    ax.set_ylabel("Classe VERDADEIRA")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(caminho_saida, dpi=120)
    plt.close(fig)


def treinar_e_avaliar_alvo(
    alvo: str,
    df: pd.DataFrame,
    usou_sintetico: bool,
) -> tuple[Pipeline, dict[str, object]]:
    """
    Treina e avalia UM alvo de ponta a ponta.

    Passos: separa X (features) e y (alvo) -> split estratificado -> treina o
    pipeline -> prediz no teste -> calcula matriz de confusao, acuracia global,
    F1 (macro e ponderado) e metricas one-vs-rest por classe -> salva o heatmap.

    Retorna o pipeline treinado e o dicionario de metricas do alvo.
    """
    banner(f"TREINO + AVALIACAO | {alvo.upper()}")

    X = df[COLUNAS_FEATURES].copy()
    y = df[alvo].copy()

    # Ordem de classes FIXA (dominio do esquema), apenas as presentes nos dados.
    dominio = DOMINIO_POR_ALVO[alvo]
    classes = [c for c in dominio if c in set(y.unique())]

    # Split ESTRATIFICADO: preserva a proporcao das classes do alvo.
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"Treino: {len(X_treino)} linhas | Teste: {len(X_teste)} linhas")
    print(f"Classes ({len(classes)}): {classes}")

    pipeline = construir_pipeline()
    pipeline.fit(X_treino, y_treino)

    y_pred = pipeline.predict(X_teste)

    # Matriz de confusao com ordem de classes fixada.
    matriz = confusion_matrix(y_teste, y_pred, labels=classes)

    acuracia_global = float(accuracy_score(y_teste, y_pred))
    f1_macro = float(f1_score(y_teste, y_pred, labels=classes, average="macro"))
    f1_ponderado = float(
        f1_score(y_teste, y_pred, labels=classes, average="weighted")
    )

    por_classe = metricas_por_classe(matriz, classes, len(y_teste))

    # Acaso (baseline) = prever sempre a classe majoritaria, so para comparar.
    acaso = float(y_teste.value_counts(normalize=True).max())

    print(f"Acuracia GLOBAL : {acuracia_global:.4f} (acaso ~ {acaso:.4f})")
    print(f"F1 macro        : {f1_macro:.4f}")
    print(f"F1 ponderado    : {f1_ponderado:.4f}")
    print("Metricas por classe (sensibilidade / especificidade / acuracia ovr):")
    for classe in classes:
        m = por_classe[classe]
        print(
            f"  {classe:>14}: sens={m['sensibilidade']:.3f} | "
            f"espec={m['especificidade']:.3f} | "
            f"acc_ovr={m['acuracia_classe']:.3f} | n={m['suporte']}"
        )

    # Heatmap da matriz de confusao. O titulo usa o rotulo em PT (o NOME do alvo
    # cru, em ingles, fica so no nome do arquivo) e carrega a acuracia global
    # como ressalva metodologica.
    titulo = (
        f"Matriz de confusao - {ROTULO_ALVO[alvo]}\n"
        f"(acuracia global = {acuracia_global:.3f})"
    )
    if usou_sintetico:
        # O PNG e o unico artefato que circula SOZINHO (slide/relatorio), longe
        # do banner "!" e do README: grava o aviso no proprio grafico.
        titulo += "\nDADOS SINTETICOS - PLACEHOLDER"
    caminho_png = DIR_SAIDA / f"matriz_confusao_{alvo}.png"
    plotar_matriz_confusao(matriz, classes, titulo, caminho_png)
    print(f"Grafico salvo: outputs/{caminho_png.name}")

    # Arredonda SO na serializacao das metricas one-vs-rest por classe (nao no
    # calculo), espelhando o :.3f dos prints: evita gravar caudas de 17 digitos
    # (ex.: 0.14285714285714285) que so geram ruido e diffs instaveis no JSON.
    por_classe_arredondado = {
        classe: {
            "sensibilidade": round(m["sensibilidade"], 4),
            "especificidade": round(m["especificidade"], 4),
            "acuracia_classe": round(m["acuracia_classe"], 4),
            "suporte": m["suporte"],
        }
        for classe, m in por_classe.items()
    }
    metricas_alvo = {
        "classes": classes,
        "acuracia_global": acuracia_global,
        "acuracia_acaso_majoritaria": acaso,
        "f1_macro": f1_macro,
        "f1_ponderado": f1_ponderado,
        "matriz_confusao": matriz.tolist(),
        "por_classe": por_classe_arredondado,
        "n_treino": int(len(X_treino)),
        "n_teste": int(len(X_teste)),
    }
    return pipeline, metricas_alvo


def main() -> None:
    DIR_SAIDA.mkdir(exist_ok=True)

    banner("Q3 - BLACK FRIDAY | MULTI-CLASSIFICACAO (3 ALVOS)")

    # -----------------------------------------------------------------------
    # 1) Carregamento ou geracao dos dados
    # -----------------------------------------------------------------------
    df, usou_sintetico = carregar_ou_gerar_dados()
    if usou_sintetico:
        aviso_placeholder()
    else:
        # So o CSV real pode chegar fora do esquema; o sintetico ja nasce certo.
        validar_esquema_real(df)
    print(f"Dataset: {df.shape[0]} vendas, {df.shape[1]} colunas.")
    print(f"Valores ausentes: {int(df.isna().sum().sum())}")
    print(f"Features de entrada: {COLUNAS_FEATURES}")
    print(f"Alvos a prever     : {ALVOS}")

    # -----------------------------------------------------------------------
    # 2) Treino + avaliacao de CADA alvo (3 modelos independentes)
    # -----------------------------------------------------------------------
    pipelines: dict[str, Pipeline] = {}
    metricas: dict[str, dict[str, object]] = {}
    for alvo in ALVOS:
        pipeline, metricas_alvo = treinar_e_avaliar_alvo(alvo, df, usou_sintetico)
        pipelines[alvo] = pipeline
        metricas[alvo] = metricas_alvo

    # -----------------------------------------------------------------------
    # 3) Persistencia dos 3 pipelines (para a inferencia)
    # -----------------------------------------------------------------------
    banner("PERSISTENCIA DOS MODELOS")
    for alvo, pipeline in pipelines.items():
        caminho_modelo = DIR_SAIDA / f"pipeline_{alvo}.joblib"
        # compress=3: reduz drasticamente o tamanho do .joblib (RandomForest
        # comprime muito bem) sem alterar as previsoes nem as metricas.
        dump(pipeline, caminho_modelo, compress=3)
        print(f"Pipeline salvo: outputs/{caminho_modelo.name}")

    # Ordem das features de entrada, util para a inferencia montar a venda nova.
    caminho_colunas = DIR_SAIDA / "colunas_entrada.json"
    with caminho_colunas.open("w", encoding="utf-8") as f:
        json.dump(COLUNAS_FEATURES, f, indent=2, ensure_ascii=False)
    print("Ordem das features salva: outputs/colunas_entrada.json")

    # -----------------------------------------------------------------------
    # 4) Metricas consolidadas em JSON
    # -----------------------------------------------------------------------
    saida_metricas = {
        "dados_sinteticos_placeholder": bool(usou_sintetico),
        "random_state": RANDOM_STATE,
        "n_linhas_dataset": int(df.shape[0]),
        "features": COLUNAS_FEATURES,
        "alvos": metricas,
    }
    caminho_metricas = DIR_SAIDA / "metricas.json"
    with caminho_metricas.open("w", encoding="utf-8") as f:
        json.dump(saida_metricas, f, indent=2, ensure_ascii=False)
    print("Metricas salvas: outputs/metricas.json")

    if usou_sintetico:
        aviso_placeholder()

    banner("CONCLUIDO")
    print("Execute agora: python src/inferencia.py (demo de inferencia).")


if __name__ == "__main__":
    main()
