"""
Q3 - Black Friday: DEMO de INFERENCIA.
=================================================

Carrega os TRES pipelines treinados (salvos por src/main.py) e, dada uma venda
NOVA, preve os tres alvos de uma vez:
  - product_category (categoria do produto)
  - payment_method   (forma de pagamento)
  - age_group        (faixa etaria)
CADA previsao acompanhada do seu GRAU DE CERTEZA = max(predict_proba).

Sem vazamento: cada pipeline ja encapsula o one-hot e o scaling, ajustados
APENAS no treino. Aqui so montamos a venda nova na ordem correta de colunas e
chamamos predict / predict_proba - nada e re-ajustado. Os alvos NUNCA entram
como entrada.

Aluno 1 - estilo metodico e didatico.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from joblib import load
from sklearn.pipeline import Pipeline

# Caminhos do projeto: calculados UMA vez e tipados (mesmos nomes de main.py),
# derivados de DIR_PROJETO; nunca recalculados dentro das funcoes.
DIR_PROJETO: Path = Path(__file__).resolve().parents[1]
DIR_DADOS: Path = DIR_PROJETO / "data"
DIR_SAIDA: Path = DIR_PROJETO / "outputs"

# Mesma lista de alvos usada no treino (um pipeline por alvo).
ALVOS = ["product_category", "payment_method", "age_group"]

# Rotulos amigaveis para a impressao final.
ROTULO_ALVO = {
    "product_category": "Categoria do produto",
    "payment_method": "Forma de pagamento",
    "age_group": "Faixa etaria",
}

# Dominios das features categoricas, espelhando os DOMINIO_* de main.py. Servem
# para AVISAR quando a venda traz um valor fora do esquema (ver validar_venda):
# o OneHotEncoder(handle_unknown="ignore") trataria esse valor como tudo-zero e
# a previsao sairia sem base, com ~50% de certeza, sem nenhum alerta.
DOMINIO_GENDER = ["M", "F"]
DOMINIO_CITY_CATEGORY = ["A", "B", "C"]


def banner(titulo: str) -> None:
    """Imprime um banner de secao padronizado."""
    print("=" * 60)
    print(titulo)
    print("=" * 60)


def carregar_pipelines() -> dict[str, Pipeline]:
    """Carrega os 3 pipelines treinados salvos em outputs/."""
    pipelines: dict[str, Pipeline] = {}
    for alvo in ALVOS:
        caminho = DIR_SAIDA / f"pipeline_{alvo}.joblib"
        if not caminho.exists():
            raise FileNotFoundError(
                f"Modelo '{caminho.name}' nao encontrado. "
                "Rode antes: python src/main.py"
            )
        pipelines[alvo] = load(caminho)
    return pipelines


def carregar_ordem_colunas() -> list[str]:
    """Carrega a ordem das features de entrada usada no treino."""
    caminho = DIR_SAIDA / "colunas_entrada.json"
    # Guarda de existencia (espelha a de carregar_pipelines): sem isto o open()
    # daria um traceback cru, inconsistente com a mensagem clara dos .joblib.
    if not caminho.exists():
        raise FileNotFoundError(
            f"Esquema de colunas '{caminho.name}' nao encontrado. "
            "Rode antes: python src/main.py"
        )
    with caminho.open("r", encoding="utf-8") as f:
        return json.load(f)


def carregar_metricas() -> dict[str, object]:
    """
    Carrega outputs/metricas.json (para anexar baseline/ressalva as previsoes).

    Tolerante por design: se o arquivo nao existir, devolve um dict vazio - a
    demo de inferencia ainda roda, so deixa de imprimir o baseline do acaso.
    """
    caminho = DIR_SAIDA / "metricas.json"
    if not caminho.exists():
        return {}
    with caminho.open("r", encoding="utf-8") as f:
        return json.load(f)


def criar_venda_nova() -> dict[str, object]:
    """
    Cria uma venda NOVA realista (valores dentro das faixas do esquema).

    Atencao: a venda foi montada A MAO, valor por valor, e NUNCA copiada de uma
    linha de data/black_friday_sintetico.csv - inferencia honesta usa um caso
    inedito. E NAO inclui nenhum dos alvos (product_category, payment_method,
    age_group): eles sao justamente o que o sistema precisa prever.
    """
    venda = {
        "gender": "F",          # M/F
        "occupation": 12,       # int 0..20
        "city_category": "A",   # A/B/C
        "stay_years": 3,        # int 0..4
        "marital_status": 1,    # 0/1
        "purchase_amount": 850.0,  # R$
        "quantity": 2,          # int 1..5
    }
    return venda


def validar_venda(venda: dict[str, object], colunas: list[str]) -> None:
    """
    Valida a venda contra o esquema do treino ANTES de montar o DataFrame.

    Duas guardas contra erro silencioso:
      (a) feature ausente vira excecao com a coluna que falta, em vez de um
          KeyError cru ao indexar [colunas] mais a frente;
      (b) valor categorico fora do dominio (gender nao em M/F; city_category nao
          em A/B/C) vira AVISO - o OneHotEncoder(handle_unknown="ignore") o
          trataria como tudo-zero e a previsao sairia sem base (~50% de certeza),
          sem nenhum alerta. Avisar e mais honesto do que prever no escuro.
    """
    faltando = [c for c in colunas if c not in venda]
    if faltando:
        raise ValueError(
            f"Venda fora do esquema: faltam as features {faltando}. "
            f"Esperado: {colunas}."
        )
    if venda["gender"] not in DOMINIO_GENDER:
        print(
            f"AVISO: gender={venda['gender']!r} fora de {DOMINIO_GENDER}; "
            "sera codificado como tudo-zero e a previsao ficara sem base."
        )
    if venda["city_category"] not in DOMINIO_CITY_CATEGORY:
        print(
            f"AVISO: city_category={venda['city_category']!r} fora de "
            f"{DOMINIO_CITY_CATEGORY}; sera codificado como tudo-zero e a "
            "previsao ficara sem base."
        )


def prever_alvo(
    pipeline: Pipeline, df_venda: pd.DataFrame
) -> tuple[str, float]:
    """
    Preve UM alvo e retorna (classe_prevista, grau_de_certeza).

    O grau de certeza e o max(predict_proba): a probabilidade da classe mais
    provavel segundo o RandomForest.
    """
    classe = str(pipeline.predict(df_venda)[0])
    probas = pipeline.predict_proba(df_venda)[0]
    grau_certeza = float(probas.max())
    return classe, grau_certeza


def main() -> None:
    banner("Q3 - BLACK FRIDAY | DEMO DE INFERENCIA")

    pipelines = carregar_pipelines()
    colunas = carregar_ordem_colunas()
    metricas = carregar_metricas()
    alvos_metricas = metricas.get("alvos", {}) if isinstance(metricas, dict) else {}
    eh_sintetico = bool(metricas.get("dados_sinteticos_placeholder"))

    venda = criar_venda_nova()
    validar_venda(venda, colunas)  # falha cedo/avisa antes de montar o DataFrame
    print("Venda NOVA apresentada ao sistema:")
    for chave, valor in venda.items():
        print(f"  {chave:>16}: {valor}")

    # Monta o DataFrame na MESMA ordem de colunas do treino (evita erro silencioso).
    df_venda = pd.DataFrame([venda])[colunas]

    banner("PREVISOES (com grau de certeza)")
    for alvo in ALVOS:
        classe, grau = prever_alvo(pipelines[alvo], df_venda)
        # Anexa o baseline do acaso (como em main.py), nunca a certeza pelada.
        acaso = alvos_metricas.get(alvo, {}).get("acuracia_acaso_majoritaria")
        sufixo = f" (acaso da classe majoritaria ~ {acaso:.3f})" if acaso else ""
        print(
            f"  {ROTULO_ALVO[alvo]:>22}: {classe:>16}  "
            f"(certeza = {grau * 100:.1f}%){sufixo}"
        )

    print("\nObs.: o grau de certeza vem de max(predict_proba) de cada modelo.")
    if eh_sintetico:
        # Sobre dados sinteticos a certeza e score RELATIVO, nao probabilidade
        # calibrada (ver outputs/debate_Q3.md): nao se calibra sobre placeholder.
        print(
            "Ressalva: dados sinteticos (placeholder) - a certeza e um score "
            "RELATIVO, nao uma probabilidade calibrada."
        )
    banner("CONCLUIDO")


if __name__ == "__main__":
    main()
