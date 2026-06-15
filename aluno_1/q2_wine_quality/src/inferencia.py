"""
Q2 - Wine Quality | Modulo de INFERENCIA.
=================================================

Carrega o pipeline vencedor salvo por src/main.py (outputs/modelo_vencedor.joblib)
e preve a 'quality' de um vinho NOVO (amostra fabricada, fora do dataset), junto
com a probabilidade da classe prevista.

Nao ha vazamento de dados: o pipeline ja foi ajustado (fit) SOMENTE no conjunto
de treino dentro de main.py. Aqui apenas usamos .predict / .predict_proba; nada
e re-ajustado e o alvo 'quality' NUNCA entra como entrada.

Uso:
    python src/inferencia.py
(o exemplo abaixo demonstra a previsao de um vinho de exemplo)

Aluno 1 - estilo metodico e didatico.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from joblib import load

DIR_PROJETO: Path = Path(__file__).resolve().parents[1]
DIR_SAIDA: Path = DIR_PROJETO / "outputs"
CAMINHO_MODELO: Path = DIR_SAIDA / "modelo_vencedor.joblib"

# Dominio da feature 'tipo' (so estes dois valores existem no treino).
DOMINIO_TIPO: dict[int, str] = {0: "tinto", 1: "branco"}


def banner(titulo: str) -> None:
    """Imprime um banner de secao padronizado."""
    print("=" * 60)
    print(titulo)
    print("=" * 60)


def carregar_artefato() -> dict[str, object]:
    """Carrega o pipeline vencedor e os metadados das colunas."""
    if not CAMINHO_MODELO.exists():
        raise FileNotFoundError(
            f"Modelo nao encontrado em {CAMINHO_MODELO}. "
            "Execute antes: python src/main.py"
        )
    return load(CAMINHO_MODELO)


def validar_entrada(vinho_novo: dict[str, float], colunas: list[str]) -> None:
    """Valida o contrato de entrada ANTES de prever (evita erro silencioso).

    Sem esta checagem, uma chave ausente (ex.: 'tipo') ou um nome digitado
    errado (ex.: 'PH' em vez de 'pH') viraria NaN em pd.DataFrame([...]), e o
    modelo AINDA preveria — uma previsao errada e silenciosa.
    """
    faltando = set(colunas) - set(vinho_novo)
    se_sobra = set(vinho_novo) - set(colunas)
    if faltando or se_sobra:
        raise ValueError(
            "Entrada incompativel com as colunas do treino. "
            f"Faltando: {sorted(faltando)} | Sobrando: {sorted(se_sobra)}"
        )

    # 'tipo' so admite 0 (tinto) ou 1 (branco); qualquer outro valor e invalido.
    tipo = vinho_novo["tipo"]
    if tipo not in DOMINIO_TIPO:
        raise ValueError(
            f"tipo deve ser 0=tinto ou 1=branco; recebido: {tipo}"
        )


def prever(
    vinho_novo: dict[str, float], artefato: dict[str, object]
) -> tuple[int, float]:
    """Preve a quality de um vinho novo a partir do artefato ja carregado.

    Parametros
    ----------
    vinho_novo : dict[str, float]
        Caracteristicas fisico-quimicas + 'tipo' (tinto=0 / branco=1).
        Deve conter exatamente as colunas usadas no treino.
    artefato : dict[str, object]
        Pipeline vencedor + metadados, ja desserializado em main().

    Retorna
    -------
    (quality_previsto, probabilidade) : tuple[int, float]
        A classe prevista e a probabilidade associada a ela.
    """
    pipeline = artefato["pipeline"]
    colunas = artefato["colunas"]

    validar_entrada(vinho_novo, colunas)

    # Garante a MESMA ordem de colunas usada no treino (sem vazamento).
    entrada = pd.DataFrame([vinho_novo], columns=colunas)

    # Rede de seguranca: nenhuma coluna pode chegar como NaN ao modelo.
    if entrada.isna().any().any():
        colunas_nan = entrada.columns[entrada.isna().any()].tolist()
        raise ValueError(f"Valores ausentes (NaN) nas colunas: {colunas_nan}")

    quality_previsto = int(pipeline.predict(entrada)[0])

    # Probabilidade da classe prevista.
    probabilidades = pipeline.predict_proba(entrada)[0]
    classes = list(pipeline.classes_)
    indice = classes.index(quality_previsto)
    probabilidade = float(probabilidades[indice])

    return quality_previsto, probabilidade


def main() -> None:
    banner("Q2 - WINE QUALITY | INFERENCIA EM VINHO NOVO")

    artefato = carregar_artefato()
    print(f"Modelo carregado: {artefato['nome_modelo']}")
    print(f"Colunas esperadas: {artefato['colunas']}")
    print()

    # Vinho de EXEMPLO: valores fisico-quimicos plausiveis de um vinho branco,
    # porem FABRICADOS a mao (nao copiados de nenhuma linha dos CSVs tinto/branco)
    # para representar de fato uma amostra NOVA, nunca vista no treino.
    vinho_exemplo: dict[str, float] = {
        "fixed acidity": 6.8,
        "volatile acidity": 0.31,
        "citric acid": 0.34,
        "residual sugar": 6.1,
        "chlorides": 0.047,
        "free sulfur dioxide": 28.0,
        "total sulfur dioxide": 122.0,
        "density": 0.9952,
        "pH": 3.21,
        "sulphates": 0.50,
        "alcohol": 10.6,
        "tipo": 1,  # branco
    }

    print(f"Vinho de entrada ({DOMINIO_TIPO[vinho_exemplo['tipo']]}):")
    for chave, valor in vinho_exemplo.items():
        print(f"  {chave:<22}: {valor}")
    print()

    quality_previsto, probabilidade = prever(vinho_exemplo, artefato)

    banner("RESULTADO DA INFERENCIA")
    print(f"Tipo             : {DOMINIO_TIPO[vinho_exemplo['tipo']]}")
    print(f"Quality previsto : {quality_previsto}")
    print(
        f"Probabilidade    : {probabilidade:.4f} ({probabilidade * 100:.2f}%) | "
        "acaso ~ 0,14 (1/7 classes); quanto MAIOR, melhor"
    )
    print(
        "Obs.: as classes 3 e 9 quase nao sao aprendidas (poucas amostras), "
        "logo notas extremas dificilmente serao previstas."
    )


if __name__ == "__main__":
    main()
