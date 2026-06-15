"""
Q1 - HEART FAILURE: DEMO DE INFERENCIA
=================================================

Carrega o pipeline KMeans treinado (salvo por src/main.py) e, dado um paciente
NOVO e realista, indica a qual grupo (cluster) ele e mais similar.

Sem vazamento: o StandardScaler foi ajustado (`fit`) UMA vez sobre a coorte
conhecida (os 299 pacientes; nao ha split por ser nao supervisionado), dentro
do pipeline. Aqui o paciente novo e SOMENTE transformado (sem refit) e
atribuido a um grupo com predict - nenhuma estatistica dele influencia o
pre-processamento; nada e re-ajustado.

Aluno 1 - estilo metodico e didatico.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Caminhos do projeto (declarados UMA vez; espelham o main.py para uniformidade)
# ---------------------------------------------------------------------------
DIR_PROJETO: Path = Path(__file__).resolve().parents[1]
DIR_DADOS: Path = DIR_PROJETO / "data"
DIR_SAIDA: Path = DIR_PROJETO / "outputs"

# ---------------------------------------------------------------------------
# Faixas clinicas plausiveis (limites LARGOS) para validar o paciente novo
# ---------------------------------------------------------------------------
# Fora destas faixas o paciente nao e clinicamente realista; checar aqui evita
# erro silencioso (ex.: age=-10 ou time=-50 hoje passariam despercebidos).
FAIXAS_PLAUSIVEIS: dict[str, tuple[float, float]] = {
    "age": (18.0, 110.0),
    "creatinine_phosphokinase": (10.0, 10000.0),
    "ejection_fraction": (5.0, 90.0),
    "platelets": (10000.0, 1000000.0),
    "serum_creatinine": (0.1, 15.0),
    "serum_sodium": (110.0, 160.0),
    "time": (0.0, 400.0),
}
COLUNAS_BINARIAS = ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking"]


def banner(titulo: str) -> None:
    """Imprime um banner de secao padronizado."""
    print("=" * 60)
    print(titulo)
    print("=" * 60)


def carregar_pipeline() -> Pipeline:
    """Carrega o pipeline treinado salvo em outputs/."""
    caminho = DIR_SAIDA / "pipeline_kmeans.joblib"
    if not caminho.exists():
        raise FileNotFoundError(
            "Modelo nao encontrado. Rode antes: python src/main.py"
        )
    return load(caminho)


def carregar_ordem_colunas() -> list[str]:
    """Carrega a ordem das colunas de entrada usada no treino."""
    caminho = DIR_SAIDA / "colunas_entrada.json"
    with caminho.open("r", encoding="utf-8") as f:
        return json.load(f)


def carregar_perfil_clusters() -> pd.DataFrame:
    """Carrega o perfil dos clusters (para descrever o grupo do paciente)."""
    caminho = DIR_SAIDA / "perfil_clusters.csv"
    return pd.read_csv(caminho, index_col="cluster")


def criar_paciente_novo() -> dict[str, float]:
    """
    Cria um paciente NOVO realista (valores dentro das faixas clinicas do dataset).

    Atencao: NAO inclui DEATH_EVENT - isso seria justamente o que o sistema de
    agrupamento nao conhece sobre um paciente desconhecido.
    """
    paciente = {
        "age": 60.0,
        "anaemia": 0,             # binaria 0/1
        "creatinine_phosphokinase": 250,
        "diabetes": 1,            # binaria 0/1
        "ejection_fraction": 35,
        "high_blood_pressure": 1, # binaria 0/1
        "platelets": 262000.0,
        "serum_creatinine": 1.2,
        "serum_sodium": 137,
        "sex": 1,                 # binaria 0/1
        "smoking": 0,             # binaria 0/1
        "time": 115,              # tempo de acompanhamento (dias)
    }
    return paciente


def validar_paciente(paciente: dict[str, float], colunas: list[str]) -> None:
    """
    Valida o paciente novo ANTES de chamar o pipeline, com mensagens nomeadas.

    Sem isto, colunas faltantes viram KeyError cru e NaN vira "Input X contains
    NaN" generico do sklearn; nomear a variavel troca o erro silencioso por uma
    mensagem que diz exatamente o que corrigir.
    """
    faltantes = set(colunas) - set(paciente)
    if faltantes:
        raise ValueError(
            "Faltam variaveis no paciente: " + ", ".join(sorted(faltantes))
        )

    for variavel in colunas:
        valor = paciente[variavel]
        if valor is None or (isinstance(valor, float) and np.isnan(valor)):
            raise ValueError(f"Variavel '{variavel}' esta vazia (None/NaN).")
        # Binarias so admitem 0 ou 1; escalar ou agrupar valores fora disso nao
        # faz sentido clinico e mascararia um erro de digitacao.
        if variavel in COLUNAS_BINARIAS and valor not in (0, 1):
            raise ValueError(
                f"Variavel binaria '{variavel}' deve ser 0 ou 1 (recebido: {valor})."
            )
        # Continuas fora da faixa plausivel: paciente nao e clinicamente realista.
        if variavel in FAIXAS_PLAUSIVEIS:
            minimo, maximo = FAIXAS_PLAUSIVEIS[variavel]
            if not (minimo <= valor <= maximo):
                raise ValueError(
                    f"Variavel '{variavel}'={valor} fora da faixa plausivel "
                    f"[{minimo}, {maximo}]."
                )


def prever_grupo(pipeline: Pipeline, paciente: dict[str, float], colunas: list[str]) -> int:
    """Monta o DataFrame na ordem correta, transforma e prediz o cluster."""
    # Garante a MESMA ordem de colunas do treino (evita erro silencioso).
    df_paciente = pd.DataFrame([paciente])[colunas]
    cluster = int(pipeline.predict(df_paciente)[0])
    return cluster


def calcular_certeza(
    pipeline: Pipeline, paciente: dict[str, float], colunas: list[str]
) -> tuple[int, np.ndarray]:
    """
    Retorna o cluster previsto e as distancias euclidianas a TODOS os centroides.

    O KMeans do pipeline expoe `transform`, que da a distancia do ponto a cada
    centroide no espaco onde ele mede; usamos isso para reportar o grau de
    certeza (sem refit, leitura deterministica do pipeline ja salvo).
    """
    df_paciente = pd.DataFrame([paciente])[colunas]
    distancias = pipeline.transform(df_paciente)[0]
    cluster = int(np.argmin(distancias))
    return cluster, distancias


def main() -> None:
    banner("Q1 - HEART FAILURE | DEMO DE INFERENCIA")

    # Envolvemos o fluxo num try/except para mostrar uma mensagem amigavel em
    # vez de um traceback cru no terminal do avaliador (evita erro silencioso).
    try:
        pipeline = carregar_pipeline()
        colunas = carregar_ordem_colunas()
        perfil = carregar_perfil_clusters()

        paciente = criar_paciente_novo()
        print("Paciente NOVO (desconhecido) apresentado ao sistema:")
        for chave, valor in paciente.items():
            print(f"  {chave:>26}: {valor}")

        # Valida ANTES de prever: troca KeyError/NaN cru por mensagem nomeada.
        validar_paciente(paciente, colunas)

        banner("RESULTADO DO AGRUPAMENTO")
        cluster, distancias = calcular_certeza(pipeline, paciente, colunas)
        print(f"Paciente atribuido ao CLUSTER {cluster}.")

        # Grau de certeza: distancia ao centroide do grupo + margem para o 2o
        # mais proximo. Nunca imprimimos o numero pelado, sempre com a direcao.
        ordenadas = np.sort(distancias)
        dist_grupo = float(ordenadas[0])
        dist_segundo = float(ordenadas[1])
        razao = dist_segundo / dist_grupo
        print(
            f"  {'Distancia ao centroide do grupo':>34}: {dist_grupo:.4f} "
            "(quanto MENOR, mais tipico do grupo)"
        )
        print(
            f"  {'Margem p/ o 2o grupo mais proximo':>34}: {dist_segundo:.4f} "
            f"(razao {razao:.2f}; >1 = atribuicao confiante)"
        )

        # Descreve o grupo usando o perfil salvo (inclui a taxa de obito do grupo,
        # que serve como caracterizacao - nunca como rotulo de treino).
        if cluster not in perfil.index:
            raise ValueError(
                f"Perfil do cluster {cluster} ausente; "
                "regenere com python src/main.py"
            )
        linha = perfil.loc[cluster]
        n_pacientes = int(linha["n_pacientes"])
        taxa_obito = float(linha["taxa_death_event"])
        print(f"Esse grupo tem {n_pacientes} pacientes de treino.")
        print(f"Taxa de obito (caracterizacao) do grupo: {taxa_obito*100:.1f}%")
        print(
            "Interpretacao: o paciente novo e clinicamente mais parecido com os "
            f"pacientes do cluster {cluster}."
        )

        # Mostra algumas medias do grupo para contextualizar.
        print("\nPerfil medio do grupo (algumas variaveis):")
        for variavel in ["age", "ejection_fraction", "serum_creatinine", "time"]:
            if variavel in linha.index:
                print(f"  media {variavel:>20}: {float(linha[variavel]):.2f}")
    except (FileNotFoundError, ValueError) as erro:
        print(f"ERRO: {erro}")
        sys.exit(1)

    banner("CONCLUIDO")
    print("Veja outputs/perfil_clusters.txt para o perfil completo dos grupos.")


if __name__ == "__main__":
    main()
