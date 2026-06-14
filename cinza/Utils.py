from scipy.stats import t, norm
import pandas as pd


# calcular o p-valor para testes t e z
def calcular_pvalor(valor, gl=None, tipo="!=", dist="t"):

    cdf = (lambda x: t.cdf(x, gl)) if dist == "t" else norm.cdf

    if tipo == ">":
        return 1 - cdf(valor)

    elif tipo == "<":
        return cdf(valor)

    else:
        return 2 * cdf(-abs(valor))


# converter uma tabela
def long_to_wide(base, categoria: str, valor: str, dropna: bool = False
    ):
    """
    Converte dados long format para wide format.

    Exemplo
    --------

    Entrada:

    | Cargo | Salario |
    |--------|----------|
    | Jr     | 3000     |
    | Jr     | 3200     |
    | Pleno  | 5000     |

    Saída:

    | Jr   | Pleno |
    |------|--------|
    | 3000 | 5000   |
    | 3200 | NaN    |
    """

    # DADOS
    dados = base[
        [categoria, valor]
    ].copy()


    # CRIA LISTA POR GRUPO
    grupos = {}

    categorias = (
        dados[categoria]
        .dropna()
        .unique()
    )

    for cat in categorias:
        grupos[cat] = (
            dados[
                dados[categoria] == cat
                ][valor]
            .reset_index(drop=True)
        )

    # CONVERTE PARA DATAFRAME
    resultado = pd.DataFrame(grupos)

    # REMOVE NaN OPCIONALMENTE
    if dropna:
        resultado = resultado.dropna()

    return resultado