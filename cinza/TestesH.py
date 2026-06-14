import math
import numpy as np
import pandas as pd
from .Utils import *
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.libqsturng import psturng
import statsmodels.api as sm
from itertools import combinations
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import (
    normal_ad,
    het_breuschpagan,
    het_goldfeldquandt,
    het_white
)
from scipy import stats
from scipy.stats import (
    t,
    chisquare,
    pearsonr,
    spearmanr,
    f_oneway,
    chi2_contingency,
    chi2,
    shapiro,
    kstest,
    levene,
    norm,
    binomtest
)
from sklearn.metrics import (
    roc_auc_score,
    roc_curve
)





# DESATIVANDO O FORMATO CIENTIFICO
pd.set_option('display.float_format', lambda x: f'{x:.4f}')


# TESTES DE HIPOTESES


# teste t para uma amostra (média)
def tt_uma_amostra(
        base, amostra1: str, constante: float,
        alfa=0.05, tipo="!="
):
    """
    Teste t de Student para uma amostra

    H0: média = constante + hipotese
    """
    dados1 = base[amostra1]


    # Estatísticas da amostra
    media = np.mean(dados1)

    # Desvio padrão amostral
    dp = np.std(dados1, ddof=1)

    # Tamanho da amostra
    n = len(dados1)

    # Erro padrão
    erro = dp / math.sqrt(n)

    # Estatística t
    t_calc = ((media - constante)) / erro

    # Graus de liberdade
    gl = n - 1

    # P-valor
    p_valor = calcular_pvalor(t_calc, gl=gl, tipo=tipo)

    # Decisão
    rejeitar = p_valor < alfa

    resultado = pd.DataFrame(
        {
            "result": [
                media,
                t_calc,
                gl,
                p_valor,
                alfa,
                rejeitar,
                tipo
            ]
        },
        index=[
            "media_amostral",
            "t",
            "gl",
            "p_valor",
            "alfa",
            "rejeitar_h0",
            "tipo"
        ]
    )
    return resultado


# teste t (de welch) para duas amostras (média)
def tt_duas_amostras_independentes_var_diferentes(
        base, amostra1: str, amostra2: str,
        alfa=0.05, tipo="!="
):
    """ Teste de hipóteses para duas amostras independentes - presumindo variâncias diferentes """
    dados1 = base[amostra1]
    dados2 = base[amostra2]

    media1 = np.mean(dados1)
    media2 = np.mean(dados2)
    dp1 = np.std(dados1, ddof=1)
    dp2 = np.std(dados2, ddof=1)
    n1 = len(dados1)
    n2 = len(dados2)

    # Erro padrão
    erro = math.sqrt((dp1 ** 2 / n1) + (dp2 ** 2 / n2))

    # Estatística t
    t_calc = ((media1 - media2)) / erro

    # Graus de liberdade de Welch
    numerador = ((dp1 ** 2 / n1) + (dp2 ** 2 / n2)) ** 2

    denominador = (
            ((dp1 ** 2 / n1) ** 2 / (n1 - 1)) +
            ((dp2 ** 2 / n2) ** 2 / (n2 - 1))
    )

    gl = numerador / denominador

    # P-valor
    p_valor = calcular_pvalor(t_calc, gl=gl, tipo=tipo)

    # Decisão
    rejeitar = p_valor < alfa

    resultado = pd.DataFrame(
        {
            "result": [
                t_calc,
                gl,
                p_valor,
                alfa,
                rejeitar,
                tipo
            ]
        },
        index=[
            "t",
            "gl",
            "p_valor",
            "alfa",
            "rejeitar_h0",
            "tipo"
        ]
    )
    return resultado


# teste t para duas amostras pareadas (média)
def tt_duas_amostras_pareadas(
        base, amostra1: str, amostra2: str,
        alfa=0.05, tipo="!="
):
    """
    Teste t para duas amostras pareadas (dependentes)

    H0: média das diferenças = hipotese
    """

    dados1 = base[amostra1]
    dados2 = base[amostra2]

    # Verifica tamanhos
    if len(dados1) != len(dados2):
        raise ValueError(
            "As amostras pareadas precisam ter o mesmo tamanho."
        )

    # Diferenças
    diferencas = dados1 - dados2

    # Estatísticas
    media_dif = np.mean(diferencas)

    # ddof=1 -> desvio amostral
    dp_dif = np.std(diferencas, ddof=1)

    n = len(diferencas)

    # Erro padrão
    erro = dp_dif / math.sqrt(n)

    # Estatística t
    t_calc = (media_dif) / erro

    # Graus de liberdade
    gl = n - 1

    # P-valor
    p_valor = calcular_pvalor(t_calc, gl=gl, tipo=tipo)

    # Decisão
    rejeitar = p_valor < alfa

    resultado = pd.DataFrame(
        {
            "result": [
                media_dif,
                t_calc,
                gl,
                p_valor,
                alfa,
                rejeitar,
                tipo
            ]
        },
        index=[
            "media_das_diferencas",
            "t",
            "gl",
            "p_valor",
            "alfa",
            "rejeitar_h0",
            "tipo"
        ]
    )
    return resultado


# teste z para uma amostra (proporção)
def tz_uma_amostra(
        base, amostra1: str, constante: float,
        alfa=0.05, tipo="!="
):
    """
    Teste Z para UMA PROPORÇÃO

    H0: p = constante
    """
    dados1 = base[amostra1]

    # Proporção amostral
    proporcao = np.mean(dados1)

    # Tamanho da amostra
    n = len(dados1)

    # Erro padrão usando H0
    erro = math.sqrt(
        (constante * (1 - constante)) / n
    )

    # Estatística Z
    z_calc = (proporcao - constante) / erro

    # P-valor
    p_valor = calcular_pvalor(z_calc, tipo=tipo, dist="z")

    # Decisão
    rejeitar = p_valor < alfa

    resultado = pd.DataFrame(
        {
            "result": [
                proporcao,
                z_calc,
                p_valor,
                alfa,
                rejeitar,
                tipo
            ]
        },
        index=[
            "proporcao_amostral",
            "z",
            "p_valor",
            "alfa",
            "rejeitar_h0",
            "tipo"
        ]
    )
    return resultado


# teste z para duas amostras (proporção)
def tz_duas_amostras(
        base, amostra1: str, amostra2: str,
        alfa=0.05, tipo="!="
):
    """
    Teste Z para duas proporções

    H0: p1 = p2
    """
    dados1 = base[amostra1]
    dados2 = base[amostra2]

    # Proporções amostrais
    p1 = np.mean(dados1)
    p2 = np.mean(dados2)

    # Tamanhos das amostras
    n1 = len(dados1)
    n2 = len(dados2)

    # Quantidade de sucessos
    x1 = np.sum(dados1)
    x2 = np.sum(dados2)

    # Proporção combinada (pooled)
    p_pool = (x1 + x2) / (n1 + n2)

    # Erro padrão
    erro = math.sqrt(
        p_pool * (1 - p_pool) * ((1 / n1) + (1 / n2))
    )

    # Estatística Z
    z_calc = (p1 - p2) / erro

    # P-valor
    p_valor = calcular_pvalor(z_calc, tipo=tipo, dist="z")

    # Decisão
    rejeitar = p_valor < alfa

    resultado = pd.DataFrame(
        {
            "result": [
                p1,
                p2,
                p_pool,
                z_calc,
                p_valor,
                alfa,
                rejeitar,
                tipo
            ]
        },
        index=[
            "proporcao1",
            "proporcao2",
            "pooled",
            "z",
            "p_valor",
            "alfa",
            "rejeitar_h0",
            "tipo"
        ]
    )
    return resultado


# tamanho da amostra para um teste A/B (proporção)
def tamanho_amostras_proporcao_AB(
        taxa_base: float | pd.Series, taxa_variante: float,
        alfa: float = 0.05, poder: float = 0.80
):
    """
    Calcula o tamanho necessário da amostra
    para teste A/B de proporções independentes
    (estilo G*Power).

    Parameters
    ----------
    taxa_base : float | pd.Series
        Taxa de conversão base.

        Ex:
            0.05 = 5%

    taxa_variante : float
        Taxa esperada da variante B.

        Ex:
            0.0535 = 5.35%

    alfa : float
        Nível de significância.

    poder : float
        Poder estatístico desejado.
    """

    # Caso receba Series binária
    taxa = (
        np.mean(taxa_base)
        if isinstance(taxa_base, pd.Series)
        else taxa_base
    )

    # Validações
    if taxa <= 0 or taxa >= 1:
        raise ValueError(
            "taxa_base deve estar entre 0 e 1."
        )

    if taxa_variante <= 0 or taxa_variante >= 1:
        raise ValueError(
            "taxa_variante deve estar entre 0 e 1."
        )

    # Taxa do grupo B
    taxa_b = taxa_variante

    # Diferença absoluta
    delta = abs(taxa_b - taxa)

    if delta == 0:
        raise ValueError(
            "As taxas não podem ser iguais."
        )

    # Z crítico
    # unilateral
    z_critico = norm.ppf(1 - alfa)

    # Poder estatístico
    beta = 1 - poder
    z_beta = norm.ppf(1 - beta)

    # Variância pooled
    p_pool = (taxa + taxa_b) / 2
    q_pool = 1 - p_pool

    # Fórmula do tamanho amostral
    n = (
                2 *
                ((z_critico + z_beta) ** 2) *
                (p_pool * q_pool)
        ) / (delta ** 2)

    # Arredondamento
    n = math.ceil(n)

    resultado = pd.DataFrame(
        {
            "result": [
                z_critico,
                n,
                n * 2,
                1 - alfa,
                poder
            ]
        },
        index=[
            "z_critico",
            "n_por_grupo",
            "n_total",
            "significancia",
            "poder_teste"
        ]
    )
    return resultado


# tamanho da amostra para um teste A/B (média)
def tamanho_amostra_media_AB(
        desvio_padrao: float, diferenca_minima: float,
        alfa: float = 0.05, poder: float = 0.80, bilateral: bool = True
):
    """
    Calcula o tamanho necessário da amostra
    para comparação entre duas médias independentes.

    Parameters
    ----------
    desvio_padrao : float
        Estimativa do desvio padrão.

    diferenca_minima : float
        Menor diferença que deseja detectar.

    alfa : float
        Nível de significância.

    poder : float
        Poder estatístico desejado.

    bilateral : bool
        Se True usa teste bilateral.
        Se False usa unilateral.
    """

    # Validações
    if desvio_padrao <= 0:
        raise ValueError(
            "desvio_padrao deve ser maior que zero."
        )

    if diferenca_minima <= 0:
        raise ValueError(
            "diferenca_minima deve ser maior que zero."
        )

    # Z crítico
    if bilateral:
        z_critico = norm.ppf(1 - alfa / 2)
    else:
        z_critico = norm.ppf(1 - alfa)

    # Poder
    beta = 1 - poder
    z_beta = norm.ppf(1 - beta)

    # Fórmula
    n = (
                2 *
                ((z_critico + z_beta) ** 2) *
                (desvio_padrao ** 2)
        ) / (diferenca_minima ** 2)

    # Arredondamento
    n = math.ceil(n)

    resultado = pd.DataFrame(
        {
            "result": [
                z_critico,
                n,
                n * 2,
                f"{(1 - alfa) * 100}%",
                f"{poder * 100}%",
                desvio_padrao,
                f"{diferenca_minima * 100}%",
                bilateral
            ]
        },
        index=[
            "z_critico",
            "n_por_grupo",
            "n_total",
            "significancia",
            "poder_teste",
            "desvio_padrao",
            "diferenca_minima",
            "bilateral"
        ]
    )
    return resultado


# tamanho da amostra para um teste A/B pareado (média)
def tamanho_amostra_media_AB_pareada(
        base, desvio_das_diferencas: float | list, diferenca_minima: float,
        alfa: float = 0.05, poder: float = 0.80, bilateral: bool = True
):

    # Caso o usuário passe duas colunas
    if isinstance(desvio_das_diferencas, list):

        if len(desvio_das_diferencas) != 2:
            raise ValueError(
                "Informe exatamente duas colunas para calcular "
                "as diferenças pareadas."
            )

        diferencas = (
                base[desvio_das_diferencas[0]]
                - base[desvio_das_diferencas[1]]
        )

        sd = np.std(diferencas, ddof=1)

    else:
        sd = desvio_das_diferencas

    # Z alfa
    if bilateral:
        z_alfa = norm.ppf(1 - alfa / 2)
    else:
        z_alfa = norm.ppf(1 - alfa)

    # Z beta (poder)
    z_beta = norm.ppf(poder)

    # Fórmula
    n = ((z_alfa + z_beta) * sd / diferenca_minima) ** 2

    resultado = pd.DataFrame(
        {
            "result": [
                math.ceil(n),
                n * 2,
                sd,
                f"{(1 - alfa) * 100}%",
                f"{poder * 100}%",
                z_alfa
            ]
        },
        index=[
            "n_por_grupo",
            "n_total",
            "dp das diferenças",
            "significância",
            "poder",
            "z beta"
        ]
    )
    return resultado


# teste ANOVA 1 fator
def teste_anova_1_fator(
        base, coluna: str | list, target: str | bool = False, alfa: float = 0.05
):
    """
    Realiza ANOVA de 1 fator.

    Modos
    -----

    1) Dados separados em colunas
    --------------------------------
    Cada coluna representa um grupo/categoria.

    Exemplo:
        colunas = ["Loja_A", "Loja_B", "Loja_C"]

    2) Dados empilhados (long format)
    --------------------------------
    Uma coluna contém os grupos/categorias
    e outra contém os valores.

    Exemplo:
        colunas = "Cargo"
        target = "Salario"

    Onde:
        - Cargo = fator
        - Salario = variável dependente
    """

    # MODO 1 -> CADA COLUNA É UM GRUPO
    if not target:

        grupos = []
        nomes_grupos = []

        for col in coluna:
            serie = base[col].dropna()
            if serie.empty:
                raise ValueError(f"A coluna '{col}' não possui dados válidos.")

            grupos.append(serie)
            nomes_grupos.append(col)

    # MODO 2 -> DADOS EMPILHADOS
    else:
        if not isinstance(coluna, str):
            raise ValueError(
                "Quando 'target' for informado, "
                "'coluna' deve conter a coluna categórica. "
            )

        fator = coluna

        dados = base[[fator, target]].dropna()

        nomes_grupos = (
            dados[fator]
            .unique()
            .tolist()
        )

        grupos = []

        for grupo in nomes_grupos:
            valores = (
                dados[dados[fator] == grupo][target]
            )

            grupos.append(valores)

    # ANOVA
    estatistica, p_valor = f_oneway(*grupos)

    # GRAUS DE LIBERDADE
    k = len(grupos)
    n_total = sum(len(g) for g in grupos)

    gl_entre = k - 1
    gl_dentro = n_total - k

    # CONCLUSÃO
    if p_valor <= alfa:
        conclusao = (
            "Rejeitamos H0. "
            "Há diferença significativa entre as médias."
        )
    else:
        conclusao = (
            "Não rejeitamos H0. "
            "Não há evidências de diferença significativa entre as médias."
        )

    # RESULTADO
    resultado = pd.DataFrame(
        {
            "result": [
                estatistica,
                p_valor,
                alfa,
                gl_entre,
                gl_dentro,
                k,
                n_total,
                conclusao
            ]
        },
        index=[
            "Estatística F",
            "p-valor",
            "Alfa",
            "GL Entre Grupos",
            "GL Dentro Grupos",
            "Quantidade de Grupos",
            "Total de Observações",
            "Conclusão"
        ]
    )
    return resultado


# teste ANOVA 2 fator SEM interação
def teste_anova_2_fator(
        base, colunas: list, target: str | bool = False, alfa: float = 0.05
):
    """
    Realiza ANOVA de 2 fatores SEM interação.

    Dados empilhados
    --------------------------------
    Exemplo:

        colunas = ["Area", "Cargo"]
        target = "Salario"

    Onde:
        - Area = fator 1
        - Cargo = fator 2
        - Salario = variável dependente

    """

    if target:
        if len(colunas) != 2:
            raise ValueError(
                "Para ANOVA de 2 fatores, "
                "'colunas' deve conter "
                "2 fatores categóricos."
            )

        fator1 = colunas[0]
        fator2 = colunas[1]

        dados = base[
            [fator1, fator2, target]
        ].dropna()

    else:
        raise ValueError(
            "ANOVA de 2 fatores requer "
            "'target' e dois fatores categóricos. "
            "Use dados no formato long/tidy."
        )

    # MODELO SEM INTERAÇÃO
    formula = (
        f"{target} ~ C({fator1}) + C({fator2})"
    )

    modelo = ols(
        formula,
        data=dados
    ).fit()

    tabela = anova_lm(
        modelo,
        typ=2
    )

    # EXTRAÇÃO DOS RESULTADOS
    f1 = tabela.loc[f"C({fator1})", "F"]
    p1 = tabela.loc[f"C({fator1})", "PR(>F)"]

    f2 = tabela.loc[f"C({fator2})", "F"]
    p2 = tabela.loc[f"C({fator2})", "PR(>F)"]

    gl1 = tabela.loc[f"C({fator1})", "df"]
    gl2 = tabela.loc[f"C({fator2})", "df"]

    gl_residuo = tabela.loc["Residual", "df"]

    # CONCLUSÕES
    if p1 <= alfa:
        conclusao1 = (
            f"Rejeitamos H0 para {fator1}. "
            "Há efeito significativo."
        )
    else:
        conclusao1 = (
            f"Não rejeitamos H0 para {fator1}. "
            "Sem evidências de efeito significativo."
        )

    if p2 <= alfa:
        conclusao2 = (
            f"Rejeitamos H0 para {fator2}. "
            "Há efeito significativo."
        )
    else:
        conclusao2 = (
            f"Não rejeitamos H0 para {fator2}. "
            "Sem evidências de efeito significativo."
        )

    # RESULTADO FINAL
    resultado = pd.DataFrame(
        {
            "F": [f1, f2],
            "p-valor": [p1, p2],
            "GL": [gl1, gl2],
            "Conclusão": [conclusao1, conclusao2]
        },
        index=[
            fator1,
            fator2
        ]
    )

    # Informações adicionais
    resultado.attrs["GL Resíduo"] = gl_residuo
    resultado.attrs["Alfa"] = alfa
    return resultado


# teste ANOVA 2 fator COM interação
def teste_anova_2_fator_interacao(
        base, colunas: list, target: str, alfa: float = 0.05
):
    """
    Realiza ANOVA de 2 fatores COM interação.

    Exemplo
    --------

    colunas = ["Area", "Cargo"]
    target = "Salario"

    Onde:
        - Area = fator 1
        - Cargo = fator 2
        - Salario = variável dependente
    """

    # VALIDAÇÃO
    if len(colunas) != 2:
        raise ValueError(
            "'colunas' deve conter exatamente "
            "2 fatores categóricos."
        )

    fator1 = colunas[0]
    fator2 = colunas[1]

    # DADOS
    dados = base[
        [fator1, fator2, target]
    ].dropna()

    # MODELO COM INTERAÇÃO
    formula = (
        f"{target} ~ C({fator1}) * C({fator2})"
    )

    modelo = ols(
        formula,
        data=dados
    ).fit()

    tabela = anova_lm(
        modelo,
        typ=2
    )

    # EXTRAÇÃO DOS RESULTADOS
    fatores = [
        f"C({fator1})",
        f"C({fator2})",
        f"C({fator1}):C({fator2})"
    ]

    nomes = [
        fator1,
        fator2,
        f"{fator1} ✻ {fator2}"
    ]

    resultados = []

    for nome_real, nome_exibicao in zip(fatores, nomes):

        f_valor = tabela.loc[nome_real, "F"]
        p_valor = tabela.loc[nome_real, "PR(>F)"]
        gl = tabela.loc[nome_real, "df"]

        # CONCLUSÃO
        if p_valor <= alfa:
            conclusao = (
                "Rejeitamos H0. "
                "Há efeito significativo."
            )
        else:
            conclusao = (
                "Não rejeitamos H0. "
                "Sem evidências de efeito significativo."
            )
        resultados.append(
            [
                f_valor,
                p_valor,
                gl,
                conclusao
            ]
        )

    # DATAFRAME FINAL
    resultado = pd.DataFrame(
        resultados,
        columns=[
            "F",
            "p-valor",
            "GL",
            "Conclusão"
        ],
        index=nomes
    )

    # METADADOS
    resultado.attrs["GL Resíduo"] = (
        tabela.loc["Residual", "df"]
    )

    resultado.attrs["Alfa"] = alfa
    resultado.attrs["Formula"] = formula
    return resultado


# comparação pos hoc de Tukey
def comparacao_posHoc_tukey(
        base, colunas: list, target: str,
        alpha: float = 0.05, all: bool = False
):
    """
    Realiza o pós-hoc de Tukey para combinações
    entre fatores.

    Exemplo
    --------

    colunas = ["Area", "Cargo"]
    target = "Salario"

    Retorna:
        X1, X2, Dif Média, Erro-padrão,
        gl, t, pTukey

    Parâmetros
    ----------
    all : bool
        - True:
            Retorna todas as comparações.

        - False:
            Retorna apenas comparações
            NÃO significativas
            (pTukey > alpha).
    """

    # VALIDAÇÃO
    if len(colunas) < 1:
        raise ValueError(
            "'colunas' deve possuir ao menos 1 fator."
        )

    # DADOS
    dados = base[
        colunas + [target]
        ].dropna().copy()

    # CRIA COMBINAÇÃO DOS FATORES
    dados["_grupo"] = (
        dados[colunas]
        .astype(str)
        .agg(" | ".join, axis=1)
    )

    # TUKEY
    tukey = pairwise_tukeyhsd(
        endog=dados[target],
        groups=dados["_grupo"],
        alpha=alpha
    )

    tabela = pd.DataFrame(
        data=tukey._results_table.data[1:],
        columns=tukey._results_table.data[0]
    )

    # EXTRAÇÃO MANUAL
    resultado = pd.DataFrame()

    resultado["X1"] = tabela["group1"]
    resultado["X2"] = tabela["group2"]
    resultado["dif media"] = tabela["meandiff"]


    # ERRO PADRÃO
    mse = tukey.variance

    n = (
        dados.groupby("_grupo")
        .size()
        .mean()
    )
    erro_padrao = np.sqrt(mse / n)

    resultado["erro-padrão"] = erro_padrao

    # GL
    gl = (
            len(dados)
            - dados["_grupo"].nunique()
    )

    resultado["gl"] = gl

    # t
    resultado["t"] = (
            resultado["dif media"]
            / resultado["erro-padrão"]
    )

    # pTukey
    resultado["pTukey"] = tabela["p-adj"]

    # FILTRO
    if not all:
        resultado = (
            resultado[
                resultado["pTukey"] > alpha
                ]
            .reset_index(drop=True)
        )
    return resultado


# teste qui-quadrado (associação de variáveis qualitativas)
def teste_qui_quadrado(
        base, coluna_grupo: str, coluna_categoria: str,
        alpha: float = 0.05, rp_abs_min: float = False
):
    """
    Realiza o teste Qui-Quadrado de independência.

    Parâmetros
    ----------
    coluna_grupo : str
        Primeira variável categórica.

    coluna_categoria : str
        Segunda variável categórica.

    alpha : float
        Nível de significância.

    Retorno
    -------
    {
        "contingencia": DataFrame,
        "resultado": DataFrame
    }
    """

    # TABELA DE CONTINGÊNCIA
    obs = pd.crosstab(
        base[coluna_grupo],
        base[coluna_categoria]
    )

    # QUI-QUADRADO
    x2, p, gl, expected = chi2_contingency(obs)

    expected = pd.DataFrame(
        expected,
        index=obs.index,
        columns=obs.columns
    )

    # RESÍDUOS PADRONIZADOS
    rp = (obs - expected) / np.sqrt(expected)

    # TABELA FINAL
    tabela_final = pd.DataFrame(index=obs.index)

    for col in obs.columns:
        tabela_final[col] = obs[col]

        tabela_final[f"E {col}"] = (
            expected[col]
            .round(2)
        )

        tabela_final[f"RP {col}"] = (
            rp[col]
            .round(2)
        )

    # FILTRO RP MÍNIMO
    if rp_abs_min is not False:
        # pega somente colunas RP
        cols_rp = [
            col for col in tabela_final.columns
            if str(col).startswith("RP ")
        ]

        # mantém linhas onde PELO MENOS um RP atende ao critério
        filtro = (
            tabela_final[cols_rp]
            .abs()
            .ge(rp_abs_min)
            .any(axis=1)
        )

        tabela_final = tabela_final[filtro]

    # RESULTADO DO TESTE
    resultado = pd.DataFrame({
        "result": [
            round(x2, 4),
            gl,
            obs.values.sum(),
            round(p, 6)
        ]
    },
        index=[
            "X²",
            "gl",
            "N",
            "p-valor"
        ])

    return {
        "tabela": tabela_final,
        "resultado": resultado
    }


# teste qui-quadrado de ajustamento
def teste_qui_quadrado_ajustamento(
        base, coluna_categoria: str,  coluna_frequencia: str,
        proporcoes_esperadas: list = None, alpha: float = 0.05
):
    """
    Realiza o teste Qui-Quadrado de Ajustamento.

    Parâmetros
    ----------
    coluna_categoria : str
        Categorias/eventos.

    coluna_frequencia : str
        Frequências absolutas observadas.

    proporcoes_esperadas : list
        Proporções esperadas para cada categoria.

        Exemplo:
        [0.25, 0.25, 0.50]

        Se None:
        assume distribuição uniforme.

    alpha : float
        Nível de significância.

    Retorno
    -------
    {
        "tabela": DataFrame,
        "resultado": DataFrame
    }
    """

    # DADOS OBSERVADOS
    categorias = base[coluna_categoria]
    observado = base[coluna_frequencia]

    n = observado.sum()
    k = len(observado)

    # PROPORÇÕES ESPERADAS
    if proporcoes_esperadas is None:

        proporcoes_esperadas = np.repeat(
            1 / k,
            k
        )

    proporcoes_esperadas = np.array(
        proporcoes_esperadas
    )

    # FREQUÊNCIAS ESPERADAS
    esperado = (
            proporcoes_esperadas * n
    )

    # RESÍDUO PADRONIZADO
    rp = (
            (observado - esperado)
            / np.sqrt(esperado)
    )

    # TESTE QUI-QUADRADO
    x2, p = chisquare(
        f_obs=observado,
        f_exp=esperado
    )

    # GRAUS DE LIBERDADE
    gl = k - 1

    # TABELA DE FREQUÊNCIAS
    frequencias = pd.DataFrame({
        coluna_categoria: categorias,
        "Observado": observado,
        "Esperado": esperado.round(2),
        "Proporção": (
                observado / n
        ).round(4),
        "RP": rp.round(2)
    })

    # RESULTADO
    resultado = pd.DataFrame({
        "resultado": [
            round(x2, 4),
            gl,
            n,
            round(p, 6)
        ]
    },
        index=[
            "X²",
            "gl",
            "N",
            "p-valor"
        ])
    return {
        "tabela": frequencias,
        "resultado": resultado
    }


# teste de correlação de Pearson (paramétrico)
def correlacao_pearson(base: pd.DataFrame, colunas: list[str]):
    """
    Calcula a correlação de Pearson entre múltiplas variáveis.

    Retorna uma tabela no formato:

                            R Pearson   p-value
    idade    | salario        0.9912    0.0009
    idade    | vendas         0.9751    0.0045
    salario  | vendas         0.9823    0.0027
    """

    resultados = {}

    for var1, var2 in combinations(colunas, 2):

        # Remove valores ausentes
        dados = base[[var1, var2]].dropna()

        # Correlação de Pearson
        r, p = pearsonr(dados[var1], dados[var2])

        # Índice formatado
        indice = f"{var1}  |  {var2}"

        resultados[indice] = {
            "R Pearson": r,
            "p-value": p
        }

    resultado = pd.DataFrame(resultados).T

    return resultado


# teste de correlação de Spearman (não paramétrico)
def correlacao_spearman(base: pd.DataFrame, colunas: list[str]):
    """
    Calcula a correlação de Spearman entre múltiplas variáveis.

    Retorna uma tabela no formato:

                             R Spearman   p-value
    idade    | salario          0.9912    0.0009
    idade    | vendas           0.9751    0.0045
    salario  | vendas           0.9823    0.0027
    """

    resultados = {}

    for var1, var2 in combinations(colunas, 2):

        # Remove valores ausentes
        dados = base[[var1, var2]].dropna()

        # Correlação de Spearman
        r, p = spearmanr(dados[var1], dados[var2])

        # Índice formatado
        indice = f"{var1}  |  {var2}"

        resultados[indice] = {
            "Rho Spearman": r,
            "p-value": p
        }

    resultado = pd.DataFrame(resultados).T

    return resultado


# teste do sinal para duas amostras pareadas
def teste_do_sinal(
        base: pd.DataFrame, colunas: list[str],
        alfa: float = 0.05, tipo=">"
):
    """
    Realiza o teste do sinal para duas amostras pareadas.

    Parâmetros
    ----------
    base : DataFrame

    colunas : list[str]
        Lista com exatamente duas colunas [antes, depois]

    alfa : float
        Nível de significância

    Retorno
    -------
    DataFrame:

                     result
    N
    Qtd +
    Qtd -
    Válidos
    p(+)
    p-valor
    """

    if len(colunas) != 2:
        raise ValueError(
            "Informe exatamente duas colunas."
        )

    col1, col2 = colunas

    # DIFERENÇAS
    diff = base[col2] - base[col1]

    # Remove empates (diferença == 0)
    sinais = diff[diff != 0]

    # CONTAGENS
    qtd_pos = (sinais > 0).sum()
    qtd_neg = (sinais < 0).sum()
    n_validos = len(sinais)

    # PROPORÇÃO OBSERVADA
    p_mais = qtd_pos / n_validos if n_validos > 0 else np.nan


    calda = "greater" if tipo == ">" else "less" if tipo == "<" else "two-sided"

    # TESTE BINOMIAL (H0: p = 0.5)
    if n_validos > 0:
        teste = binomtest(
            k=qtd_pos,
            n=n_validos,
            p=0.5,
            alternative=calda
        )
        p_valor = teste.pvalue
    else:
        p_valor = np.nan

    # DECISÃO
    decisao = (
        "Rejeita H0"
        if p_valor < alfa
        else "Não rejeita H0"
    )

    # RESULTADO
    resultado = pd.DataFrame({
        "result": [
            len(diff),     # N total
            qtd_pos,
            qtd_neg,
            n_validos,
            round(p_mais, 4),
            round(p_valor, 6),
            decisao
        ]
    },
        index=[
            "N",
            "Qtd +",
            "Qtd -",
            "Válidos",
            "p(+)",
            "p-valor",
            "Decisão"
        ])

    return resultado


# teste de McNemar
def teste_de_mcNemar(
        base: pd.DataFrame, colunas: list[str],
        alfa: float = 0.05, correcao: bool = False
) -> dict:
    """
    Realiza o teste de McNemar para dados pareados categóricos binários.

    Parâmetros
    ----------
    base : pd.DataFrame
        Base de dados contendo as duas variáveis categóricas.

    colunas : list[str]
        Lista contendo exatamente duas colunas:
        [antes, depois]

    alfa : float
        Nível de significância.
    correcao: bool
        Faz o cálculo com correção de continuidade de Yates
    Retorna
    -------
    dict
        {
            "tabela": tabela_contingencia,
            "resultado": tabela_resultado
        }
    """

    # Validação
    if len(colunas) != 2:
        raise ValueError(
            "Informe exatamente duas colunas: [antes, depois]"
        )

    antes, depois = colunas

    # Remove NA
    dados = base[[antes, depois]].dropna()

    # Tabela de contingência
    contingencia = pd.crosstab(
        dados[antes],
        dados[depois],
        margins=True,
        margins_name="Total"
    )

    # Obtém células discordantes
    # b = antes positivo -> depois negativo
    # c = antes negativo -> depois positivo
    tabela_sem_total = pd.crosstab(
        dados[antes],
        dados[depois]
    )

    if tabela_sem_total.shape != (2, 2):
        raise ValueError(
            "O teste de McNemar exige uma tabela 2x2."
        )

    b = tabela_sem_total.iloc[0, 1]
    c = tabela_sem_total.iloc[1, 0]

    # Estatística de McNemar
    if correcao:
        # Correção de continuidade
        x2 = (abs(b - c) - 1) ** 2 / (b + c)
    else:
        x2 = (b - c) ** 2 / (b + c)


    # p-valor
    p = 1 - chi2.cdf(x2, df=1)

    # Resultado
    resultado = pd.DataFrame({
        "result": [
            round(x2, 4),
            1,
            round(p, 4),
            len(dados)
        ]
    },
        index=["X²", "gl", "p-value", "N"]
    )

    # transforma em object para aceitar texto
    resultado = resultado.astype(object)

    # Interpretação
    if p < alfa:
        conclusao = (
            "Rejeitar H0"
        )
    else:
        conclusao = (
            "Não se rejeitar H0"
        )

    resultado.loc["Conclusão", "result"] = conclusao

    return {
        "tabela": contingencia,
        "resultado": resultado
    }



#----------------------------------------------#
#     REGRESSÃO LINEAR & LOGÍSTICA BINÁRIA     #
#----------------------------------------------#

# REGRESSÃO LINEAR SIMPLES
def regressao_linear_simples(
        base: pd.DataFrame,
        y: str,
        x: str,
        alpha: float = 0.05
):
    """
    Regressão linear simples.

    Parâmetros
    ----------
    base : DataFrame

    y : str
        Variável dependente

    x : str
        Variável independente

    alpha : float
        Nível de significância para IC

    Retorna
    -------
    dict contendo:
    - modelo
    - coeficientes
    - residuos
    - valores_previstos
    """

    # DADOS
    dados = base[[y, x]].dropna()

    y_dados = dados[y]
    x_dados = dados[x]

    # ADICIONA INTERCEPTO
    x_modelo = sm.add_constant(x_dados)

    # AJUSTA MODELO
    modelo = sm.OLS(
        y_dados,
        x_modelo
    ).fit()

    # TABELA DO MODELO
    r = np.sqrt(modelo.rsquared)

    tabela_modelo = pd.DataFrame(
        {
            "Resultado": [
                r,
                modelo.rsquared,
                modelo.fvalue,
                int(modelo.df_model),
                int(modelo.df_resid),
                modelo.f_pvalue
            ]
        },

        index=[
            "R",
            "R²",
            "F",
            "gl1",
            "gl2",
            "p-value"
        ]
    )

    # COEFICIENTES
    conf = modelo.conf_int(alpha=alpha)

    coeficientes = pd.DataFrame(
        {
            "Estimativas": modelo.params,
            "Erro-padrão": modelo.bse,
            "Lim. inferior": conf[0],
            "Lim. superior": conf[1],
            "t": modelo.tvalues,
            "p-value": modelo.pvalues
        }
    )

    # RENOMEIA ÍNDICES
    coeficientes.index = [
        "Intercepto" if i == "const" else i
        for i in coeficientes.index
    ]

    # RESÍDUOS
    residuos = pd.DataFrame(
        {
            "Resíduos": modelo.resid
        }
    )

    # VALORES PREVISTOS
    valores_previstos = pd.DataFrame(
        {
            "Valores Previstos": modelo.fittedvalues
        }
    )

    # ARREDONDAMENTO
    tabela_modelo = tabela_modelo.round(4)
    coeficientes = coeficientes.round(4)
    valores_previstos = valores_previstos.round(4)

    # RESULTADO
    return {
        "modelo": tabela_modelo,
        "coeficientes": coeficientes,
        "residuos": residuos,
        "valores_previstos": valores_previstos
    }


# REGRESSÃO LINEAR MULTIPLA
def regressao_linear_multipla(
    base: pd.DataFrame,
    y: str,
    covariaveis: list[str] = None,
    fatores: list[str] = None,
    referencias: dict[str, str] = None,
    alpha: float = 0.05
):
    """
    Regressão Linear Múltipla com tratamento automático de variáveis categóricas.

    Parâmetros
    ----------
    base : pd.DataFrame
        DataFrame contendo os dados.

    y : str
        Variável dependente.

    covariaveis : list[str]
        Variáveis numéricas.

    fatores : list[str]
        Variáveis qualitativas.

    referencias : dict[str, str]
        Dicionário contendo:
        {
            "coluna_fator": "categoria_referencia"
        }

    alpha : float
        Nível de significância para IC.

    Retorno
    -------
    modelo : pd.DataFrame
        Métricas do modelo.

    coeficientes : pd.DataFrame
        Tabela de coeficientes.

    modelo_fit:  RegressionResults
        Modelo treinado.

    residuos: pd.DataFrame
        Tabela com os residuos.

    preditos: pd.DataFrame
        Tabela com os valores preditos.
    """

    covariaveis = covariaveis or []
    fatores = fatores or []
    referencias = referencias or {}

    df = base.copy()

    X = pd.DataFrame(index=df.index)

    # Variáveis Numéricas
    for coluna in covariaveis:
        X[coluna] = df[coluna]

    # Variáveis Categóricas
    nomes_dummies = {}
    for fator in fatores:
        if fator not in referencias:
            raise ValueError(
                f"A variável '{fator}' precisa de uma categoria de referência."
            )

        referencia = referencias[fator]

        categorias = list(df[fator].dropna().unique())

        if referencia not in categorias:
            raise ValueError(
                f"A referência '{referencia}' não existe na coluna '{fator}'."
            )

        # Mantém referência primeiro
        categorias_ordenadas = [referencia] + [
            c for c in categorias if c != referencia
        ]

        df[fator] = pd.Categorical(
            df[fator],
            categories=categorias_ordenadas
        )

        dummies = pd.get_dummies(
            df[fator],
            prefix="",
            prefix_sep="",
            drop_first=True
        ).astype(int)

        # Renomear dummies
        renomeadas = {}

        for coluna_dummy in dummies.columns:
            novo_nome = f"*{coluna_dummy} - {referencia}"
            renomeadas[coluna_dummy] = novo_nome
            nomes_dummies[novo_nome] = coluna_dummy

        dummies = dummies.rename(columns=renomeadas)

        X = pd.concat([X, dummies], axis=1)

    # Remover NA
    dados = pd.concat([df[y], X], axis=1).dropna()

    y_final = dados[y]
    X_final = dados.drop(columns=[y])

    # Modelo
    X_final = sm.add_constant(X_final)

    modelo_fit = sm.OLS(y_final, X_final).fit()


    # Métricas do Modelo
    rmse = np.sqrt(modelo_fit.mse_resid)

    modelo = pd.DataFrame({
        "result": [
            modelo_fit.rsquared ** 0.5,
            modelo_fit.rsquared,
            modelo_fit.rsquared_adj,
            rmse,
            modelo_fit.fvalue,
            int(modelo_fit.df_model),
            int(modelo_fit.df_resid),
            modelo_fit.f_pvalue
        ]
    },
    index=[
        "R",
        "R²",
        "R² ajustado",
        "RMSE",
        "F",
        "gl1",
        "gl2",
        "p-value"
    ])

    # Coeficientes
    conf = modelo_fit.conf_int(alpha=alpha)

    coeficientes = pd.DataFrame({
        "Preditor": [
            "Intercepto" if i == "const" else i
            for i in modelo_fit.params.index
        ],
        "Estimativas": modelo_fit.params.values,
        "Erro-Padrão": modelo_fit.bse.values,
        "Lim. inferior": conf[0].values,
        "Lim. superior": conf[1].values,
        "t": modelo_fit.tvalues.values,
        "p-value": modelo_fit.pvalues.values
    })

    # Ordenação
    # Numéricas primeiro, dummies depois
    ordem = ["Intercepto"]

    ordem += covariaveis

    for fator in fatores:
        referencia = referencias[fator]

        categorias = [
            c for c in df[fator].cat.categories
            if c != referencia
        ]

        for categoria in categorias:
            ordem.append(f"*{categoria} - {referencia}")

    coeficientes["ordem"] = coeficientes["Preditor"].apply(
        lambda x: ordem.index(x) if x in ordem else 999
    )

    coeficientes = (
        coeficientes
        .sort_values("ordem")
        .drop(columns="ordem")
        .reset_index(drop=True)
    )
    modelo = modelo.round(5)
    coeficientes = coeficientes.round(5)
    return {
        "modelo": modelo,
        "coeficientes": coeficientes,
        "modelo_fit": modelo_fit,
        "residuos": modelo_fit.resid,
        "preditos": modelo_fit.fittedvalues
    }



# REGRESSÃO LOGÍSTICA (BINÁRIA)
def regressao_logistica_binaria(
    base: pd.DataFrame,
    y: str,
    covariaveis: list[str] = None,
    fatores: list[str] = None,
    referencias: dict[str, str] = None,
    alpha: float = 0.05,
    cutoff: float = 0.5
):
    """
    Regressão Logística Binária com tratamento automático de variáveis categóricas.

    Parâmetros
    ----------
    base : pd.DataFrame
        DataFrame contendo os dados.

    y : str
        Variável dependente binária.

    covariaveis : list[str]
        Variáveis numéricas.

    fatores : list[str]
        Variáveis qualitativas.

    referencias : dict[str, str]
        Dicionário contendo:
        {
            "coluna_fator": "categoria_referencia"
        }

    alpha : float
        Nível de significância para IC.

    cutoff : float
        Ponto de corte para classificar a probabilidade em 0 ou 1.

    Retorno
    -------
    dict com:
        - modelo
        - coeficientes
        - matriz_confusao
        - tabela_metricas
        - modelo_fit
        - residuos
        - preditos
    """

    covariaveis = covariaveis or []
    fatores = fatores or []
    referencias = referencias or {}

    df = base.copy()

    if y not in df.columns:
        raise ValueError(f"A variável dependente '{y}' não existe no dataframe.")

    # Garantia de variável dependente binária
    y_series = df[y].copy()

    if y_series.nunique(dropna=True) != 2:
        raise ValueError(
            f"A variável dependente '{y}' precisa ser binária, com exatamente 2 categorias."
        )

    # Se não for numérica binária 0/1, transforma em 0/1
    if not pd.api.types.is_numeric_dtype(y_series):
        classes = list(pd.unique(y_series.dropna()))
        if len(classes) != 2:
            raise ValueError(
                f"A variável dependente '{y}' precisa ter exatamente 2 categorias válidas."
            )
        mapping_y = {classes[0]: 0, classes[1]: 1}
        y_series = y_series.map(mapping_y)
    else:
        valores_validos = set(y_series.dropna().unique())
        if not valores_validos.issubset({0, 1}):
            raise ValueError(
                f"A variável dependente '{y}' precisa estar codificada como 0 e 1."
            )

    X = pd.DataFrame(index=df.index)

    # Variáveis Numéricas
    for coluna in covariaveis:
        X[coluna] = df[coluna]

    # Variáveis Categóricas
    for fator in fatores:
        if fator not in referencias:
            raise ValueError(
                f"A variável '{fator}' precisa de uma categoria de referência."
            )

        referencia = referencias[fator]
        categorias = list(df[fator].dropna().unique())

        if referencia not in categorias:
            raise ValueError(
                f"A referência '{referencia}' não existe na coluna '{fator}'."
            )

        categorias_ordenadas = [referencia] + [c for c in categorias if c != referencia]

        df[fator] = pd.Categorical(df[fator], categories=categorias_ordenadas)

        dummies = pd.get_dummies(
            df[fator],
            prefix="",
            prefix_sep="",
            drop_first=True
        ).astype(int)

        renomeadas = {}
        for coluna_dummy in dummies.columns:
            renomeadas[coluna_dummy] = f"*{coluna_dummy} - {referencia}"

        dummies = dummies.rename(columns=renomeadas)
        X = pd.concat([X, dummies], axis=1)

    # Remover NA
    dados = pd.concat([y_series.rename(y), X], axis=1).dropna()

    y_final = dados[y]
    X_final = dados.drop(columns=[y])

    # Modelo Logístico
    X_final = sm.add_constant(X_final, has_constant="add")

    modelo_fit = sm.GLM(
        y_final,
        X_final,
        family=sm.families.Binomial()
    ).fit()

    # Predições
    prob_predita = modelo_fit.predict(X_final)
    classe_predita = (prob_predita >= cutoff).astype(int)

    # Resíduos
    residuos = pd.DataFrame({
        "observado": y_final.values,
        "probabilidade_predita": prob_predita.values,
        "residuo_resposta": (y_final - prob_predita).values
    }, index=dados.index)

    preditos = pd.DataFrame({
        "probabilidade": prob_predita.values,
        "classe_prevista": classe_predita.values
    }, index=dados.index)

    # Matriz de confusão
    matriz_confusao = pd.crosstab(
        pd.Series(y_final, name="Real"),
        pd.Series(classe_predita, name="Previsto"),
        rownames=["Real"],
        colnames=["Previsto"],
        dropna=False
    )

    for linha in [0, 1]:
        if linha not in matriz_confusao.index:
            matriz_confusao.loc[linha] = 0
    for coluna in [0, 1]:
        if coluna not in matriz_confusao.columns:
            matriz_confusao[coluna] = 0

    matriz_confusao = matriz_confusao.sort_index().reindex(sorted(matriz_confusao.columns), axis=1)

    tn = matriz_confusao.loc[0, 0]
    fp = matriz_confusao.loc[0, 1]
    fn = matriz_confusao.loc[1, 0]
    tp = matriz_confusao.loc[1, 1]

    matriz_confusao["% Acerto"] = [
        (
            tn / (tn + fp) * 100
            if (tn + fp) > 0
            else np.nan
        ),
        (
            tp / (tp + fn) * 100
            if (tp + fn) > 0
            else np.nan
        )
    ]

    acuracia = (tp + tn) / max((tp + tn + fp + fn), 1)
    precisao = tp / max((tp + fp), 1)
    recall = tp / max((tp + fn), 1)
    especificidade = tn / max((tn + fp), 1)
    f1_score = (
            2 * precisao * recall
            / max((precisao + recall), 1e-10)
    )
    auc = roc_auc_score(
        y_final,
        prob_predita
    )
    # KS
    fpr, tpr, _ = roc_curve(
        y_final,
        prob_predita
    )
    ks = np.max(np.abs(tpr - fpr))
    # Gini
    gini = 2 * auc - 1
    tabela_metricas = pd.DataFrame({
        "result": [
            acuracia,
            precisao,
            recall,
            especificidade,
            f1_score,
            auc,
            ks,
            gini
        ]
    }, index=[
        "Acurácia",
        "Precisão",
        "Recall",
        "Especificidade",
        "F1-Score",
        "AUC",
        "KS",
        "Gini"
    ])

    # Métricas do Modelo
    llf = modelo_fit.llf
    llnull = modelo_fit.llnull if hasattr(modelo_fit, "llnull") else np.nan
    deviance = modelo_fit.deviance if hasattr(modelo_fit, "deviance") else np.nan
    null_deviance = modelo_fit.null_deviance if hasattr(modelo_fit, "null_deviance") else np.nan

    if pd.notna(llnull) and llnull != 0:
        pseudo_r2 = 1 - (llf / llnull)
    else:
        pseudo_r2 = np.nan

    lr_chi2 = 2 * (llf - llnull) if pd.notna(llnull) else np.nan
    gl = int(modelo_fit.df_model) if hasattr(modelo_fit, "df_model") else np.nan
    p_lr = stats.chi2.sf(lr_chi2, gl) if pd.notna(lr_chi2) and pd.notna(gl) else np.nan

    modelo = pd.DataFrame({
        "result": [
            llf,
            modelo_fit.aic,
            modelo_fit.bic_llf,
            deviance,
            null_deviance,
            pseudo_r2,
            lr_chi2,
            gl,
            p_lr
        ]
    }, index=[
        "Log-Likelihood",
        "AIC",
        "BIC",
        "Deviance",
        "Null Deviance",
        "Pseudo R² (McFadden)",
        "LR Chi²",
        "gl",
        "p-value"
    ])

    # Coeficientes
    conf = modelo_fit.conf_int(alpha=alpha)
    odds_ratio = np.exp(modelo_fit.params)
    lim_inf_or = np.exp(conf[0])
    lim_sup_or = np.exp(conf[1])

    coeficientes = pd.DataFrame({
        "Preditor": [
            "Intercepto" if i == "const" else i
            for i in modelo_fit.params.index
        ],
        "Estimativas (log-Odds)": modelo_fit.params.values,
        "Odds Ratio": odds_ratio.values,
        "%Impacto (Odds)": [f"{v:+.1f}%" for v in ((odds_ratio.values - 1) * 100)],
        "Erro-Padrão": modelo_fit.bse.values,
        "Lim. inferior": lim_inf_or.values,
        "Lim. superior": lim_sup_or.values,
        "z": modelo_fit.tvalues.values,
        "p-value Wald": modelo_fit.pvalues.values
    })

    # Ordenação: covariáveis primeiro, fatores por último
    ordem = ["Intercepto"] + covariaveis

    for fator in fatores:
        referencia = referencias[fator]
        categorias = [c for c in df[fator].cat.categories if c != referencia]
        for categoria in categorias:
            ordem.append(f"*{categoria} - {referencia}")

    coeficientes["ordem"] = coeficientes["Preditor"].apply(
        lambda x: ordem.index(x) if x in ordem else 999
    )

    coeficientes = (
        coeficientes
        .sort_values("ordem")
        .drop(columns="ordem")
        .reset_index(drop=True)
        .round(5)
    )

    modelo = modelo.round(5)
    tabela_metricas = tabela_metricas.round(5)
    matriz_confusao = matriz_confusao.astype(int)

    return {
        "modelo": modelo,
        "coeficientes": coeficientes,
        "matriz_confusao": matriz_confusao,
        "tabela_metricas": tabela_metricas,
        "modelo_fit": modelo_fit,
        "residuos": residuos,
        "preditos": preditos
    }




#############################################
#       VERIFICAÇÃO DE PRESSUPOSTOS         #
#############################################

# teste de normalidade (Shapiro, Kolmogorov, Anderson)
def teste_normalidade(base=None, dados: str | list | pd.Series = None):
    """
    Teste de normalidade com:
    - Shapiro-Wilk
    - Kolmogorov-Smirnov
    - Anderson-Darling

    Ha: os dados não seguem uma 'distribuição normal'
    p-value > alfa -> Ha
    """

    # MÚLTIPLAS COLUNAS
    if isinstance(dados, list):
        resultados = {}
        for col in dados:
            resultados[col] = teste_normalidade(base,
                dados=col
            ).iloc[:, 0]

        return pd.DataFrame(resultados)


    # COLUNA ÚNICA
    nome_coluna = "Variável"

    if isinstance(dados, str):
        nome_coluna = dados
        dados = base[dados]

    elif isinstance(dados, pd.Series):
        nome_coluna = dados.name if dados.name else "Variável"

    dados = pd.Series(dados).dropna()

    # SHAPIRO-WILK
    s_estatistica, s_p_valor = shapiro(dados)

    # KOLMOGOROV-SMIRNOV
    media = np.mean(dados)
    desvio = np.std(dados, ddof=1)

    dados_padronizados = (dados - media) / desvio

    k_estatistica, k_p_valor = kstest(
        dados_padronizados,
        "norm"
    )

    # ANDERSON-DARLING
    a_estatistica, a_p_valor = normal_ad(dados)

    # RESULTADO
    val = {
        "Shapiro-Wilk statistic": s_estatistica,
        "Kolmogorov-Smirnov statistic": k_estatistica,
        "Anderson-Darling statistic": a_estatistica,

        "Shapiro-Wilk p-value": s_p_valor,
        "Kolmogorov-Smirnov p-value": k_p_valor,
        "Anderson-Darling p-value": a_p_valor,
    }
    return pd.DataFrame(
        val,
        index=[nome_coluna]
    ).T


# teste de heterocedasticidade (Breusch-Pagan, Goldfeld-Quandt, Harrison-McCabe)
def teste_heterocedasticidade(
        base: pd.DataFrame, y: str, x: str | list
):
    """
    Testes de heterocedasticidade:
    - Breusch-Pagan
    - Goldfeld-Quandt
    - Harrison-McCabe

    H0: os resíduos possuem homocedasticidade
    Ha: os resíduos possuem heterocedasticidade

    p-value < alfa -> rejeita H0
    """

    # MULTIPLAS VARIÁVEIS
    if isinstance(x, str):
        x = [x]

    # DADOS
    dados = base[[y] + x].dropna()

    y_dados = dados[y]
    x_dados = dados[x]

    # ADICIONA INTERCEPTO
    x_dados = sm.add_constant(x_dados)

    # AJUSTA MODELO
    modelo = sm.OLS(
        y_dados,
        x_dados
    ).fit()

    residuos = modelo.resid

    # BREUSCH-PAGAN
    bp_estatistica, bp_p_valor, _, _ = het_breuschpagan(
        residuos,
        x_dados
    )

    # GOLDFELD-QUANDT
    gq_estatistica, gq_p_valor, _ = het_goldfeldquandt(
        y_dados,
        x_dados
    )

    # WHITE TEST
    w_estatistica, w_p_valor, _, _ = het_white(
        residuos,
        x_dados
    )

    # RESULTADO
    val = {
        "statistic": [
            bp_estatistica,
            gq_estatistica,
            w_estatistica
        ],
        "p-value": [
            bp_p_valor,
            gq_p_valor,
            w_p_valor
        ]
    }
    return pd.DataFrame(
        val,
        index=[
            "Breusch-Pagan",
            "Goldfeld-Quandt",
            "White"
        ]
    )


# teste de homogeneidade das variâncias (levene)
def teste_levene(
        base, colunas: list, center: str = "median", alfa: float = 0.05
):
    """
    Realiza o teste de Levene para homogeneidade das variâncias.

    Parâmetros
    ----------
    colunas : list
        Lista com os nomes das colunas numéricas.

    center : str, default="median"
        Medida central utilizada:
        - "mean"
        - "median"
        - "trimmed"

    alfa : float, default=0.05
        Nível de significância.

    Retorno
    -------
    pd.DataFrame
        Resultado do teste.
    """
    grupos = []

    for col in colunas:
        serie = base[col].dropna()

        if serie.empty:
            raise ValueError(f"A coluna '{col}' não possui dados válidos.")

        grupos.append(serie)

    estatistica, p_valor = levene(*grupos, center=center)

    if p_valor > alfa:
        conclusao = (
            "Não rejeitamos H0. "
            "Variâncias Homogêneas."
        )
    else:
        conclusao = (
            "Rejeitamos H0. "
            "variâncias Heterogêneas"
        )

    resultado = pd.DataFrame(
        {
            "result": [
                estatistica,
                p_valor,
                alfa,
                center,
                conclusao
            ]
        },
        index=[
            "Estatística",
            "p-valor",
            "Alfa",
            "Centro",
            "Conclusão"
        ]
    )
    return resultado


# teste de Durbin-Watson (
def teste_durbin_watson(residuos: pd.Series):
    """
    Realiza o teste de autocorrelação de Durbin-Watson.

    Parâmetros
    ----------
    residuos : array-like
        Resíduos do modelo.

    Retorno
    -------
    DataFrame
        Estatística DW e autocorrelação aproximada.

    Interpretacao
    -------------
    | DW           | Interpretação           |
    | ------------ | ----------------------- |
    | ≈ 2          | sem autocorrelação      |
    | < 2          | autocorrelação positiva |
    | > 2          | autocorrelação negativa |
    | próximo de 0 | forte positiva          |
    | próximo de 4 | forte negativa          |

    """

    # estatística DW
    dw = durbin_watson(residuos)

    # aproximação da autocorrelação
    autocorrelacao = 1 - (dw / 2)

    return pd.DataFrame(
        {
            "Autocorrelação": autocorrelacao,
            "Estatística DW": dw
        },
        index=["Durbin-Watson"]
    )


# teste de colinearidade
def estatistica_colinearidade(base: pd.DataFrame, colunas: list[str]):
    """
    Calcula Variance Inflation Factor (VIF)
    e tolerância das variáveis independentes.

    Parâmetros
    ----------
    base : DataFrame
        Base de dados.

    colunas : list[str]
        Variáveis independentes.

    Retorno
    -------
    DataFrame
        VIF e tolerância.

    Interpretação
    -------------
    | VIF   | Interpretação     |
    |-------|-------------------|
    | 1     | Sem colinearidade |
    | 1-5   | Baixa/moderada    |
    | > 5   | Problemática      |
    | > 10  | Grave             |
    """

    X = base[colunas].copy()

    # adiciona intercepto
    X_const = sm.add_constant(X)

    vif = []

    for i in range(1, X_const.shape[1]):

        vif_valor = variance_inflation_factor(
            X_const.values,
            i
        )

        tolerancia = (
            0
            if np.isinf(vif_valor)
            else 1 / vif_valor
        )

        vif.append(
            {
                "Variável": X_const.columns[i],
                "VIF": vif_valor,
                "Tolerância": tolerancia
            }
        )

    return (
        pd.DataFrame(vif)
        .set_index("Variável")
        .round(3)
    )


