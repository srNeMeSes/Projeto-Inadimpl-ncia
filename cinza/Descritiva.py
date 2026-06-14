
import pandas as pd
import numpy as np

from scipy.stats import mode
from scipy.stats import t
from scipy.stats import skew
from scipy.stats import kurtosis
from scipy.stats import entropy


def describe_df(df: pd.DataFrame, confianca=0.95):

    _EPS = 1e-10  # threshold para desvio ~zero

    numericas = []
    categoricas = []
    frequencias = []

    for coluna in df.columns:

        serie = df[coluna]

        # =====================
        # NUMÉRICAS
        # =====================

        if pd.api.types.is_numeric_dtype(serie):

            x = serie.dropna()

            if len(x) < 2:
                continue

            n = len(x)

            media = x.mean()
            mediana = x.median()

            moda = x.mode()
            moda = moda.iloc[0] if len(moda) else np.nan

            variancia = x.var(ddof=1)
            desvio = x.std(ddof=1)

            erro_padrao = desvio / np.sqrt(n)

            alpha = 1 - confianca

            t_critico = t.ppf(
                1 - alpha/2,
                df=n-1
            )

            margem = t_critico * erro_padrao

            q1 = x.quantile(.25)
            q3 = x.quantile(.75)

            iqr = q3 - q1

            li_out = q1 - 1.5 * iqr
            ls_out = q3 + 1.5 * iqr

            # Skewness e kurtosis só fazem sentido com variância real
            dados_variam = desvio > _EPS

            numericas.append({

                "variavel": coluna,

                "n": n,
                "missing": serie.isna().sum(),

                "media": media,
                "mediana": mediana,
                "moda": moda,

                "min": x.min(),
                "max": x.max(),
                "amplitude": x.max() - x.min(),

                "q1": q1,
                "q2": mediana,
                "q3": q3,

                "variancia": variancia,
                "desvio_padrao": desvio,

                "coef_variacao_%":
                    (desvio/media)*100
                    if media != 0 else np.nan,

                "erro_padrao": erro_padrao,

                "assimetria":
                    skew(x, bias=False)
                    if dados_variam else np.nan,

                "curtose":
                    kurtosis(x, fisher=True, bias=False)
                    if dados_variam else np.nan,

                "p5":  x.quantile(.05),
                "p95": x.quantile(.95),

                "iqr": iqr,

                "lim_inf_outlier": li_out,
                "lim_sup_outlier": ls_out,

                "qtd_outliers":
                    ((x < li_out) |
                     (x > ls_out)).sum(),

                "ic95_li": media - margem,
                "ic95_ls": media + margem

            })

        # =====================
        # CATEGÓRICAS
        # =====================

        else:

            x = serie.dropna()

            freq = x.value_counts()

            prop = x.value_counts(normalize=True)

            moda = freq.index[0] if len(freq) else None
            freq_moda = freq.iloc[0] if len(freq) else 0

            categoricas.append({

                "variavel": coluna,

                "n": len(x),
                "missing": serie.isna().sum(),

                "categorias": x.nunique(),

                "moda": moda,
                "freq_moda": freq_moda,

                "perc_moda":
                    round(freq_moda / len(x) * 100, 2)
                    if len(x) else np.nan,

                "entropia":
                    entropy(prop, base=2)
                    if len(prop) else np.nan,

                "indice_simpson":
                    1 - np.sum(prop**2)
                    if len(prop) else np.nan
            })

            for categoria, freq_abs in freq.items():

                frequencias.append({

                    "variavel":   coluna,
                    "categoria":  categoria,
                    "frequencia": freq_abs,

                    "percentual":
                        round(freq_abs / len(x) * 100, 2)
                })

    return {
        "numericas":   pd.DataFrame(numericas),
        "categoricas": pd.DataFrame(categoricas),
        "frequencias": pd.DataFrame(frequencias)
    }


def tabela_frequencia(df, coluna_x, coluna_y):

    tabela = pd.crosstab(
        df[coluna_x],
        df[coluna_y]
    )

    # Total por categoria
    tabela["Total"] = tabela.sum(axis=1)

    # Total geral
    total_geral = tabela["Total"].sum()

    # PropPop
    tabela["PropPop"] = (
        tabela["Total"]
        / total_geral
        * 100
    ).round(2)

    # Colunas da variável y
    classes = tabela.columns[:-2]

    # Formatar frequência + percentual
    for classe in classes:

        percentual = (
            tabela[classe]
            / tabela["Total"]
            * 100
        ).round(2)

        tabela[classe] = (
            tabela[classe].astype(str)
            + " ("
            + percentual.astype(str)
            + "%)"
        )

    tabela["PropPop"] = tabela["PropPop"].astype(str) + "%"

    return tabela.reset_index()

    tabela = pd.crosstab(
        df[coluna_x],
        df[coluna_y]
    )

    # Total por categoria
    tabela["Total"] = tabela.sum(axis=1)

    # Total geral
    total_geral = tabela["Total"].sum()

    # Percentual da população
    tabela["PropPop"] = (
        tabela["Total"]
        / total_geral
        * 100
    ).round(2)

    # Percentuais das classes
    classes = tabela.columns[:-2]

    for classe in classes:
        tabela[f"{classe}perc"] = (
            tabela[classe]
            / tabela["Total"]
            * 100
        ).round(2)

    return tabela.reset_index()


def information_value(
    base: pd.DataFrame,
    colunas: list[str] = None,
    y: str = None,
    evento: object = None
) -> dict:
    """
    Calcula o Information Value (IV) de variáveis categóricas
    em relação a uma variável alvo binária.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contendo os dados de entrada.
    colunas : list[str], optional
        Lista com os nomes das variáveis categóricas a serem analisadas.
        Se None, todas as colunas categóricas do DataFrame (exceto `y`)
        são selecionadas automaticamente.
    y : str
        Nome da variável alvo binária. Deve conter exatamente dois
        valores distintos.
    evento : object, optional
        Valor de `y` que representa o evento (classe positiva / classe 1).
        Se None, o maior valor após ordenação é usado como evento
        (comportamento padrão — correto para y ∈ {0, 1}).
        Útil quando a ordenação lexicográfica não reflete a semântica
        do evento (ex: evento="Default" em y ∈ {"No Default", "Default"}).

    Returns
    -------
    dict
        Dicionário onde:
        - Cada chave é o nome de uma variável analisada, com valor sendo
          um pd.DataFrame com as colunas:
          [<variavel>, Frequencia, %Frequencia, Frequencia_0,
           Frequencia_1, %<y>, Dist_0, Dist_1, WoE, IV]
          incluindo uma linha final "Total".
        - A chave "IV_Geral" contém um DataFrame resumo com as colunas
          [Variavel, IV, Interpretacao], ordenado por IV decrescente.

    Raises
    ------
    ValueError
        Se `y` for None, não existir no DataFrame ou não for binária.

    Notes
    -----
    Interpretação do IV:
        < 0.02        → Sem poder preditivo
        0.02 a < 0.10 → Fraco
        0.10 a < 0.30 → Médio
        0.30 a < 0.50 → Forte
        >= 0.50       → Suspeito / Possível overfitting

    Valores ausentes são agrupados como a categoria "MISSING".
    Divisões por zero no WoE são evitadas com suavização de Laplace (α = 0.5).
    Examples
    --------
    >>> r = information_value(df, colunas=["Formacao", "Salario"], y="Churn")
    >>> r["IV_Geral"]
    >>> r["Formacao"]
    """

    # ------------------------------------------------------------------
    # 1. Validações
    # ------------------------------------------------------------------
    if y is None:
        raise ValueError("O parâmetro 'y' é obrigatório.")

    if y not in base.columns:
        raise ValueError(
            f"A coluna alvo '{y}' não existe no DataFrame. "
            f"Colunas disponíveis: {list(base.columns)}"
        )

    classes = sorted(base[y].dropna().unique())

    if len(classes) != 2:
        raise ValueError(
            f"'{y}' deve ter exatamente 2 classes. "
            f"Encontradas: {len(classes)} → {classes}"
        )

    if evento is not None:
        if evento not in classes:
            raise ValueError(
                f"'evento={evento}' não encontrado em '{y}'. "
                f"Valores disponíveis: {classes}"
            )
        classe_1 = evento
        classe_0 = next(c for c in classes if c != evento)
    else:
        classe_0, classe_1 = classes[0], classes[1]


    # ------------------------------------------------------------------
    # 2. Base de cálculo (exclui NaN no alvo)
    # ------------------------------------------------------------------
    df_base = base[base[y].notna()].copy()

    total_geral = len(df_base)
    total_0     = int((df_base[y] == classe_0).sum())
    total_1     = int((df_base[y] == classe_1).sum())

    # ------------------------------------------------------------------
    # 3. Seleção automática de colunas categóricas
    # ------------------------------------------------------------------
    if colunas is None:
        colunas = [
            col for col in df_base.columns
            if col != y and (
                pd.api.types.is_object_dtype(df_base[col])      or
                pd.api.types.is_categorical_dtype(df_base[col]) or
                pd.api.types.is_bool_dtype(df_base[col])
            )
        ]
        if not colunas:
            raise ValueError(
                "Nenhuma coluna categórica encontrada no DataFrame "
                "(excluindo a variável alvo)."
            )

    # ------------------------------------------------------------------
    # 4. Cálculo do IV por variável
    # ------------------------------------------------------------------
    #epsilon       = 1e-10
    resultado     = {}
    iv_geral_rows = []

    for coluna in colunas:

        # Tratar ausentes como categoria explícita "MISSING"
        serie = df_base[coluna].fillna("MISSING").astype(str)

        frame = pd.DataFrame(
            {coluna: serie.values, y: df_base[y].values},
            index=df_base.index
        )

        # Contagem cruzada: categoria × classe alvo
        counts = (
            frame
            .groupby(coluna, observed=True)[y]
            .value_counts()
            .unstack(fill_value=0)
        )

        # Garante que ambas as colunas de classe existam
        for cls in [classe_0, classe_1]:
            if cls not in counts.columns:
                counts[cls] = 0

        counts = (
            counts
            .rename(columns={classe_0: "Frequencia_0",
                              classe_1: "Frequencia_1"})
            [["Frequencia_0", "Frequencia_1"]]
            .reset_index()
        )

        # Frequências absolutas e relativas
        counts["Frequencia"]    = counts["Frequencia_0"] + counts["Frequencia_1"]
        counts["%Frequencia"]   = (counts["Frequencia"]   / total_geral * 100).round(2)
        counts[f"%{y}"]         = (counts["Frequencia_1"] / counts["Frequencia"] * 100).round(2)

        # Distribuições por classe em escala percentual (0–100)
        counts["Dist_0"]        = (counts["Frequencia_0"] / total_0 * 100).round(4)
        counts["Dist_1"]        = (counts["Frequencia_1"] / total_1 * 100).round(4)

        # WoE com suavização de Laplace (α = 0.5) para evitar log(0)
        n_cats = len(counts)
        d0_smooth = (counts["Frequencia_0"] + 0.5) / (total_0 + 0.5 * n_cats)
        d1_smooth = (counts["Frequencia_1"] + 0.5) / (total_1 + 0.5 * n_cats)
        counts["WoE"] = np.log(d1_smooth / d0_smooth).round(4)
        counts["IV"] = ((d1_smooth - d0_smooth) * counts["WoE"]).round(6)

        iv_total = counts["IV"].sum()

        # Ordenar por contribuição IV decrescente
        counts = (
            counts
            .sort_values("IV", ascending=False)
            .reset_index(drop=True)
        )

        # Ordenação final das colunas
        counts = counts[[
            coluna,
            "Frequencia", "%Frequencia",
            "Frequencia_0", "Frequencia_1",
            f"%{y}",
            "Dist_0", "Dist_1",
            "WoE", "IV"
        ]]

        # Linha "Total"
        linha_total = pd.DataFrame([{
            coluna:         "Total",
            "Frequencia":   total_geral,
            "%Frequencia":  100.0,
            "Frequencia_0": total_0,
            "Frequencia_1": total_1,
            f"%{y}":        round(total_1 / total_geral * 100, 2),
            "Dist_0":       100.0,
            "Dist_1":       100.0,
            "WoE":          np.nan,
            "IV":           round(iv_total, 6)
        }])

        resultado[coluna] = pd.concat(
            [counts, linha_total],
            ignore_index=True
        )

        iv_geral_rows.append({
            "Variavel": coluna,
            "IV":       round(iv_total, 6)
        })

    # ------------------------------------------------------------------
    # 5. IV_Geral — resumo com interpretação
    # ------------------------------------------------------------------
    def _interpretar(iv: float) -> str:
        if   iv < 0.02: return "Sem poder preditivo"
        elif iv < 0.10: return "Fraco"
        elif iv < 0.30: return "Médio"
        elif iv < 0.50: return "Forte"
        else:           return "Suspeito / Possível overfitting"

    iv_geral = (
        pd.DataFrame(iv_geral_rows)
        .sort_values("IV", ascending=False)
        .reset_index(drop=True)
    )
    iv_geral["Interpretacao"] = iv_geral["IV"].apply(_interpretar)

    resultado["IV_Geral"] = iv_geral

    return resultado



def categorizar_intervalos(base, coluna, vmin, vmax, passos):
    """
    Transforma uma variável numérica em categórica.

    Exemplo:
    min=1500
    passos=500
    max=20000

    Até 1500
    1501 - 2000
    2001 - 2500
    ...
    19501 - 20000
    + 20000
    """

    bins = [-np.inf]

    atual = vmin

    while atual < vmax:
        bins.append(atual)
        atual += passos

    bins.append(vmax)
    bins.append(np.inf)

    labels = [f"~ {vmin}"]

    inicio = vmin + 1

    while inicio <= vmax:
        fim = min(inicio + passos - 1, vmax)
        labels.append(f"{inicio} - {fim}")
        inicio += passos

    labels.append(f"+ {vmax}")

    return pd.cut(
        base[coluna],
        bins=bins,
        labels=labels,
        include_lowest=True
    )











