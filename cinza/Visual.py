import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import probplot
import math
import pandas as pd
import numpy as np
from scipy import stats
from .Utils import *
import statsmodels.api as sm


# estilo branco com grid suave
sns.set_style("whitegrid")



# visual boxplot
def v_boxplot(base, colunas=None, ncols=4):
    dados = base.select_dtypes(include="number")
    if colunas:
        dados = dados[colunas]
    n = len(dados.columns)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5 * ncols, 5 * nrows)
    )
    axes = axes.flatten()
    for i, coluna in enumerate(dados.columns):
        sns.boxplot(y=dados[coluna], ax=axes[i])
        axes[i].set_title(coluna)

    # Remove eixos vazios
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    return plt.show()


# visual histograma
def v_histogram(base, colunas: list = None, bins: str | int = 'auto'):

    for col in colunas:
        fig, ax = plt.subplots(figsize=(12, 5))

        sns.histplot(base[col], bins=bins, kde=True, ax=ax)

        ax.set_title(f"Distribuição - {col}")

        plt.tight_layout()

    return plt.show()


# visual QQ plot
def v_QQ_plot(
        base, coluna: str | list, dist: str = "norm",
        residual: bool = False, figsize: tuple = (6, 6)
):
    """
    Plota QQ-Plots.

    Parâmetros
    ----------
    coluna : str | list
        Nome da coluna ou lista de colunas.

    dist : str, default="norm"
        Distribuição teórica.

    residual : bool, default=False
        - False:
            Cria um QQ-Plot para cada coluna.
        - True:
            Junta os resíduos padronizados das colunas
            e cria um único QQ-Plot.

    figsize : tuple
        Tamanho da figura.
    """

    # TRANSFORMA EM LISTA
    if isinstance(coluna, str):
        coluna = [coluna]

    # RESIDUAL = TRUE
    if residual:
        residuos = []

        for col in coluna:
            serie = base[col].dropna()

            # Resíduo padronizado
            resid = (serie - serie.mean()) / serie.std(ddof=1)

            residuos.extend(resid.tolist())

        residuos = np.array(residuos)

        plt.figure(figsize=figsize)

        probplot(residuos, dist=dist, plot=plt)

        plt.title(
            f"QQ-Plot dos Resíduos\n({', '.join(coluna)})"
        )

        plt.grid(alpha=0.3)

    # QQ-PLOT INDIVIDUAL
    else:
        n = len(coluna)

        fig, axes = plt.subplots(
            nrows=n,
            figsize=(figsize[0], figsize[1] * n)
        )

        # Quando há apenas 1 gráfico
        if n == 1:
            axes = [axes]

        for ax, col in zip(axes, coluna):
            serie = base[col].dropna()

            probplot(serie, dist=dist, plot=ax)

            ax.set_title(f"QQ-Plot - {col}")
            ax.grid(alpha=0.3)

        plt.tight_layout()

    plt.show()


# visual de análise de variâncias
def v_analise_variancia(
        base, colunas: str | list | None = None, target: str | bool = False,
        titulo: str = "Análise de Variâncias", palette: str = "bright",
        figsize: tuple = (10, 6), jitter: bool = False, dp: bool = False,
        ic: bool = False, alpha_ic: float = 0.95, media: bool = True,
        mediana: bool = False, grid: bool = True
):
    """
    Visualização de grupos para análise de variância.

    Modos
    -----

    1) Wide format
    ----------------
    Cada coluna representa um grupo.

    Exemplo:
        colunas=["Loja_A", "Loja_B"]

    2) Long format
    ----------------
    Uma coluna categórica + target.

    Exemplo:
        colunas="Cargo"
        target="Salario"

    Parâmetros
    ----------
    colunas : str | list
        Colunas dos grupos.

    target : str | bool
        Variável dependente.

    titulo : str
        Título do gráfico.

    palette : str
        Paleta seaborn.

    figsize : tuple
        Tamanho da figura.

    jitter : bool
        Dispersão horizontal dos pontos.

    dp : bool
        Exibe desvio padrão.

    ic : bool
        Exibe intervalo de confiança.

    alpha_ic : float
        Nível de confiança do IC.

    media : bool
        Exibe linha da média.

    mediana : bool
        Exibe linha da mediana.

    grid : bool
        Exibe grid.
    """

    # DADOS LONG FORMAT
    if target:
        if not isinstance(colunas, str):
            raise ValueError(
                "Quando 'target' for informado, "
                "'colunas' deve ser uma coluna categórica."
            )

        df = long_to_wide(
            base,
            categoria=colunas,
            valor=target
        )

    # DADOS WIDE FORMAT
    else:
        if colunas:
            df = base[colunas].copy()
        else:
            df = base.copy()

        df = df.select_dtypes(include=np.number)

    # VALIDAÇÃO
    if df.empty:
        raise ValueError(
            "Nenhuma coluna numérica encontrada."
        )

    # FORMATO LONGO PARA PLOT
    dados = (
        df
        .melt(
            var_name="Grupo",
            value_name="Valor"
        )
        .dropna()
    )

    # FIGURA
    fig, ax = plt.subplots(figsize=figsize)

    # PONTOS
    sns.stripplot(
        data=dados,
        x="Grupo",
        y="Valor",
        hue="Grupo",
        palette=palette,
        size=6,
        jitter=jitter,
        legend=False,
        alpha=0.7,
        ax=ax
    )

    # MÉDIA / MEDIANA / DP / IC
    for i, col in enumerate(df.columns):

        serie = df[col].dropna()

        media_col = serie.mean()
        mediana_col = serie.median()
        dp_col = serie.std(ddof=1)

        # MÉDIA
        if media:
            ax.hlines(
                y=media_col,
                xmin=i - 0.25,
                xmax=i + 0.25,
                colors="black",
                linewidth=2,
                label="Média" if i == 0 else ""
            )

        # MEDIANA
        if mediana:
            ax.hlines(
                y=mediana_col,
                xmin=i - 0.20,
                xmax=i + 0.20,
                colors="red",
                linewidth=2,
                linestyles="dashed",
                label="Mediana" if i == 0 else ""
            )

        # DESVIO PADRÃO
        if dp:
            ax.errorbar(
                x=i,
                y=media_col,
                yerr=dp_col,
                fmt="none",
                color="black",
                capsize=5
            )

        # INTERVALO DE CONFIANÇA
        if ic:
            erro = (
                    stats.sem(serie)
                    * stats.t.ppf(
                (1 + alpha_ic) / 2,
                len(serie) - 1
            )
            )

            ax.errorbar(
                x=i,
                y=media_col,
                yerr=erro,
                fmt="none",
                color="blue",
                linewidth=2,
                capsize=6
            )

    # ESTÉTICA
    ax.set_title(
        titulo,
        fontsize=20,
        weight="bold"
    )

    ax.set_xlabel(
        "Grupo",
        fontsize=12
    )

    ax.set_ylabel(
        "Valores",
        fontsize=12
    )

    if grid:
        ax.grid(
            alpha=0.25,
            linestyle="--"
        )

    sns.despine()

    plt.tight_layout()
    plt.show()


# visual usado para verificar a
def v_interacao_posHoc(base, x: str, hue: str, y: str):
    """
    Visual para verificar interação entre fatores.
    """

    plt.figure(figsize=(10, 6))

    dados = base[[x, hue, y]].dropna()

    sns.pointplot(
        data=dados,
        x=x,
        y=y,
        hue=hue,
        errorbar=("ci", 95),
        dodge=0.25,
        palette="bright",
        capsize=.08,
        markers="o",
        linestyles="-"
    )

    plt.title(
        f"Interação: {x} × {hue}",
        fontsize=16,
        weight="bold"
    )

    plt.ylabel(y)
    plt.xlabel(x)

    plt.tight_layout()
    plt.show()



# DISPERSÃO DOS RESÍDUOS
def v_plot_residuos(
        base: pd.DataFrame,
        residuos,
        colunas: str | list[str],
        figsize=(6, 4),
        alpha=0.8
):
    """
    Plota gráficos de dispersão
    entre resíduos e variáveis independentes.

    Parâmetros
    ----------
    base : DataFrame

    residuos : array-like
        Resíduos do modelo.

    colunas : str | list[str]
        Variáveis independentes.

    figsize : tuple
        Tamanho de cada gráfico.

    alpha : float
        Transparência dos pontos.
    """

    # TRANSFORMA EM LISTA
    if isinstance(colunas, str):
        colunas = [colunas]

    # garante 1D
    if isinstance(residuos, pd.DataFrame):
        residuos = residuos.squeeze()

    # QUANTIDADE DE GRÁFICOS
    n = len(colunas)

    fig, axes = plt.subplots(
        nrows=n,
        ncols=1,
        figsize=(figsize[0], figsize[1] * n)
    )

    # CASO TENHA APENAS 1
    if n == 1:
        axes = [axes]

    # PLOTS
    for ax, coluna in zip(axes, colunas):

        dados = pd.DataFrame(
            {
                "x": base[coluna],
                "residuos": residuos
            }
        ).dropna()

        ax.scatter(
            dados["x"],
            dados["residuos"],
            alpha=alpha
        )

        ax.axhline(
            y=0,
            linestyle="--"
        )

        ax.set_xlabel(coluna)

        ax.set_ylabel("Resíduos")

        ax.set_title(
            f"Resíduos vs {coluna}"
        )

    plt.tight_layout()

    plt.show()