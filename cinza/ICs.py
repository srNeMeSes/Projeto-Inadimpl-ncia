from scipy.stats import t
from math import sqrt
import numpy as np
from scipy.stats import norm


def ic_media_amostra(amostra, confianca=0.95):
    """
    Calcula o intervalo de confiança para a média populacional
    com base em uma amostra.

    Parameters
    ----------
    amostra : array-like
        Dados da amostra.
    confianca : float
        Nível de confiança (0.95 = 95%).

    Returns
    -------
    dict {
        media
        n
        erro_padrao
        margem_erro
        nivel_confianca
        intervalo_confianca
        }
    """

    amostra = np.asarray(amostra)

    n = len(amostra)
    media = np.mean(amostra)
    desvio = np.std(amostra, ddof=1)

    return ic_media(media=media, dp=desvio, n=n, confianca=confianca)


def ic_media(media: float, dp: float, n: int, confianca: float = 0.95):
    """
        Calcula o intervalo de confiança para a média populacional.

        Parameters
        ----------
        media : float
            Média da amostra.
        dp : float
            Desvio padrão da amostra.
        n : int
            Tamanho da amostra.
        confianca : float
            Nível de confiança (0.95 = 95%).

       Returns
       -------
        dict {
            media
            n
            erro_padrao
            margem_erro
            nivel_confianca
            intervalo_confianca
        }
       """
    alpha = 1 - confianca

    t_critico = t.ppf(
        q=1 - alpha / 2,
        df=n - 1
    )

    erro_padrao = dp / sqrt(n)

    margem_erro = t_critico * erro_padrao

    li = media - margem_erro
    ls = media + margem_erro

    return {
        "media": media,
        "n": n,
        "erro_padrao": erro_padrao,
        "margem_erro": margem_erro,
        "nivel_confianca": confianca,
        "intervalo_confianca": (li, ls),
        "interpretacao":
            f"A média populacional foi estimada em {media:.2f} unidades,"
            f" com intervalo de confiança de {confianca * 100:.2f}%"
            f" variando de {li:.2f} a {ls:.2f} e margem de erro igual a {margem_erro:.2f} unidades."
    }


def ic_proporcao(sucessos, n, confianca=0.95):
    """
    Calcula o intervalo de confiança para uma proporção populacional.

    Parameters
    ----------
    sucessos : int
        Quantidade de sucessos observados.
    n : int
        Tamanho da amostra.
    confianca : float
        Nível de confiança.

    Returns
    -------
    dict
    """

    p = sucessos / n

    alpha = 1 - confianca

    z_critico = norm.ppf(
        q=1 - alpha / 2
    )

    erro_padrao = sqrt(
        p * (1 - p) / n
    )

    margem_erro = z_critico * erro_padrao

    li = p - margem_erro
    ls = p + margem_erro

    return {
        "p": p,
        "n": n,
        "erro_padrao": erro_padrao,
        "margem_erro": margem_erro,
        "nivel_confianca": confianca,
        "intervalo_confianca": (max(0, li), min(1, ls)),
        "interpretacao":
        f"A proporção populacional foi estimada em {p*100:.2f}%,"
        f" com intervalo de confiança de {confianca * 100:.2f}%"
        f" variando de {max(0, li)*100:.2f}% a {min(1, ls)*100:.2f}%"
        f" e margem de erro de {margem_erro*100:.2f} pontos percentuais."
    }
