"""
Ridgeline plot interativo - Temperatura no Brasil por decada
Seminario 01 - Visualizacao Computacional (SCC0252)

Dataset: "Brazil Weather, Conventional Stations (1961-2019)"
https://www.kaggle.com/datasets/saraivaufc/conventional-weather-stations-brazil

O dataset reune series diarias de varias estacoes meteorologicas
convencionais do INMET espalhadas pelo Brasil. Este script:

  1. Baixa o dataset do Kaggle (kagglehub, requer credenciais Kaggle
     configuradas em ~/.kaggle/kaggle.json ou nas variaveis de ambiente
     KAGGLE_USERNAME / KAGGLE_KEY).
  2. Le e concatena os CSVs de todas as estacoes.
  3. Detecta automaticamente as colunas de data e temperatura (os nomes
     variam entre arquivos do INMET/BDMEP), calcula a temperatura media
     diaria e agrupa os registros por decada.
  4. Monta um ridgeline plot interativo com Plotly (uma "crista" de
     densidade por decada, cores indicando a evolucao no tempo).

Como configurar o Kaggle antes de rodar:
  1. Crie uma conta em kaggle.com e va em Account > Create New API Token.
  2. Isso baixa um arquivo kaggle.json. Coloque-o em:
       Windows: C:\\Users\\<voce>\\.kaggle\\kaggle.json
       Linux/Mac: ~/.kaggle/kaggle.json
  3. pip install -r requirements.txt
  4. python ridgeline_temperatura.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #

KAGGLE_DATASET = "saraivaufc/conventional-weather-stations-brazil"

# Faixa fisicamente plausivel de temperatura no Brasil (°C). Qualquer valor
# fora disso e tratado como erro de sensor / valor sentinela (ex.: -9999,
# comum em exports do INMET/BDMEP para "sem dado").
TEMP_MIN_VALIDA = -10.0
TEMP_MAX_VALIDA = 45.0

# Numero minimo de leituras validas para uma decada entrar no grafico.
MIN_LEITURAS_POR_DECADA = 500

# Se a base total for muito grande, limitamos o numero de pontos usados no
# KDE de cada decada (a suavizacao da densidade fica igual, mas mais rapida).
MAX_AMOSTRAS_KDE_POR_DECADA = 200_000

OUTPUT_HTML = Path(__file__).parent / "ridgeline_temperatura_brasil.html"

RNG_SEED = 42


# --------------------------------------------------------------------------- #
# 1. DOWNLOAD DO DATASET
# --------------------------------------------------------------------------- #

def baixar_dataset() -> Path:
    """Baixa (ou reusa o cache local) do dataset via kagglehub."""
    import kagglehub

    print(f"Baixando/verificando dataset '{KAGGLE_DATASET}' via kagglehub...")
    caminho = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    print(f"Dataset disponivel em: {caminho}")
    return caminho


# --------------------------------------------------------------------------- #
# 2. DETECCAO DE COLUNAS
# --------------------------------------------------------------------------- #

def detectar_colunas(df: pd.DataFrame) -> dict:
    """
    Os arquivos do INMET/BDMEP variam de estacao para estacao (maiusculas,
    acentos, "Temperatura Maxima (C)" vs "TEMPERATURA MAXIMA, DIARIA (°C)"
    etc). Em vez de fixar nomes exatos, procuramos por palavras-chave.
    """
    normalizados = {
        col: (
            str(col)
            .lower()
            .strip()
            .replace("á", "a").replace("â", "a").replace("ã", "a")
            .replace("é", "e").replace("ê", "e")
            .replace("í", "i")
            .replace("ó", "o").replace("ô", "o")
            .replace("ú", "u")
            .replace("ç", "c")
        )
        for col in df.columns
    }

    def procurar(*palavras_chave: str) -> str | None:
        for original, norm in normalizados.items():
            if all(p in norm for p in palavras_chave):
                return original
        return None

    return {
        "data": procurar("data") or procurar("date"),
        "tmax": procurar("temp", "max") or procurar("tmax"),
        "tmin": procurar("temp", "min") or procurar("tmin"),
        "tmed": procurar("temp", "med") or procurar("temp", "compensada"),
        "estacao": procurar("estacao") or procurar("nome") or procurar("station"),
    }


# --------------------------------------------------------------------------- #
# 3. CARGA E LIMPEZA
# --------------------------------------------------------------------------- #

def ler_um_arquivo(caminho_csv: Path) -> pd.DataFrame | None:
    """Le um CSV de estacao, tentando encodings/separadores comuns do BDMEP."""
    for encoding in ("utf-8", "latin1"):
        for sep in (",", ";"):
            try:
                df = pd.read_csv(
                    caminho_csv, encoding=encoding, sep=sep, low_memory=False
                )
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
            if df.shape[1] >= 2:
                return df
    return None


def carregar_dados_brutos(pasta_dataset: Path) -> pd.DataFrame:
    arquivos_csv = sorted(pasta_dataset.rglob("*.csv"))
    if not arquivos_csv:
        raise FileNotFoundError(
            f"Nenhum CSV encontrado em {pasta_dataset}. Verifique o download."
        )

    print(f"Encontrados {len(arquivos_csv)} arquivos de estacao. Lendo...")

    partes = []
    colunas_referencia = None

    for i, caminho in enumerate(arquivos_csv, start=1):
        df = ler_um_arquivo(caminho)
        if df is None or df.empty:
            continue

        colunas = detectar_colunas(df)
        if colunas_referencia is None:
            colunas_referencia = colunas
            print("Colunas detectadas no primeiro arquivo:")
            for chave, valor in colunas.items():
                print(f"  {chave:8s} -> {valor}")
            if colunas["data"] is None or (
                colunas["tmax"] is None and colunas["tmed"] is None
            ):
                print(
                    "\nAVISO: nao foi possivel detectar automaticamente as "
                    "colunas de data/temperatura. Confira df.columns abaixo "
                    "e ajuste `detectar_colunas` manualmente.\n",
                    list(df.columns),
                )
                sys.exit(1)

        registro = pd.DataFrame()
        registro["data"] = pd.to_datetime(df[colunas["data"]], errors="coerce", dayfirst=True)

        if colunas["tmed"]:
            registro["temp"] = pd.to_numeric(df[colunas["tmed"]], errors="coerce")
        else:
            tmax = pd.to_numeric(df[colunas["tmax"]], errors="coerce")
            tmin = (
                pd.to_numeric(df[colunas["tmin"]], errors="coerce")
                if colunas["tmin"]
                else np.nan
            )
            registro["temp"] = np.where(tmin.notna(), (tmax + tmin) / 2, tmax) if colunas["tmin"] else tmax

        registro["estacao"] = (
            df[colunas["estacao"]].astype(str) if colunas["estacao"] else caminho.stem
        )

        partes.append(registro)

        if i % 50 == 0 or i == len(arquivos_csv):
            print(f"  ... {i}/{len(arquivos_csv)} arquivos processados")

    dados = pd.concat(partes, ignore_index=True)
    return dados


def limpar_e_agrupar_por_decada(dados: pd.DataFrame) -> pd.DataFrame:
    dados = dados.dropna(subset=["data", "temp"]).copy()
    dados = dados[
        (dados["temp"] >= TEMP_MIN_VALIDA) & (dados["temp"] <= TEMP_MAX_VALIDA)
    ]

    dados["ano"] = dados["data"].dt.year
    dados["decada"] = (dados["ano"] // 10 * 10).astype(int)
    dados = dados[dados["decada"] >= 1960]

    contagem = dados["decada"].value_counts()
    decadas_validas = contagem[contagem >= MIN_LEITURAS_POR_DECADA].index
    dados = dados[dados["decada"].isin(decadas_validas)]

    print("\nRegistros validos por decada:")
    print(dados["decada"].value_counts().sort_index())

    return dados


# --------------------------------------------------------------------------- #
# 4. RIDGELINE INTERATIVO (PLOTLY)
# --------------------------------------------------------------------------- #

def montar_ridgeline(dados: pd.DataFrame) -> go.Figure:
    decadas = sorted(dados["decada"].unique())
    rng = np.random.default_rng(RNG_SEED)

    grade_x = np.linspace(TEMP_MIN_VALIDA, TEMP_MAX_VALIDA, 400)

    # Cada crista sobrepoe um pouco a de baixo, no estilo classico ridgeline.
    espacamento_y = 0.55
    amplitude_maxima = espacamento_y * 2.2

    cores = _gerar_paleta(len(decadas))

    fig = go.Figure()

    for i, decada in enumerate(decadas):
        amostras = dados.loc[dados["decada"] == decada, "temp"].to_numpy()
        if len(amostras) > MAX_AMOSTRAS_KDE_POR_DECADA:
            amostras = rng.choice(
                amostras, size=MAX_AMOSTRAS_KDE_POR_DECADA, replace=False
            )

        kde = gaussian_kde(amostras)
        densidade = kde(grade_x)
        densidade = densidade / densidade.max() * amplitude_maxima

        base_y = i * espacamento_y
        cor = cores[i]

        # linha de base invisivel: e contra ela que a crista seguinte
        # (fill="tonexty") preenche, dando o efeito classico do ridgeline.
        fig.add_trace(
            go.Scatter(
                x=grade_x,
                y=np.full_like(grade_x, base_y),
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=grade_x,
                y=base_y + densidade,
                mode="lines",
                line=dict(color=cor, width=1.5),
                fill="tonexty",
                fillcolor=cor.replace("rgb", "rgba").replace(")", ", 0.75)"),
                name=f"{decada}s",
                customdata=np.column_stack(
                    [np.full_like(grade_x, decada), densidade]
                ),
                hovertemplate=(
                    "Decada: %{customdata[0]:.0f}s<br>"
                    "Temperatura: %{x:.1f} °C<br>"
                    "Densidade relativa: %{customdata[1]:.2f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title="Distribuicao da temperatura media diaria no Brasil, por decada<br>"
        "<sup>Fonte: INMET/BDMEP - estacoes convencionais (1961-2019), via Kaggle</sup>",
        xaxis_title="Temperatura media diaria (°C)",
        yaxis=dict(
            title="Decada",
            tickmode="array",
            tickvals=[i * espacamento_y for i in range(len(decadas))],
            ticktext=[f"{d}s" for d in decadas],
        ),
        template="plotly_white",
        hovermode="closest",
        legend_title="Decada (clique para ocultar)",
        height=700,
        width=1000,
    )

    return fig


def _gerar_paleta(n: int) -> list[str]:
    """Gradiente do azul (decadas antigas) ao vermelho (decadas recentes)."""
    import plotly.colors as pc

    escala = pc.sample_colorscale("RdYlBu_r", np.linspace(0.05, 0.95, n))
    return escala


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #

def main() -> None:
    pasta_dataset = baixar_dataset()
    dados_brutos = carregar_dados_brutos(pasta_dataset)
    dados = limpar_e_agrupar_por_decada(dados_brutos)

    if dados["decada"].nunique() < 2:
        raise RuntimeError(
            "Poucas decadas com dados suficientes para o ridgeline. "
            "Revise MIN_LEITURAS_POR_DECADA ou a deteccao de colunas."
        )

    fig = montar_ridgeline(dados)
    fig.write_html(OUTPUT_HTML, include_plotlyjs="cdn")
    print(f"\nGrafico interativo salvo em: {OUTPUT_HTML}")
    fig.show()


if __name__ == "__main__":
    main()
