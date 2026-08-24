"""
Ridgeline plot interativo - Temperatura no Brasil por decada, regiao e metrica
Seminario 01 - Visualizacao Computacional (SCC0252)

Dataset: "Brazil Weather, Conventional Stations (1961-2019)"
https://www.kaggle.com/datasets/saraivaufc/conventional-weather-stations-brazil

O dataset reune series diarias de varias estacoes meteorologicas
convencionais do INMET espalhadas pelo Brasil, num unico arquivo grande
(~780 MB / 12,2 milhoes de linhas), mais um CSV auxiliar com metadados de
cada estacao (nome no formato "CIDADE - UF"). Este script:

  1. Baixa o dataset do Kaggle (kagglehub, requer um token de API salvo em
     ~/.kaggle/access_token.txt, ~/.kaggle/kaggle.json, ou nas variaveis de
     ambiente KAGGLE_API_TOKEN / KAGGLE_USERNAME+KAGGLE_KEY).
  2. Le a tabela principal (detectando automaticamente as colunas de data e
     temperatura, ja que os nomes variam entre exports do INMET/BDMEP) e o
     CSV de metadados das estacoes, do qual deriva a UF e a regiao de cada
     estacao.
  3. Agrupa os registros por decada, regiao e metrica (media / maxima /
     minima).
  4. Monta um ridgeline plot interativo com Plotly: uma "crista" de
     densidade (KDE) por decada, com dois seletores (Regiao, Metrica) que
     trocam quais cristas ficam visiveis via JavaScript.

Como configurar o Kaggle antes de rodar:
  1. Crie uma conta em kaggle.com e va em Account > Create New API Token.
  2. Salve o token (string) em texto puro em:
       Windows: C:\\Users\\<voce>\\.kaggle\\access_token.txt
       Linux/Mac: ~/.kaggle/access_token.txt
  3. pip install -r requirements.txt
  4. python ridgeline_temperatura.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #

KAGGLE_DATASET = "saraivaufc/conventional-weather-stations-brazil"

# Faixa fisicamente plausivel de temperatura no Brasil (°C), aplicada a cada
# metrica (media/maxima/minima) individualmente. Qualquer valor fora disso e
# tratado como erro de sensor / valor sentinela (ex.: -9999, comum em exports
# do INMET/BDMEP para "sem dado").
TEMP_MIN_VALIDA = -15.0
TEMP_MAX_VALIDA = 48.0

# Numero minimo de leituras validas para uma decada aparecer numa combinacao
# especifica de (metrica, regiao). Combinacoes com poucos dados (ex.: regioes
# com poucas estacoes historicas) simplesmente mostram menos cristas.
MIN_LEITURAS_POR_CELULA = 200

# Limite de pontos usados no KDE de cada crista (deixa o calculo rapido sem
# mudar a forma da curva).
MAX_AMOSTRAS_KDE = 50_000

OUTPUT_HTML = Path(__file__).parent / "ridgeline_temperatura_brasil.html"

RNG_SEED = 42

METRICAS = [("Média", "tmed"), ("Máxima", "tmax"), ("Mínima", "tmin")]

REGIOES_ORDENADAS = ["Brasil (todas)", "Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]

UF_PARA_REGIAO = {
    "AC": "Norte", "AP": "Norte", "AM": "Norte", "PA": "Norte",
    "RO": "Norte", "RR": "Norte", "TO": "Norte",
    "AL": "Nordeste", "BA": "Nordeste", "CE": "Nordeste", "MA": "Nordeste",
    "PB": "Nordeste", "PE": "Nordeste", "PI": "Nordeste", "RN": "Nordeste", "SE": "Nordeste",
    "DF": "Centro-Oeste", "GO": "Centro-Oeste", "MT": "Centro-Oeste", "MS": "Centro-Oeste",
    "ES": "Sudeste", "MG": "Sudeste", "RJ": "Sudeste", "SP": "Sudeste",
    "PR": "Sul", "RS": "Sul", "SC": "Sul",
}


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
# 2. DETECCAO DE COLUNAS / ENCODING
# --------------------------------------------------------------------------- #

def _normalizar(texto: str) -> str:
    texto = str(texto).lower().strip()
    substituicoes = {
        "á": "a", "â": "a", "ã": "a", "à": "a",
        "é": "e", "ê": "e",
        "í": "i",
        "ó": "o", "ô": "o", "õ": "o",
        "ú": "u",
        "ç": "c",
    }
    for original, sem_acento in substituicoes.items():
        texto = texto.replace(original, sem_acento)
    return texto


def detectar_colunas(cabecalho: pd.DataFrame) -> dict:
    """
    Os arquivos do INMET/BDMEP variam de export para export (maiusculas,
    acentos, "Temperatura Maxima (C)" vs "TEMPERATURA MAXIMA, DIARIA (°C)"
    etc). Em vez de fixar nomes exatos, procuramos por palavras-chave.
    """
    normalizados = {col: _normalizar(col) for col in cabecalho.columns}

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
        "estacao": procurar("estacao") or procurar("station"),
    }


def detectar_separador_e_encoding(caminho_csv: Path) -> tuple[str, str] | None:
    """
    Descobre encoding/separador lendo so o cabecalho (nrows=0), o que e
    barato mesmo em arquivos gigantes como o deste dataset (~780 MB).
    """
    for encoding in ("utf-8", "latin1"):
        for sep in (";", ","):
            try:
                cabecalho = pd.read_csv(caminho_csv, encoding=encoding, sep=sep, nrows=0)
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
            if cabecalho.shape[1] >= 2:
                return encoding, sep
    return None


# --------------------------------------------------------------------------- #
# 3. CARGA
# --------------------------------------------------------------------------- #

def carregar_dados_brutos(pasta_dataset: Path) -> pd.DataFrame:
    """
    O dataset nao vem "um CSV por estacao": vem como uma unica tabela grande
    com todas as estacoes (conventional_weather_stations_inmet_brazil_*.csv)
    mais CSVs auxiliares de metadados (codigos de estacao, codigos de direcao
    do vento) que nao tem colunas de data/temperatura. Cada CSV encontrado e
    inspecionado pelo cabecalho e os que nao parecem tabelas de medicoes sao
    pulados, em vez de travar o script.
    """
    arquivos_csv = sorted(pasta_dataset.rglob("*.csv"))
    if not arquivos_csv:
        raise FileNotFoundError(
            f"Nenhum CSV encontrado em {pasta_dataset}. Verifique o download."
        )

    print(f"Encontrados {len(arquivos_csv)} arquivo(s) CSV. Inspecionando...")

    partes = []

    for caminho in arquivos_csv:
        config = detectar_separador_e_encoding(caminho)
        if config is None:
            print(f"  [pulado] {caminho.name}: nao foi possivel ler o arquivo.")
            continue
        encoding, sep = config

        cabecalho = pd.read_csv(caminho, encoding=encoding, sep=sep, nrows=0)
        colunas = detectar_colunas(cabecalho)

        if colunas["data"] is None or not (colunas["tmed"] or colunas["tmax"] or colunas["tmin"]):
            print(
                f"  [pulado] {caminho.name}: sem colunas de data/temperatura "
                "reconheciveis (provavelmente um arquivo auxiliar de metadados)."
            )
            continue

        colunas_uteis = sorted(
            {colunas[chave] for chave in ("data", "tmed", "tmax", "tmin", "estacao") if colunas[chave]}
        )
        print(f"  [OK] {caminho.name} -> lendo colunas: {colunas_uteis}")

        df = pd.read_csv(caminho, encoding=encoding, sep=sep, usecols=colunas_uteis, low_memory=False)

        registro = pd.DataFrame()
        registro["data"] = pd.to_datetime(df[colunas["data"]], errors="coerce", dayfirst=True)
        for chave_metrica in ("tmed", "tmax", "tmin"):
            if colunas[chave_metrica]:
                registro[chave_metrica] = pd.to_numeric(
                    df[colunas[chave_metrica]], errors="coerce"
                ).astype("float32")
            else:
                registro[chave_metrica] = np.float32(np.nan)

        registro["estacao"] = df[colunas["estacao"]] if colunas["estacao"] else pd.NA

        print(f"           {len(registro):,} linhas lidas")
        partes.append(registro)

    if not partes:
        raise RuntimeError(
            "Nenhum arquivo com colunas de data/temperatura reconheciveis foi "
            "encontrado no dataset baixado."
        )

    return pd.concat(partes, ignore_index=True)


def carregar_estacoes(pasta_dataset: Path) -> pd.DataFrame:
    """
    Le o CSV auxiliar de metadados das estacoes (nome no formato "CIDADE -
    UF") para poder mapear cada codigo de estacao a uma regiao do Brasil.
    """
    for caminho in sorted(pasta_dataset.rglob("*.csv")):
        config = detectar_separador_e_encoding(caminho)
        if config is None:
            continue
        encoding, sep = config

        cabecalho = pd.read_csv(caminho, encoding=encoding, sep=sep, nrows=0)
        colunas_norm = {col: _normalizar(col) for col in cabecalho.columns}
        col_nome = next((c for c, n in colunas_norm.items() if "nome" in n), None)
        col_codigo = next((c for c, n in colunas_norm.items() if "codigo" in n), None)
        if not col_nome or not col_codigo:
            continue

        df = pd.read_csv(caminho, encoding=encoding, sep=sep, usecols=[col_nome, col_codigo])
        uf = df[col_nome].astype(str).str.rsplit(" - ", n=1).str[-1].str.strip().str.upper()

        estacoes = pd.DataFrame(
            {
                "estacao": df[col_codigo],
                "regiao": uf.map(UF_PARA_REGIAO).fillna("Outras"),
            }
        )
        print(f"Metadados de estacoes carregados de {caminho.name} ({len(estacoes)} estacoes).")
        return estacoes

    print(
        "AVISO: nao encontrei um CSV de metadados de estacoes (colunas "
        "nome/codigo). O filtro por regiao ficara indisponivel."
    )
    return pd.DataFrame(columns=["estacao", "regiao"])


def preparar_dados(dados: pd.DataFrame, estacoes: pd.DataFrame) -> pd.DataFrame:
    dados = dados.dropna(subset=["data"]).copy()

    for coluna in ("tmed", "tmax", "tmin"):
        fora_da_faixa = ~dados[coluna].between(TEMP_MIN_VALIDA, TEMP_MAX_VALIDA)
        dados.loc[fora_da_faixa, coluna] = np.nan

    dados["ano"] = dados["data"].dt.year
    dados["decada"] = (dados["ano"] // 10 * 10).astype(int)
    dados = dados[(dados["decada"] >= 1960) & (dados["decada"] <= 2010)]

    dados = dados.merge(estacoes, on="estacao", how="left")
    dados["regiao"] = dados["regiao"].fillna("Outras")

    print("\nLinhas sem regiao identificada:", int((dados["regiao"] == "Outras").sum()))
    print("\nRegistros validos (temperatura media) por decada:")
    print(dados.loc[dados["tmed"].notna(), "decada"].value_counts().sort_index())

    return dados


# --------------------------------------------------------------------------- #
# 4. RIDGELINE INTERATIVO (PLOTLY + SELETORES)
# --------------------------------------------------------------------------- #

def _titulo_texto(metrica_label: str) -> str:
    return (
        f"Distribuição da temperatura {metrica_label.lower()} diária no Brasil, por década"
        "<br><sup>Fonte: INMET/BDMEP - estações convencionais (1961-2019), via Kaggle</sup>"
    )


def _eixo_x_texto(metrica_label: str) -> str:
    return f"Temperatura {metrica_label.lower()} diária (°C)"


def _gerar_paleta(n: int) -> list[str]:
    """Gradiente do azul (decadas antigas) ao vermelho (decadas recentes)."""
    import plotly.colors as pc

    return pc.sample_colorscale("RdYlBu_r", np.linspace(0.05, 0.95, n))


def montar_ridgeline_interativo(dados: pd.DataFrame) -> tuple[go.Figure, list[str], str]:
    decadas = sorted(dados["decada"].unique())
    espacamento_y = 0.55
    amplitude_maxima = espacamento_y * 2.2
    cores = _gerar_paleta(len(decadas))
    grade_x = np.linspace(TEMP_MIN_VALIDA, TEMP_MAX_VALIDA, 400)
    rng = np.random.default_rng(RNG_SEED)

    fig = go.Figure()
    grupos_por_trace: list[str] = []  # mesmo tamanho de fig.data: a que combinacao metrica|regiao cada trace pertence
    grupo_padrao = f"{METRICAS[0][0]}|{REGIOES_ORDENADAS[0]}"

    for metrica_label, metrica_col in METRICAS:
        for regiao_label in REGIOES_ORDENADAS:
            subset_regiao = (
                dados if regiao_label == "Brasil (todas)" else dados[dados["regiao"] == regiao_label]
            )
            grupo = f"{metrica_label}|{regiao_label}"
            visivel_por_padrao = grupo == grupo_padrao

            for i, decada in enumerate(decadas):
                amostras = subset_regiao.loc[subset_regiao["decada"] == decada, metrica_col].dropna().to_numpy()
                if len(amostras) < MIN_LEITURAS_POR_CELULA:
                    continue
                if len(amostras) > MAX_AMOSTRAS_KDE:
                    amostras = rng.choice(amostras, size=MAX_AMOSTRAS_KDE, replace=False)

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
                        visible=visivel_por_padrao,
                    )
                )
                grupos_por_trace.append(grupo)

                fig.add_trace(
                    go.Scatter(
                        x=grade_x,
                        y=base_y + densidade,
                        mode="lines",
                        line=dict(color=cor, width=1.5),
                        fill="tonexty",
                        fillcolor=cor.replace("rgb", "rgba").replace(")", ", 0.75)"),
                        name=f"{decada}s",
                        showlegend=False,
                        customdata=np.column_stack([np.full_like(grade_x, decada), densidade]),
                        hovertemplate=(
                            "Década: %{customdata[0]:.0f}s<br>"
                            "Temperatura: %{x:.1f} °C<br>"
                            "Densidade relativa: %{customdata[1]:.2f}<extra></extra>"
                        ),
                        visible=visivel_por_padrao,
                    )
                )
                grupos_por_trace.append(grupo)

    metrica_padrao, regiao_padrao = grupo_padrao.split("|")

    fig.update_layout(
        title=_titulo_texto(metrica_padrao),
        xaxis_title=_eixo_x_texto(metrica_padrao),
        yaxis=dict(
            title="Década",
            tickmode="array",
            tickvals=[i * espacamento_y for i in range(len(decadas))],
            ticktext=[f"{d}s" for d in decadas],
        ),
        template="plotly_white",
        hovermode="closest",
        height=650,
        width=980,
        margin=dict(t=90),
    )

    return fig, grupos_por_trace, grupo_padrao


PAGINA_TEMPLATE = """<!doctype html>
<html lang="pt-br">
<head>
<meta charset="utf-8">
<title>Ridgeline - Temperatura no Brasil</title>
<style>
  body { font-family: Arial, Helvetica, sans-serif; margin: 24px; }
  h1 { font-size: 20px; }
  .controles { display: flex; gap: 28px; margin-bottom: 16px; align-items: center; }
  .controles label { font-weight: bold; margin-right: 8px; }
  .controles select { font-size: 14px; padding: 4px 10px; }
</style>
</head>
<body>
  <h1>Ridgeline: distribuição de temperatura no Brasil por década</h1>
  <div class="controles">
    <div>
      <label for="sel-metrica">Métrica</label>
      <select id="sel-metrica">__OPCOES_METRICA__</select>
    </div>
    <div>
      <label for="sel-regiao">Região</label>
      <select id="sel-regiao">__OPCOES_REGIAO__</select>
    </div>
  </div>
  __DIV_PLOTLY__
  <script>
    const grupos = __GRUPOS_JSON__;
    const titulos = __TITULOS_JSON__;
    const eixos = __EIXOS_JSON__;

    const selMetrica = document.getElementById("sel-metrica");
    const selRegiao = document.getElementById("sel-regiao");
    selMetrica.value = "__METRICA_PADRAO__";
    selRegiao.value = "__REGIAO_PADRAO__";

    function atualizarGrafico() {
      const metrica = selMetrica.value;
      const regiao = selRegiao.value;
      const chave = metrica + "|" + regiao;
      const visivel = grupos.map(function (g) { return g === chave; });
      const gd = document.getElementById("grafico");
      Plotly.restyle(gd, { visible: visivel });
      Plotly.relayout(gd, {
        "title.text": titulos[metrica],
        "xaxis.title.text": eixos[metrica],
      });
    }

    selMetrica.addEventListener("change", atualizarGrafico);
    selRegiao.addEventListener("change", atualizarGrafico);
  </script>
</body>
</html>
"""


def montar_pagina_html(fig: go.Figure, grupos_por_trace: list[str], grupo_padrao: str) -> str:
    div_html = fig.to_html(full_html=False, include_plotlyjs="cdn", div_id="grafico")

    opcoes_metrica = "".join(f'<option value="{m}">{m}</option>' for m, _ in METRICAS)
    opcoes_regiao = "".join(f'<option value="{r}">{r}</option>' for r in REGIOES_ORDENADAS)
    metrica_padrao, regiao_padrao = grupo_padrao.split("|")

    titulos = {m: _titulo_texto(m) for m, _ in METRICAS}
    eixos = {m: _eixo_x_texto(m) for m, _ in METRICAS}

    return (
        PAGINA_TEMPLATE.replace("__DIV_PLOTLY__", div_html)
        .replace("__OPCOES_METRICA__", opcoes_metrica)
        .replace("__OPCOES_REGIAO__", opcoes_regiao)
        .replace("__GRUPOS_JSON__", json.dumps(grupos_por_trace, ensure_ascii=False))
        .replace("__TITULOS_JSON__", json.dumps(titulos, ensure_ascii=False))
        .replace("__EIXOS_JSON__", json.dumps(eixos, ensure_ascii=False))
        .replace("__METRICA_PADRAO__", metrica_padrao)
        .replace("__REGIAO_PADRAO__", regiao_padrao)
    )


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #

def main() -> None:
    pasta_dataset = baixar_dataset()
    dados_brutos = carregar_dados_brutos(pasta_dataset)
    estacoes = carregar_estacoes(pasta_dataset)
    dados = preparar_dados(dados_brutos, estacoes)

    if dados["decada"].nunique() < 2:
        raise RuntimeError(
            "Poucas decadas com dados suficientes para o ridgeline. "
            "Revise MIN_LEITURAS_POR_CELULA ou a deteccao de colunas."
        )

    fig, grupos_por_trace, grupo_padrao = montar_ridgeline_interativo(dados)
    html = montar_pagina_html(fig, grupos_por_trace, grupo_padrao)
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"\nGrafico interativo salvo em: {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
