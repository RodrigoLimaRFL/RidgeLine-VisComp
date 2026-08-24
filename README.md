# Ridgeline: Temperatura no Brasil
Visualização em ridgeline plot da temperatura no Brasil por década, região e mês, com [Plotly](https://plotly.com/python/).

## Dataset

[Brazil Weather, Conventional Stations (1961-2019)](https://www.kaggle.com/datasets/saraivaufc/conventional-weather-stations-brazil) — séries diárias de temperatura (média, máxima e mínima) de 265 estações meteorológicas convencionais do INMET/BDMEP, de 1961 a 2019.

## Como rodar

1. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

2. Configure um token de API do Kaggle (Account → Create New API Token em [kaggle.com](https://www.kaggle.com/settings)) e salve o valor em texto puro em:

   ```
   Windows: C:\Users\<voce>\.kaggle\access_token.txt
   Linux/Mac: ~/.kaggle/access_token.txt
   ```

3. Rode o script (baixa o dataset automaticamente na primeira vez):

   ```bash
   python ridgeline_temperatura.py
   ```

Isso gera `ridgeline_temperatura_brasil.html` — abra no navegador para o gráfico interativo (seletores de métrica, região e década).
