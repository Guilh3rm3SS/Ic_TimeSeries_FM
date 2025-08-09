
# pip3 install timesfm[torch]

from collections import defaultdict
import numpy as np
import torch
import timesfm
import pandas as pd

# Cria dataframe
url = "https://drive.google.com/uc?id=1qAi5oqUUp-i34MoW5fg_G1o6iFIKlidJ"
df = pd.read_csv(url, index_col=0, parse_dates=True)

# Converte a coluna de data do df para datetime
df['dt'] = pd.to_datetime(df['dt'], format='%d/%m/%Y %H:%M')
df = df.drop(df.columns[20], axis=1)

# Divide a série temporal
# 2192 - 1534 = 658 + 500 = 1158
split_index = int(len(df) * 0.7)
split_index = split_index - 500  # Ajusta o índice para considerar o contexto de 500 pontos
df = df.iloc[split_index:]
# df.set_index('dt', inplace=True)

output_path = input("Insira o caminho para salvar o CSV: ")
if not output_path.endswith('.csv'):
    output_path += '.csv'

# Define as colunas do df que serão consideradas como covariáveis, caso não haja, pode ficar vazio
covariables = [
    "Bonete_level", "Manuel_Díaz_level", "Cuñapirú_level", "Mazangano_level", "Coelho_level",
    "Paso_de_las_Toscas_level", "Aguiar_level", "Laguna_I_level", "Laguna_II_level", "Pereira_level",
    "San_Gregorio_level", "Paso_de_los_Toros_level", "Salsipuedes_level", "Sarandi_del_Yi_level",
    "Durazno_level", "Polanco_level", "Lugo_level",
    "Bonete_precipitation", "Manuel_Diaz_precipitation", "Cuñapirú_precipitation", "Mazagano_precipitation",
    "Coelho_precipitation", "Paso_de_las_Toscas_precipitation", "Aguiar_precipitation", "Laguna_I_precipitation",
    "Laguna_II_precipitation", "Pereira_precipitation", "San_Gregorio_precipitation",
    "Paso_de_los_toros_precipitation", "Salsipuedes_precipitation", "Sarandi_del_Yi_precipitation",
    "Polanco_precipitation", "Durazno_precipitation", "Paso_de_Lugo_precipitation",
    "Mercedes_precipitation"
]

only_levels = [
    "Bonete_level", "Manuel_Díaz_level", "Cuñapirú_level", "Mazangano_level", "Coelho_level",
    "Paso_de_las_Toscas_level", "Aguiar_level", "Laguna_I_level", "Laguna_II_level", "Pereira_level",
    "San_Gregorio_level", "Paso_de_los_Toros_level", "Salsipuedes_level", "Sarandi_del_Yi_level",
    "Durazno_level", "Polanco_level", "Lugo_level",
]

mercedes_precipitation = ["Mercedes_precipitation"]

mercedes_precipitation_lagged = ["Mercedes_precipitation_lagged"]

# Define a coluna target, que será a que queremos prever
target = 'Mercedes_level'

import timesfm
# Loading the timesfm-2.0 checkpoint:
# For Torch
# Carrega o checkpoint do timesFM, não é necessário editar
model = timesfm.TimesFm(
      hparams=timesfm.TimesFmHparams(
          backend="gpu",
          per_core_batch_size=32,
          horizon_len=30,
          num_layers=50,
          use_positional_embedding=False,
          context_len=2048,
      ),
      checkpoint=timesfm.TimesFmCheckpoint(
          huggingface_repo_id="google/timesfm-2.0-500m-pytorch"),
  )

#Função para prever o valor das covariáveis históricas
def forecast_historic_covariates(input_ts, horizon_len):
  forecast, _ = model.forecast(inputs=[input_ts], freq=[0] * len(input_ts))
#   print("finished covar", flush=True)
  return forecast[:horizon_len]

# Função que separa os dados em batches
def get_batched_data_fn(
    df, covariables, target,
    batch_size,
    context_len,
    horizon_len,
):

    examples = defaultdict(list)
    num_examples = 0
    # Itera pelos timepoints definindo os pontos de contexto e os de horizonte (Os que devem ser previstos)
    for start in range(500 - context_len, len(df) - (context_len + horizon_len), 1):
        num_examples += 1

        # Pega os pontos de contexto (Input) do Target
        examples["inputs"].append(df[target].iloc[start:start + context_len].values.tolist())
        # Pega os pontos de horizonte (Output) do Target
        examples["outputs"].append(df[target].iloc[start + context_len:start + context_len + horizon_len].tolist())
        # Pega os datetimes do Target
        examples["dt"].append(df["dt"].iloc[start + context_len:start + context_len + horizon_len].tolist())

        # Pega os pontos para cada uma das covariáveis (O número de pontos das covariáveis é igual a soma dos pontos do input e output do exemplo)
        for covar in covariables:
            hist_covar = df[covar].iloc[start:start + context_len].values.tolist()
            future_covar = forecast_historic_covariates(hist_covar, horizon_len)

            future_covar_flattened = np.array(future_covar).ravel()
            examples[covar].append(np.concatenate([np.array(hist_covar), future_covar_flattened]).tolist()) # Convert back to list for appending


    # Função que separa os exemplos em batches de acordo com o batch size
    def data_fn():
        for i in range(1 + (num_examples - 1) // batch_size):
            yield {k: v[(i * batch_size) : ((i + 1) * batch_size)] for k, v in examples.items()}

    return data_fn

"""Definição das Funções para avaliar o erro do modelo"""

from scipy.stats import linregress

def mse(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    return np.mean((y_pred - y_true) ** 2)

def mae(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    return np.mean(np.abs(y_pred - y_true))

def r_squared(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1 - (ss_res / ss_tot)

def rmse(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    return np.sqrt(np.mean((y_pred - y_true) ** 2))

def nse(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    if denominator == 0:
        return np.nan
    return 1 - (numerator / denominator)

def pbias(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    total_error = np.sum(y_pred - y_true)
    total_true = np.sum(y_true)
    if total_true == 0:
        return np.nan
    return 100 * (total_error / total_true)

def ve(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    numerator = np.sum(np.abs(y_pred - y_true))
    denominator = np.sum(y_true)
    if denominator == 0:
        return np.nan
    return 1 - (numerator / denominator)


def p_value(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    slope, intercept, r_value, p_value, std_err = linregress(y_true, y_pred)
    return p_value

"""Fazer teste de previsão com covariáveis"""

import time

def covariable_forecast(
    df, covariables, target,
    batch_size,
    context_len,
    horizon_len,
    forecast_name = "forecast",
    input_data = None
):
    # # Obter dados
    # input_data = get_batched_data_fn(
    #     df,
    #     covariables=covariables,
    #     target=target,
    #     batch_size=batch_size,
    #     context_len=context_len,
    #     horizon_len=horizon_len
    # )

    all_forecasts = []
    all_targets = []
    all_dt = []

    # Loop que passa por cada batch
    for i, example in enumerate(input_data()):
        # Adicionar as listas de timepoints das covariáveis a um dicionário
        covariables_dict = {covar: example[covar] for covar in covariables}

        # Tempo de início para benchmarking
        start_time = time.time()

        # Previsão com covariáveis
        cov_forecast, _ = model.forecast_with_covariates(
            inputs=example["inputs"],  # Pontos de contexto do Target
            dynamic_numerical_covariates=covariables_dict,
            dynamic_categorical_covariates={},
            static_numerical_covariates={},
            static_categorical_covariates={},
            freq=[0] * len(example["inputs"]),
            xreg_mode="xreg + timesfm",
            ridge=0.0,
            force_on_cpu=False,
            normalize_xreg_target_per_input=True
        )

        cov_forecast = np.array(cov_forecast)

        # Acumular previsões e outputs reais
        all_forecasts.append(cov_forecast[:, -1])
        all_targets.append(np.array(example["outputs"])[:, -1]) # Convert to numpy array first for slicing
        all_dt.append(np.array(example["dt"])[:, -1])

        print(f"\rFinished batch {i} in {time.time() - start_time:.2f} seconds", end="")

    # Concatena todas as previsões e targets
    all_forecasts = np.concatenate(all_forecasts, axis=0).ravel()
    all_targets = np.concatenate(all_targets, axis=0).ravel()
    all_dt = np.concatenate(all_dt, axis=0).ravel()


    return_df = pd.DataFrame({
        "dt": all_dt,
        forecast_name: all_forecasts
    })

    return return_df

"""Fazer teste de previsão sem covariáveis"""

import time


def simple_forecast(
    df, target,
    batch_size,
    context_len,
    horizon_len,
    forecast_name = "forecast"
):
    # Obter dados
    input_data = get_batched_data_fn(df, covariables=[], target=target, batch_size=batch_size, context_len=context_len, horizon_len=horizon_len)

    all_forecasts = []
    all_targets = []
    all_dt = []

    # Loop que passa por cada batch
    for i, example in enumerate(input_data()):
        # Tempo de início para benchmarking
        start_time = time.time()

        # Previsão sem covariáveis, utilizando apenas o target
        raw_forecast, _ = model.forecast(inputs=example["inputs"], freq=[0] * len(example["inputs"]))

        # Acumula previsões e valores reais
        all_forecasts.append(raw_forecast[:, -1])
        all_targets.append(np.array(example["outputs"])[:, -1]) # Convert to numpy array first for slicing
        all_dt.append(np.array(example["dt"])[:, -1])

        print(f"\rFinished batch {i} in {time.time() - start_time:.2f} seconds", end="")

    # Concatena todas as previsões e targets
    all_forecasts = np.concatenate(all_forecasts, axis=0).ravel()
    all_targets = np.concatenate(all_targets, axis=0).ravel()
    all_dt = np.concatenate(all_dt, axis=0).ravel()



    return_df = pd.DataFrame({
        "dt": all_dt,
        forecast_name: all_forecasts
    })

    return return_df

# simple_data = simple_forecast(df, target, 50, 160, 7)
# simple_data

# cov_data = covariable_forecast(df, mercedes_precipitation, target, 200, 160, 7, 'one_covar_name')
# cov_data

# Salvar CSV

# df_ba = pd.DataFrame({"dt": df["dt"], "observed": df[target]})
# df_ba = df_ba.merge(cov_data, on="dt", how="left")

# df_ba.to_csv(output_path, index=False)

# print(f"CSV salvo em: {output_path}")

# from google.colab import drive
# drive.mount('/content/drive')

# hyperparams = [(200, 7, 1), (200, 7, 7), (200, 7, 30), (200, 160, 1), (200, 160, 7), (200, 160, 30), (200, 365, 1), (200, 365, 7), (200, 365, 30), (200, 500, 1), (200, 500, 7), (200, 500, 30)]

# context_values = [7, 30, 170, 227, 365, 399, 433, 494]
# horizon_values = [1, 7, 15, 30]
context_values = [32, 160, 352, 480]
horizon_values = [1, 7, 15, 30]

# DataFrame base com as datas e valores observados
df_base = pd.DataFrame({"dt": df["dt"], "observed": df[target]})

for context_len in context_values:
    for horizon_len in horizon_values:
        print(context_len, horizon_len, flush=True)
        
        # Previsão sem covariáveis
        # define a quantidade de amostras da previsão
        # for i in range(100):
        simple_name = (f"Mercedes_Level_{context_len}_{horizon_len}")
        simple_data = simple_forecast(df, target, 200, context_len, horizon_len, simple_name)
        print(f"Previsão sem covariáveis: {simple_name}", flush=True)
        df_base = df_base.merge(simple_data, on="dt", how="left")
        

# Salvar CSV
df_base.to_csv(output_path, index=False)

print(f"CSV salvo em: {output_path}")

# hyperparams = [(200, 7, 1), (200, 7, 7), (200, 7, 30), (200, 160, 1), (200, 160, 7), (200, 160, 30), (200, 365, 1), (200, 365, 7), (200, 365, 30), (200, 500, 1), (200, 500, 7), (200, 500, 30)]


# # DataFrame base com as datas e valores observados
# df_base = pd.DataFrame({"dt": df["dt"], "observed": df[target]})

# for batch_size, context_len, horizon_len in hyperparams:
#     print(context_len, horizon_len, flush=True)
#     # Previsão sem covariáveis
#     simple_name = (f"Mercedes_Level_no_Covariables_{context_len}_{horizon_len}")
#     simple_data = simple_forecast(df, target, batch_size, context_len, horizon_len, simple_name)
#     print(f"Previsão sem covariáveis: {simple_name}", flush=True)
#     df_base = df_base.merge(simple_data, on="dt", how="left")
    
#     # Obter dados
#     input_data = get_batched_data_fn(
#     df,
#     covariables=covariables,
#     target=target,
#     batch_size=batch_size,
#     context_len=context_len,
#     horizon_len=horizon_len
#     )
    
#     # Previsão com a covariavel de chuva no mercedes
#     one_covar_name = (f"Mercedes_Level_+_Mercedes_Rain_{context_len}_{horizon_len}")
#     one_covar_data = covariable_forecast(df, mercedes_precipitation, target, batch_size, context_len, horizon_len, one_covar_name, input_data)
#     print(f"Previsão com a covariável de chuva no mercedes: {one_covar_name}", flush=True)
#     df_base = df_base.merge(one_covar_data, on="dt", how="left")
#     # Previsão com a covariavel de chuva no mercedes com lag de -4
#     # lag_covar_name = (f"Mercedes_Level_+_Mercedes_Rain_Lag_{context_len}_{horizon_len}")
#     # lag_covar_data = covariable_forecast(df_lagged, mercedes_precipitation_lagged, target, batch_size, context_len, horizon_len, lag_covar_name)
#     # df_base = df_base.merge(lag_covar_data, on="dt", how="left")
#     # Previsão com todas as covariáveis de nível de chuva
#     all_level_name = (f"Mercedes_Level_+_All_Levels_{context_len}_{horizon_len}")
#     all_level_data = covariable_forecast(df, only_levels, target, batch_size, context_len, horizon_len, all_level_name, input_data)
#     print(f"Previsão com todas as covariáveis de nível de rio: {all_level_name}", flush=True)
#     df_base = df_base.merge(all_level_data, on="dt", how="left")
#     # Previsão com todas as covariáveis do data frame
#     all_covar_name = (f"Mercedes_Level_+_All_Levels_+_All_Rain_{context_len}_{horizon_len}")
#     all_covar_data = covariable_forecast(df, covariables, target, batch_size, context_len, horizon_len, all_covar_name, input_data)
#     print(f"Previsão com todas as covariáveis de nível de chuva e precipitação: {all_covar_name}", flush=True)
#     df_base = df_base.merge(all_covar_data, on="dt", how="left")


# # Caminho para salvar no Google Drive (ajuste a pasta se quiser)
# output_path = "Resultados_Mercedes_forecast_Timepoints_window_2.csv"

# # Salvar CSV
# df_base.to_csv(output_path, index=False)

# print(f"CSV salvo em: {output_path}")

