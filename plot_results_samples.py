import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.stats import linregress
from matplotlib.backends.backend_pdf import PdfPages

def mse(y_pred, y_true):
    return np.mean((np.array(y_pred) - np.array(y_true)) ** 2)

def mae(y_pred, y_true):
    return np.mean(np.abs(np.array(y_pred) - np.array(y_true)))

def r_squared(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return np.nan if ss_tot == 0 else 1 - ss_res / ss_tot

def rmse(y_pred, y_true):
    return np.sqrt(mse(y_pred, y_true))

def nse(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    num = np.sum((y_true - y_pred) ** 2)
    den = np.sum((y_true - np.mean(y_true)) ** 2)
    return np.nan if den == 0 else 1 - num / den

def pbias(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    err = np.sum(y_pred - y_true)
    return np.nan if np.sum(y_true) == 0 else 100 * err / np.sum(y_true)

def ve(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    return np.nan if np.sum(y_true) == 0 else 1 - (np.sum(np.abs(y_pred - y_true)) / np.sum(y_true))

def p_value(y_pred, y_true):
    slope, intercept, r_val, p_val, std_err = linregress(y_true, y_pred)
    return p_val

# Parâmetros conhecidos
context_values = [7, 30, 170, 227, 365, 399, 433, 494]
horizon_values = [1, 7, 15, 30]
num_amostras = 100

# Entrada do usuário
csv_path = input("Digite o caminho do arquivo CSV: ")
df_result = pd.read_csv(csv_path)

# Preparação dos dados
df_result.iloc[:, 0] = pd.to_datetime(df_result.iloc[:, 0])
df_result = df_result[500:]
observed_col = df_result.columns[1]

metric_results = []
forecast_dfs = {}  # Para salvar os forecasts médios para os gráficos de dispersão

# === Parte 1: cálculo das métricas ===
for context in context_values:
    for horizon in horizon_values:
        # Regex para pegar todas as colunas correspondentes a esse contexto/horizonte
        padrao = fr"Mercedes_Level_{context}_{horizon}_\d+"
        cols = [col for col in df_result.columns if re.fullmatch(padrao, col)]
        if not cols or len(cols) < num_amostras:
            continue  # pula se não tiver as 100 amostras

        df_preds = df_result[cols].copy()
        df_mean = df_preds.mean(axis=1)

        target = df_result[observed_col]
        valid_idx = (~df_mean.isna()) & (~target.isna())
        forecast_clean = df_mean[valid_idx].values
        target_clean = target[valid_idx].values

        if len(forecast_clean) > 0:
            metrics = (
                mae(forecast_clean, target_clean),
                mse(forecast_clean, target_clean),
                r_squared(forecast_clean, target_clean),
                rmse(forecast_clean, target_clean),
                pbias(forecast_clean, target_clean),
                ve(forecast_clean, target_clean),
                p_value(forecast_clean, target_clean)
            )
        else:
            metrics = (np.nan,) * 7

        model_name = f"Mercedes_{context}_{horizon}"
        metric_results.append((model_name, context, horizon, *metrics))
        forecast_dfs[model_name] = (forecast_clean, target_clean)

# Criar DataFrame com resultados
metrics_df = pd.DataFrame(metric_results, columns=[
    "Modelo", "Contexto", "Horizonte", "MAE", "MSE", "R²", "RMSE", "PBIAS (%)", "Volumetric Efficiency", "P-Value"
])
metrics_df.set_index("Modelo", inplace=True)

# === Parte 2: Geração dos gráficos ===
pdf_name = input("Digite o nome do arquivo PDF (sem extensão): ")
with PdfPages(f"{pdf_name}.pdf") as pdf:

    metrics_to_plot = ["MAE", "MSE", "R²", "RMSE", "PBIAS (%)", "Volumetric Efficiency", "P-Value"]

    # Gráficos de métricas por horizonte
    for metric in metrics_to_plot:
        plt.figure(figsize=(12, 8))
        for horizon in horizon_values:
            df_h = metrics_df[metrics_df['Horizonte'] == horizon]
            plt.plot(df_h['Contexto'], df_h[metric], marker='o', label=f'Horizonte {horizon}')
        if metric == "P-Value":
            plt.yscale('log')
            plt.axhline(y=0.005, color='red', linestyle='--', linewidth=1, label='Limite 0.005')
            plt.ylabel(f'{metric} (log scale)')
        else:
            plt.ylabel(metric)
        plt.xlabel("Contexto")
        plt.title(f'{metric} por Contexto e Horizonte')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    # Gráficos de dispersão
    for model_name, (forecast_clean, target_clean) in forecast_dfs.items():
        if len(forecast_clean) > 1:
            slope, intercept, r_val, p_val, _ = linregress(target_clean, forecast_clean)
            reg_line = slope * target_clean + intercept
            r2 = r_val**2

            plt.figure(figsize=(8, 8))
            plt.scatter(target_clean, forecast_clean, alpha=0.6, label='Previsões')
            plt.plot(target_clean, reg_line, 'r-', label=f'Regressão Linear\n'
                                                         f'y = {slope:.2f}x + {intercept:.2f}\n'
                                                         f'p = {p_val:.3e}\n'
                                                         f'R² = {r2:.3f}')
            min_val = min(target_clean.min(), forecast_clean.min())
            max_val = max(target_clean.max(), forecast_clean.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Referência 1:1')

            plt.xlabel("Observado", fontsize=14)
            plt.ylabel("Previsto (média 100 amostras)", fontsize=14)
            plt.title(f"Dispersão: {model_name}", fontsize=16)
            plt.legend(fontsize=12)
            plt.grid(True)
            plt.tight_layout()
            pdf.savefig()
            plt.close()

    # Gráfico de área de probabilidade
    for context in context_values:
        for horizon in horizon_values:
            padrao = fr"Mercedes_Level_{context}_{horizon}_\d+"
            cols = [col for col in df_result.columns if re.fullmatch(padrao, col)]
            if len(cols) < num_amostras:
                continue

            preds = df_result[cols]
            dates = df_result['dt']  # primeira coluna
            target = df_result[observed_col]

            lower = preds.quantile(0.05, axis=1)
            upper = preds.quantile(0.95, axis=1)
            mean = preds.mean(axis=1)

            plt.figure(figsize=(14, 5))
            plt.plot(dates, target, label="Observado", color='black', linewidth=1.5)
            plt.plot(dates, mean, label="Média previsão", color='blue', linestyle='--')
            plt.fill_between(dates, lower, upper, color='blue', alpha=0.3, label='Intervalo 90%')

            plt.title(f"Contexto {context}, Horizonte {horizon} dias")
            plt.xlabel("Data")
            plt.ylabel("Nível (m)")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            pdf.savefig()
            plt.close()
            
print(f"Tudo pronto! PDF salvo como '{pdf_name}.pdf'.")
