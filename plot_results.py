import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from scipy.stats import linregress
from matplotlib.backends.backend_pdf import PdfPages

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



# Caminho para o arquivo no seu Drive
csv_path = input("Digite o caminho do arquivo CSV: ")

# Carregar com pandas
df_result = pd.read_csv(csv_path)

# Garante que a coluna de data seja interpretada como datetime
df_result.iloc[:, 0] = pd.to_datetime(df_result.iloc[:, 0])
df_result = df_result[500:]

# df_result = df_result.drop(
#     columns=[
#         "Mercedes_Level_no_Covariables_7_30",
#         "Mercedes_Level_+_Mercedes_Rain_7_30",
#         "Mercedes_Level_+_All_Levels_7_30",
#         "Mercedes_Level_+_All_Levels_+_All_Rain_7_30",
#     ]
# )


# === Parte 1: Cálculo das métricas ===

observed_col = df_result.columns[1]
model_cols = df_result.columns[2:]
metric_results = []

for model_col in model_cols:
    forecast = df_result[model_col]
    target = df_result[observed_col]
    valid_idx = (~forecast.isna()) & (~target.isna())
    forecast_clean = forecast[valid_idx].values
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

    metric_results.append((model_col, *metrics))

metrics_df = pd.DataFrame(metric_results, columns=[
    "Modelo", "MAE", "MSE", "R²", "RMSE", "PBIAS (%)", "Volumetric Efficiency", "P-Value"
])
metrics_df.set_index('Modelo', inplace=True)

metrics_df['numero_final'] = metrics_df.index.to_series().apply(
    lambda x: int(re.search(r'(\d+)$', x).group(1)) if re.search(r'(\d+)$', x) else None
)
metrics_df['modelo_base'] = metrics_df.index.to_series().apply(
    lambda x: re.sub(r'[-_ ]?\d+$', '', x)
)

horizon_sizes = [1, 7, 15, 30]
metrics_to_plot = ["MAE", "MSE", "R²", "RMSE", "PBIAS (%)", "Volumetric Efficiency", "P-Value"]

# === Parte 2: Geração dos gráficos e salvamento no PDF ===
pdf_name = input("Digite o nome do arquivo PDF (sem extensão): ")
with PdfPages(f"pdf_outs/{pdf_name}.pdf") as pdf:

    # Gráficos de métricas por horizonte
    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 10))
        for horizon_size in horizon_sizes:
            df_h = metrics_df[metrics_df['numero_final'] == horizon_size]
            plt.plot(df_h['modelo_base'], df_h[metric], marker='o', label=f'Horizonte {horizon_size}')
        if metric == "P-Value":
            plt.yscale('log')
            plt.axhline(y=0.005, color='red', linestyle='--', linewidth=1, label='Limite 0.005')
            plt.ylabel(f'{metric} (log scale)')
        else:
            plt.ylabel(metric)

        plt.title(metric)
        plt.xlabel("Modelo")
        plt.grid(True)
        plt.xticks(rotation=90)
        plt.legend()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    # Gráficos de dispersão com regressão linear
    for horizon in horizon_sizes:
        model_cols = [col for col in df_result.columns if re.search(fr'[_\- ]?{horizon}$', col)]
        for model_col in model_cols:
            forecast = df_result[model_col]
            target = df_result[observed_col]
            valid_idx = (~forecast.isna()) & (~target.isna())
            forecast_clean = forecast[valid_idx].values
            target_clean = target[valid_idx].values

            if len(forecast_clean) > 1:
                slope, intercept, r_value, p_val, std_err = linregress(target_clean, forecast_clean)
                reg_line = slope * target_clean + intercept
                r_squared = r_value**2

                plt.figure(figsize=(8, 8))
                plt.scatter(target_clean, forecast_clean, alpha=0.6, label='Previsões')
                plt.plot(target_clean, reg_line, 'r-', label=f'Regressão Linear\n'
                                                             f'y = {slope:.2f}x + {intercept:.2f}\n'
                                                             f'p = {p_val:.3e}\n'
                                                             f'R² = {r_squared:.3f}')
                min_val = min(target_clean.min(), forecast_clean.min())
                max_val = max(target_clean.max(), forecast_clean.max())
                plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Referência 1:1')

                plt.xlabel("Observado", fontsize=14)
                plt.ylabel("Previsto", fontsize=14)
                plt.title(f"Dispersão: {model_col}", fontsize=16)
                plt.legend(fontsize=12)
                plt.grid(True)
                plt.tight_layout()
                pdf.savefig()
                plt.close()



print(f"Tudo pronto! PDF salvo como '{pdf_name}.pdf'.")