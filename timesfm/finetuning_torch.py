from os import path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
# import yfinance as yf
from finetuning.finetuning_torch import FinetuningConfig, TimesFMFinetuner
from huggingface_hub import snapshot_download
from torch.utils.data import Dataset

from timesfm import TimesFm, TimesFmCheckpoint, TimesFmHparams
from timesfm.pytorch_patched_decoder import PatchedTimeSeriesDecoder
import os
from collections import defaultdict
import time
import wandb


url = "https://drive.google.com/uc?id=1qAi5oqUUp-i34MoW5fg_G1o6iFIKlidJ"
df = pd.read_csv(url, index_col=0, parse_dates=True)
df = df.loc[:, ~df.columns.duplicated()]
df['dt'] = pd.to_datetime(df['dt'], format='%d/%m/%Y %H:%M')

series = df["Mercedes_level"].values



class TimeSeriesDataset(Dataset):
  """Dataset for time series data compatible with TimesFM."""

  def __init__(self,
               series: np.ndarray,
               context_length: int,
               horizon_length: int,
               freq_type: int = 0):
    """
        Initialize dataset.

        Args:
            series: Time series data
            context_length: Number of past timesteps to use as input
            horizon_length: Number of future timesteps to predict
            freq_type: Frequency type (0, 1, or 2)
        """
    if freq_type not in [0, 1, 2]:
      raise ValueError("freq_type must be 0, 1, or 2")

    self.series = series
    self.context_length = context_length
    self.horizon_length = horizon_length
    self.freq_type = freq_type
    self._prepare_samples()

  def _prepare_samples(self) -> None:
    """Prepare sliding window samples from the time series."""
    self.samples = []
    total_length = self.context_length + self.horizon_length

    for start_idx in range(0, len(self.series) - total_length + 1):
      end_idx = start_idx + self.context_length
      x_context = self.series[start_idx:end_idx]
      x_future = self.series[end_idx:end_idx + self.horizon_length]
      self.samples.append((x_context, x_future))

  def __len__(self) -> int:
    return len(self.samples)

  def __getitem__(
      self, index: int
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_context, x_future = self.samples[index]

    x_context = torch.tensor(x_context, dtype=torch.float32)
    x_future = torch.tensor(x_future, dtype=torch.float32)

    input_padding = torch.zeros_like(x_context)
    freq = torch.tensor([self.freq_type], dtype=torch.long)

    return x_context, input_padding, freq, x_future

def prepare_datasets(series: np.ndarray,
                     context_length: int,
                     horizon_length: int,
                     freq_type: int = 0,
                     train_split: float = 0.8) -> Tuple[Dataset, Dataset, Dataset]:
  """
    Prepare training and validation datasets from time series data.

    Args:
        series: Input time series data
        context_length: Number of past timesteps to use
        horizon_length: Number of future timesteps to predict
        freq_type: Frequency type (0, 1, or 2)
        train_split: Fraction of data to use for training

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
  train_size = int(len(series) * train_split)
  train_limit = int(train_size * 0.5)
  train_data = series[:train_limit]
  val_data = series[train_limit:train_size]
  test_data = series[train_size:]

  # Create datasets with specified frequency type
  train_dataset = TimeSeriesDataset(train_data,
                                    context_length=context_length,
                                    horizon_length=horizon_length,
                                    freq_type=freq_type)

  val_dataset = TimeSeriesDataset(val_data,
                                  context_length=context_length,
                                  horizon_length=horizon_length,
                                  freq_type=freq_type)

  test_dataset = TimeSeriesDataset(test_data,
                                   context_length=context_length,
                                   horizon_length=horizon_length,
                                   freq_type=freq_type)

  return train_dataset, val_dataset, test_dataset

"""### Model Creation"""

def get_model(load_weights: bool = False, horizon_len: int = 30, context_len: int = 32):
  device = "cuda" if torch.cuda.is_available() else "cpu"
  repo_id = "google/timesfm-2.0-500m-pytorch"
  hparams = TimesFmHparams(
      backend=device,
      per_core_batch_size=32,
      horizon_len=horizon_len,
      num_layers=50,
      use_positional_embedding=False,
      context_len=context_len,  # Context length can be anything up to 2048 in multiples of 32
  )
  tfm = TimesFm(hparams=hparams,
                checkpoint=TimesFmCheckpoint(huggingface_repo_id=repo_id))

  model = PatchedTimeSeriesDecoder(tfm._model_config)
  if load_weights:
    checkpoint_path = path.join(snapshot_download(repo_id), "torch_model.ckpt")
    loaded_checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(loaded_checkpoint)
  return model, hparams, tfm._model_config

def plot_predictions(
    model: TimesFm,
    val_dataset: Dataset,
    save_path: Optional[str] = "predictions.png",
) -> None:
  """
    Plot model predictions against ground truth for a batch of validation data.

    Args:
      model: Trained TimesFM model
      val_dataset: Validation dataset
      save_path: Path to save the plot
    """
  import matplotlib.pyplot as plt

  model.eval()

  x_context, x_padding, freq, x_future = val_dataset[0]
  x_context = x_context.unsqueeze(0)  # Add batch dimension
  x_padding = x_padding.unsqueeze(0)
  freq = freq.unsqueeze(0)
  x_future = x_future.unsqueeze(0)

  device = next(model.parameters()).device
  x_context = x_context.to(device)
  x_padding = x_padding.to(device)
  freq = freq.to(device)
  x_future = x_future.to(device)

  with torch.no_grad():
    predictions = model(x_context, x_padding.float(), freq)
    predictions_mean = predictions[..., 0]  # [B, N, horizon_len]
    last_patch_pred = predictions_mean[:, -1, :]  # [B, horizon_len]

  context_vals = x_context[0].cpu().numpy()
  future_vals = x_future[0].cpu().numpy()
  # pred_vals = last_patch_pred[0].cpu().numpy()
  pred_vals = predictions_q05[0, :, -1].cpu().numpy()  # [N]


  context_len = len(context_vals)
  horizon_len = len(future_vals)

  plt.figure(figsize=(12, 6))

  plt.plot(range(context_len),
           context_vals,
           label="Historical Data",
           color="blue",
           linewidth=2)

  plt.plot(
      range(context_len, context_len + horizon_len),
      future_vals,
      label="Ground Truth",
      color="green",
      linestyle="--",
      linewidth=2,
  )

  plt.plot(range(context_len, context_len + horizon_len),
           pred_vals,
           label="Prediction",
           color="red",
           linewidth=2)

  plt.xlabel("Time Step")
  plt.ylabel("Value")
  plt.title("TimesFM Predictions vs Ground Truth")
  plt.legend()
  plt.grid(True)

  if save_path:
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

  plt.close()
  
  print(len(future_vals))
  return {
      "context": context_vals,
      "future": future_vals,
      "prediction": pred_vals
  }

def get_data(context_len: int,
             horizon_len: int,
             freq_type: int = 0) -> Tuple[Dataset, Dataset, Dataset]:

  #df = yf.download("AAPL", start="2010-01-01", end="2019-01-01")
  #time_series = df["Close"].values

  train_dataset, val_dataset, test_dataset = prepare_datasets(
      series=series,
      context_length=context_len,
      horizon_length=horizon_len,
      freq_type=freq_type,
      train_split=0.5,
  )

  print(f"Created datasets:")
  print(f"- Training samples: {len(train_dataset)}")
  print(f"- Validation samples: {len(val_dataset)}")
  print(f"- Test samples: {len(test_dataset)}")
  print(f"- Using frequency type: {freq_type}")
  return train_dataset, val_dataset, test_dataset



def sliding_forecast_last_step(
    model: torch.nn.Module,
    dataset: Dataset,
    df_with_dt: pd.DataFrame,
    target_col: str = "Mercedes_level",
    prediction_name: str = "prediction"
    
) -> pd.DataFrame:
    """
    Faz previsões deslizantes a partir de um dataset (ex: val_dataset) e retorna DataFrame com dt, real e predito.

    Args:
        model: Modelo TimesFM treinado.
        dataset: Dataset do tipo TimeSeriesDataset (já com contexto/horizonte/freq_type configurados).
        df_with_dt: DataFrame original contendo a coluna 'dt'.
        target_col: Nome da coluna de valores reais no DataFrame original.

    Returns:
        DataFrame com colunas [dt, real, prediction].
    """
    model.eval()
    device = next(model.parameters()).device

    records = []

    # Offset de datas no DataFrame original
    context_len = dataset.context_length
    horizon_len = dataset.horizon_length
    total_offset = context_len + horizon_len

    for i in range(len(dataset)):
        x_context, x_padding, freq, x_future = dataset[i]

        x_context = x_context.unsqueeze(0).to(device)
        x_padding = x_padding.unsqueeze(0).to(device)
        freq = freq.unsqueeze(0).to(device)
        x_future = x_future.unsqueeze(0).to(device)

        with torch.no_grad():
            predictions = model(x_context, x_padding.float(), freq)
            pred_q50 = predictions[..., 1]  # pega quantil 0.5
            last_pred = pred_q50[:, -1, -1]  # último valor do último patch

        pred_val = last_pred.item()
        real_val = x_future[0, -1].item()

        # Índice da data correspondente ao último valor previsto
        dt_idx = i + total_offset - 1
        if dt_idx < len(df_with_dt):
            dt = df_with_dt.iloc[dt_idx]["dt"]
            records.append({
                "dt": dt,
                # "observed": real_val,
                prediction_name: pred_val
            })

    return pd.DataFrame(records)


  
def single_gpu_example(context_len: int = 32,
                       horizon_len: int = 30,
                       pred_name: str = "prediction"):
  """Basic example of finetuning TimesFM on stock data."""
  model, hparams, tfm_config = get_model(load_weights=True, context_len=context_len, horizon_len=horizon_len)
  config = FinetuningConfig(batch_size=32,
                            num_epochs=6,
                            learning_rate=1e-4,
                            use_wandb=False,
                            # early_stopping_patience=3,
                            freq_type=0,
                            log_every_n_steps=10,
                            val_check_interval=0.5,
                            use_quantile_loss=False)

  train_dataset, val_dataset, test_dataset = get_data(context_len,
                                        tfm_config.horizon_len,
                                        freq_type=config.freq_type)
  finetuner = TimesFMFinetuner(model, config)
  # wandb.run.name = pred_name
  print(f"\nStarting finetuning...{context_len}, {horizon_len}")
  results = finetuner.finetune(train_dataset=train_dataset, val_dataset=val_dataset)

  print("\nFinetuning completed!")
  print(f"Training history: {len(results['history']['train_loss'])} epochs")

  
  df_test = df[len(df)//2:]
  
  rdf = sliding_forecast_last_step(
      model=model,
      dataset=test_dataset,
      df_with_dt=df_test,
      target_col="Mercedes_level",
      prediction_name=pred_name)

  return rdf


def predict_separated_finetunning():
  # tem que ser nultiplos de 32 para o finetunning
  context_values = [32, 160, 352]
  horizon_values = [1, 7, 15, 30]
  # context_values = [32]
  # horizon_values = [1]
  
  
  output_path = "finetuning_results_separated_finetunning_1.csv"
  
  df_base = pd.DataFrame({"dt": df["dt"], "observed": df["Mercedes_level"]})

  
  for context_len in context_values:
    for horizon_len in horizon_values:
      print(f"\nRunning finetuning with context_len={context_len}, horizon_len={horizon_len}")
      simple_name = (f"Mercedes_Level_{context_len}_{horizon_len}")
      simple_data = single_gpu_example(context_len=context_len, horizon_len=horizon_len, pred_name=simple_name)
      print(f"Previsão sem covariáveis: {simple_name}", flush=True)
      df_base = df_base.merge(simple_data, on="dt", how="left")
  
  df_base.to_csv(output_path, index=False)

  print(f"CSV salvo em: {output_path}")




def predict_single_finetunning():
  # tem que ser nultiplos de 32 para o finetunning
  context_values = [32, 160, 224, 352, 384, 480]
  horizon_values = [1, 7, 15, 30]
  # context_values = [32]
  # horizon_values = [1]
  df_test = df[len(df)//2:]
  df_base = pd.DataFrame({"dt": df["dt"], "observed": df["Mercedes_level"]})
  output_path = "finetuning_results_single_finetunning_100_e.csv"
  
  
  model, hparams, tfm_config = get_model(load_weights=True, context_len=512, horizon_len=30)
  config = FinetuningConfig(batch_size=32,
                            num_epochs=5,
                            learning_rate=1e-4,
                            use_wandb=False,
                            freq_type=0,
                            log_every_n_steps=10,
                            val_check_interval=0.5,
                            use_quantile_loss=True)
  
  
  finetuner = TimesFMFinetuner(model, config)
  for context_len in context_values:
    for horizon_len in horizon_values:
      train_dataset, val_dataset, test_dataset = get_data(context_len,
                                        tfm_config.horizon_len,
                                        freq_type=config.freq_type)
      print(f"\nStarting finetuning...{context_len}, {horizon_len}")
      results = finetuner.finetune(train_dataset=train_dataset, val_dataset=val_dataset)

      print("\nFinetuning completed!")
      print(f"Training history: {len(results['history']['train_loss'])} epochs")
  

  
  for context_len in context_values:
    for horizon_len in horizon_values:
      print(f"\nRunning finetuning with context_len={context_len}, horizon_len={horizon_len}")
      _, _, test_dataset = get_data(context_len,
                                        horizon_len,
                                        freq_type=config.freq_type)
      
      simple_name = (f"Mercedes_Level_{context_len}_{horizon_len}")
      simple_data = sliding_forecast_last_step(
        model=model,
        dataset=test_dataset,
        df_with_dt=df_test,
        target_col="Mercedes_level",
        prediction_name=simple_name)
      print(f"Previsão sem covariáveis: {simple_name}", flush=True)
      df_base = df_base.merge(simple_data, on="dt", how="left")
  
  df_base.to_csv(output_path, index=False)

  print(f"CSV salvo em: {output_path}")

predict_separated_finetunning()
