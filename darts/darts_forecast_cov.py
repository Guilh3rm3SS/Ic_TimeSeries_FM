

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from darts.models import RNNModel
from sklearn.preprocessing import MaxAbsScaler
import optuna
from pytorch_lightning.callbacks import EarlyStopping
from darts.metrics import smape
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.utils.utils import generate_index


class CustomRNNModel(RNNModel):
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4)

        # Warmup de 3 epochs
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=3)
        # Cosine decay após warmup
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=self.n_epochs - 3)

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[3]
        )

        return [optimizer], [scheduler]

def make_scheduler_factory(epochs):
    def scheduler_factory(optimizer):
        return SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.1, total_iters=3),
                CosineAnnealingLR(optimizer, T_max=epochs - 3)
            ],
            milestones=[3]
        )
    return scheduler_factory


def train_model_hyperparam_search(
    train,
    val,
    covariate_train,
    covariate_val,
    context_len=30,
    horizon_len=1,
    model_type='LSTM',
    epochs=40,
    n_trials=20
):

    def objective(trial):
        hidden_dim = trial.suggest_int("hidden_dim", 16, 128, log=True)
        n_rnn_layers = trial.suggest_int("n_rnn_layers", 1, 4)
        dropout = trial.suggest_float("dropout", 0.0, 0.5)

        early_stopper = EarlyStopping(monitor="val_loss", patience=5, verbose=False)

        model = CustomRNNModel(
            model=model_type,
            training_length=int((context_len + horizon_len) * 1.5),
            input_chunk_length=context_len,
            output_chunk_length=horizon_len,
            hidden_dim=hidden_dim,
            n_rnn_layers=n_rnn_layers,
            dropout=dropout,
            batch_size=32,
            n_epochs=epochs,
            random_state=42,
            pl_trainer_kwargs={"callbacks": [early_stopper]},
        )

        model.fit(series=train,  val_series=val, past_covariates=covariate_train,  val_past_covariates=covariate_val, verbose=False)
        preds = model.predict(n=horizon_len, series=train)
        return smape(val, preds)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    print("Melhores parâmetros:", study.best_trial.params)


    # Otimização com Optuna
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    print("Melhores parâmetros encontrados:")
    print(study.best_trial.params)

    # Treinamento com parâmetros otimizados usando apenas o treino
    best_params = study.best_trial.params
    best_model = RNNModel(
      model=model_type,
      training_length=int((context_len + horizon_len) * 1.5),
      input_chunk_length=context_len,
      output_chunk_length=horizon_len,
      hidden_dim=best_params["hidden_dim"],
      n_rnn_layers=best_params["n_rnn_layers"],
      dropout=best_params["dropout"],
      batch_size=32,
      n_epochs=epochs,
      random_state=42,
      optimizer_cls=AdamW,
      optimizer_kwargs={"lr": 1e-3, "weight_decay": 1e-4},
      lr_scheduler_cls=make_scheduler_factory(epochs),
      pl_trainer_kwargs={
            "accelerator": "auto",
            "callbacks": [EarlyStopping(monitor="val_loss", patience=5)]
        }
    )   
    
    # full_train = train.concatenate(val)
    best_model.fit(series=train, val_series=val, past_covariates=covariate_train,  val_past_covariates=covariate_val, verbose=True)
    # return best_model, study
    return best_model

def backtest_model(
    test_series,
    past_covariates,
    horizon_len,
    model
):
  forecast_series = model.historical_forecasts(
    test_series,
    forecast_horizon=horizon_len,
    past_covariates=past_covariates,
    retrain=False,
    verbose=True,
    last_points_only=True,
    stride=1
  )

  return forecast_series

def get_covariate_dictionary():
    all_covariates = [
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

    only_level_covs = [
        "Bonete_level", "Manuel_Díaz_level", "Cuñapirú_level", "Mazangano_level", "Coelho_level",
        "Paso_de_las_Toscas_level", "Aguiar_level", "Laguna_I_level", "Laguna_II_level", "Pereira_level",
        "San_Gregorio_level", "Paso_de_los_Toros_level", "Salsipuedes_level", "Sarandi_del_Yi_level",
        "Durazno_level", "Polanco_level", "Lugo_level",
    ]

    high_correlation_cov = ["Aguiar_level", "Laguna_I_level", "Laguna_II_level", "Pereira_level", "Paso_de_los_Toros_level", "Durazno_level", "Polanco_level"]

    mercedes_precipitation_cov = ["Mercedes_precipitation"]

    covariate_dict = {
    "all_covariates": all_covariates,
    "only_level_covs": only_level_covs,
    "high_correlation_cov": high_correlation_cov,
    "mercedes_precipitation_cov": mercedes_precipitation_cov,
    }   
    
    return covariate_dict
    

def main():
    url = "https://drive.google.com/uc?id=1qAi5oqUUp-i34MoW5fg_G1o6iFIKlidJ"
    full_df = pd.read_csv(url, index_col=0, parse_dates=True)


    full_df['dt'] = pd.to_datetime(full_df['dt'], format='%d/%m/%Y %H:%M')

    df = full_df[['dt', 'Mercedes_level']]
    target = 'Mercedes_level'
    covariate_dict = get_covariate_dictionary()
    
    series = TimeSeries.from_dataframe(full_df, time_col='dt', value_cols=target + covariate_dict['all_covariates'])

    # Define a coluna target, que será a que queremos prever
    
    # context_lengths = [7, 32, 160, 352]
    # horizon_lengths = [1, 7, 15, 30]

    context_lengths = [7]
    horizon_lengths = [1, 7]

    train, test = series.split_after(0.60)
    train, val = train.split_after(0.70)


    base_result = test[target].to_dataframe()
    base_result.rename(columns={target: "observed"}, inplace=True)


    scaler = Scaler(MaxAbsScaler())
    train = scaler.fit_transform(train)
    val = scaler.transform(val)
    test = scaler.transform(test)

    
    for cov_name, cov_set in covariate_dict.items():
        for context_len in context_lengths:
            for horizon_len in horizon_lengths:
            
                if ((1.5*(context_len + horizon_len)) + 1) > len(val):
                    continue
                
                
                print(f"Context length: {context_len}, Horizon length: {horizon_len}")
                column_name = f"{cov_name}_{context_len}_{horizon_len}"
                model = train_model_hyperparam_search(train[target], val[target], train[cov_set], val[cov_set], context_len=context_len, horizon_len=horizon_len, model_type='LSTM', epochs=2)
                forecast = backtest_model(test[target], test[cov_set], horizon_len, model)
                forecast = scaler.inverse_transform(forecast)
                forecast_df = forecast.to_dataframe()
                forecast_df.rename(columns={"Mercedes_level": column_name}, inplace=True)
                base_result = base_result.merge(forecast_df, on="dt", how="left")

    print(base_result)
    path = 'Mercedes_LSTM_opt_40_earlys_with_cov.csv'

    # Salva como CSV
    base_result.to_csv(path, index=True)
    


if __name__ == "__main__":
    main()
