from functools import partial
from glob import glob
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import yaml
from pytorch_lightning import loggers, Trainer
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging, ModelCheckpoint

from bracket import generateBracket, scoreBracket, applyResultsToBracket
from load_data import getPossMatches
from model import GameSequencePredictor
from dataloader import GameDataModule
from prep_data import loadFramesForTorch, formatForTorch


def objective(trial: optuna.Trial, config=None):
    config['seq_predictor']['latent_size'] = trial.suggest_int('latent_size', 5, 75, 5)
    config['seq_predictor']['lr'] = trial.suggest_float('lr', 1e-9, 1e-3, log=True)
    config['seq_predictor']['weight_decay'] = trial.suggest_float('weight_decay', 1e-9, .8, log=True)
    config['seq_predictor']['scheduler_gamma'] = trial.suggest_float('scheduler_gamma', .1, .9999)
    config['seq_predictor']['betas'] = [trial.suggest_float('beta0', .7, .9999), trial.suggest_float('beta1', .1, .6)]
    config['seq_predictor']['activation'] = trial.suggest_categorical('activation', ['gelu', 'selu', 'silu'])
    datapath = config['dataloader']['datapath']
    adf, avodf = loadFramesForTorch(datapath)

    season_total = []
    for season in np.random.choice([2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024], 5):
        config['dataloader']['season'] = season
        # First, run the encoding to try and reduce the dimension of the data
        data = GameDataModule(**config['dataloader'])
        data.setup()

        # Get the model, experiment, logger set up
        config['seq_predictor']['init_size'] = data.train_dataset.data_len
        config['seq_predictor']['extra_size'] = data.train_dataset.extra_len
        model = GameSequencePredictor(**config['seq_predictor'])
        trainer = Trainer(logger=False, max_epochs=config['seq_predictor']['training']['max_epochs'],
                          default_root_dir=config['seq_predictor']['training']['weights_path'],
                          log_every_n_steps=config['seq_predictor']['training']['log_epoch'],
                          num_sanity_val_steps=0, detect_anomaly=False, callbacks=
                          [EarlyStopping(monitor='val_loss', patience=config['seq_predictor']['training']['patience'],
                                         check_finite=True)])
        trainer.fit(model, datamodule=data)
        model.eval()

        extra_df, extra0_df = getPossMatches(avodf, season=season, datapath=datapath)
        poss_results = pd.DataFrame(index=extra_df.index, columns=['Res'])
        for i in range(0, extra_df.shape[0], 128):
            block = extra_df.iloc[i:i + 128]
            torch_data = []
            for idx in block.index:
                torch_data.append(formatForTorch(adf, extra_df, extra0_df, season, idx, config['seq_predictor']['in_channels'], .5))
            t_data = torch.cat([c[0].unsqueeze(0) for c in torch_data], dim=0)
            o_data = torch.cat([c[1].unsqueeze(0) for c in torch_data], dim=0)
            tav_data = torch.cat([c[2].unsqueeze(0) for c in torch_data], dim=0)
            oav_data = torch.cat([c[3].unsqueeze(0) for c in torch_data], dim=0)
            predictions = 1 - model(t_data, o_data, tav_data, oav_data).detach().numpy()
            poss_results.loc[block.index, 'Res'] = predictions
        truth_br = generateBracket(season, True, datapath=datapath)
        test = generateBracket(season, True, datapath=datapath)
        res = 0
        for r in range(100):
            test = applyResultsToBracket(test, poss_results, select_random=True, random_limit=.1)
            res += scoreBracket(test, truth_br) / 100.
        season_total.append(res)
    return np.mean(season_total)


if __name__ == '__main__':

    with open('./run_params.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)


    study = optuna.create_study(direction='maximize',
                                storage='sqlite:///db.sqlite3',
                                study_name='madness')
    objective = partial(objective, config=config)
    study.optimize(objective, 1000)

