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
from model import GameSequencePredictor
from dataloader import GameDataModule


def objective(trial: optuna.Trial, config=None):
    config['seq_predictor']['latent_size'] = trial.suggest_int('latent_size', 5, 75, 5)
    config['seq_predictor']['lr'] = trial.suggest_categorical('lr', [1., .01, .0001, .000001, .00000001, .0000000001])
    config['seq_predictor']['weight_decay'] = trial.suggest_float('weight_decay', 1e-9, .8, log=True)
    config['seq_predictor']['scheduler_gamma'] = trial.suggest_float('scheduler_gamma', .1, .9999)
    config['seq_predictor']['betas'] = [trial.suggest_float('beta0', .1, .9999), trial.suggest_float('beta1', .1, .9999)]


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

        datapath = config['dataloader']['datapath']
        dp = f'{datapath}/p{season}'  # Note the p in the path for possible matches
        if Path(dp).exists():
            files = glob(f'{dp}/*.pt')
            if len(files) > 0:
                ch_d = [torch.load(g) for g in files]
                t_data = torch.cat([c[0].unsqueeze(0) for c in ch_d], dim=0)
                o_data = torch.cat([c[1].unsqueeze(0) for c in ch_d], dim=0)
                tav_data = torch.cat([c[2].unsqueeze(0) for c in ch_d], dim=0)
                oav_data = torch.cat([c[3].unsqueeze(0) for c in ch_d], dim=0)
                predictions = 1. - model(t_data, o_data, tav_data, oav_data).detach().numpy()

                file_data = [Path(c).stem for c in files]
                gid = [int(c.split('_')[0]) for c in file_data]
                tid = [int(c.split('_')[1]) for c in file_data]
                oid = [int(c.split('_')[2]) for c in file_data]
                seas = [season for _ in file_data]
                small_res = pd.DataFrame(data=np.stack([gid, seas, tid, oid, predictions]).T,
                                         columns=['gid', 'season', 'tid', 'oid', 'Res'])
                small_res = small_res.set_index(['gid', 'season', 'tid', 'oid'])
                truth_br = generateBracket(season, True, datapath=datapath)
                test = generateBracket(season, True, datapath=datapath)
                res = 0
                for r in range(200):
                    try:
                        test = applyResultsToBracket(test, small_res, select_random=True, random_limit=.2)
                        res += scoreBracket(test, truth_br) / 100.
                    except KeyError:
                        continue
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
                                study_name='madness_3rd')
    objective = partial(objective, config=config)
    study.optimize(objective, 1000)

