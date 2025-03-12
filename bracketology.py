from glob import glob
from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging
from tqdm import tqdm
import torch
import yaml
import pandas as pd
from bracket import Bracket, generateBracket, applyResultsToBracket, scoreBracket
from dataloader import GameDataModule
from load_data import getPossMatches, getMatches
from sklearn.ensemble import RandomForestClassifier
import numpy as np

from model import Predictor, GameSequencePredictor
from wrappers import SKLearnWrapper


def runCalcs(runs, feats, datapath, model, model_type: str = 'torch'):
    res_df = pd.DataFrame(columns=np.arange(2003, 2024), index=np.arange(runs))
    for season in range(2003, 2024):
        if season == 2020:
            continue
        test = generateBracket(season, True, datapath=datapath)
        df0, df1 = getPossMatches(feats, season, datapath=datapath)
        if model_type == 'torch':
            df0_tensor = torch.tensor(df0.values, dtype=torch.float32, device=model.device)
            df1_tensor = torch.tensor(df1.values, dtype=torch.float32, device=model.device)
            mdl_res = model(df0_tensor, df1_tensor)
            results = pd.DataFrame(index=df0.index, columns=['Res'], data=mdl_res.cpu().data.numpy())
        else:
            mdl_res = model.forward(df0 - df1)
            results = pd.DataFrame(index=df0.index, columns=['Res'], data=mdl_res)
        truth_br = generateBracket(season, True, datapath=datapath)
        res = list()
        for r in tqdm(range(runs)):
            test = applyResultsToBracket(test, results, select_random=True, random_limit=1.)
            fscore = scoreBracket(test, truth_br)
            res_df.loc[r, season] = fscore
            res.append(scoreBracket(test, truth_br))
    return res_df


if __name__ == '__main__':
    with open('./run_params.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    results = pd.DataFrame()
    bracket_res = {}
    for season in range(2010, 2024):
        if season == 2020:
            continue
        config['dataloader']['season'] = season
        data = GameDataModule(**config['dataloader'])
        data.setup()

        # Get the model, experiment, logger set up
        config['seq_predictor']['init_size'] = data.train_dataset.data_len
        config['seq_predictor']['extra_param_size'] = data.train_dataset.extra_len
        mdl_name = f"{config['seq_predictor']['name']}"
        model = GameSequencePredictor(**config['seq_predictor'])
        expected_lr = max((config['seq_predictor']['lr'] * config['seq_predictor']['scheduler_gamma'] ** (
                    config['seq_predictor']['training']['max_epochs'] *
                    config['seq_predictor']['training']['swa_start'])), 1e-9)
        trainer = Trainer(logger=False, max_epochs=config['seq_predictor']['training']['max_epochs'],
                          default_root_dir=config['seq_predictor']['training']['weights_path'],
                          log_every_n_steps=config['seq_predictor']['training']['log_epoch'],
                          num_sanity_val_steps=0, detect_anomaly=False, callbacks=
                          [EarlyStopping(monitor='val_loss', patience=config['seq_predictor']['training']['patience'],
                                         check_finite=True),
                           StochasticWeightAveraging(swa_lrs=expected_lr,
                                                     swa_epoch_start=config['seq_predictor']['training']['swa_start'])])

        trainer.fit(model, datamodule=data)

        model.eval()

        datapath = config['dataloader']['datapath']
        dp = f'{datapath}/p{season}'  # Note the p in the path for possible matches
        if Path(dp).exists():
            files = glob(f'{dp}/*.pt')
            if len(files) > 0:
                bracket_res[season] = []
                ch_d = [torch.load(g) for g in files]
                t_data = torch.cat([c[0].unsqueeze(0) for c in ch_d], dim=0)
                o_data = torch.cat([c[1].unsqueeze(0) for c in ch_d], dim=0)
                tav_data = torch.cat([c[2].unsqueeze(0) for c in ch_d], dim=0)
                oav_data = torch.cat([c[3].unsqueeze(0) for c in ch_d], dim=0)
                targets = np.array([c[4] for c in ch_d])
                predictions = model(t_data, o_data, tav_data, oav_data).detach().numpy()

                file_data = [Path(c).stem for c in files]
                gid = [int(c.split('_')[0]) for c in file_data]
                tid = [int(c.split('_')[1]) for c in file_data]
                oid = [int(c.split('_')[2]) for c in file_data]
                seas = [season for _ in file_data]
                small_res = pd.DataFrame(data=np.stack([gid, seas, tid, oid, predictions]).T,
                                                  columns=['gid', 'season', 'tid', 'oid', 'Res'])
                small_res = small_res.set_index(['gid', 'season', 'tid', 'oid'])
                results = pd.concat((results, small_res))
                truth_br = generateBracket(season, True, datapath=datapath)
                test = generateBracket(season, True, datapath=datapath)
                for r in range(100):
                    test = applyResultsToBracket(test, small_res, select_random=True, random_limit=.1)
                    bracket_res[season].append(scoreBracket(test, truth_br))

