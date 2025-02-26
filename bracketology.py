from pathlib import Path

from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import yaml
import pandas as pd
from urllib3.util import SKIPPABLE_HEADERS

from bracket import Bracket, generateBracket, applyResultsToBracket, scoreBracket
from load_data import getPossMatches, getMatches
from sklearn.ensemble import RandomForestClassifier
import numpy as np

from model import Predictor
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

    res_dict = {}
    for method in ['Simple', 'Gauss', 'Elo', 'Rank', 'Recent']:
        for prenorm in [True, False]:
            fnme = f'MNormalized{method}EncodedData' if prenorm else f'M{method}EncodedData'
            mdl_name = f"{config['model']['name']}_{fnme}"
            enc_df = pd.read_csv(f'{config["dataloader"]["datapath"]}/{fnme}.csv').set_index(['season', 'tid'])

            # Grab predictor model
            model = Predictor.load_from_checkpoint(
                Path(f"{config['model']['training']['weights_path']}/{mdl_name}.ckpt"),
            **config['model'], strict=False)
            model.eval()

            res_dict[fnme] = runCalcs(100, enc_df, config["dataloader"]["datapath"], model)






