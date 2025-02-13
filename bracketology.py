from pathlib import Path
from tqdm import tqdm
import torch
import yaml
import pandas as pd
from bracket import Bracket, generateBracket, applyResultsToBracket, scoreBracket
from load_data import getPossMatches
import numpy as np

from model import Predictor

def runCalcs(runs, feats, datapath, model, model_type: str = 'torch'):
    for season in range(2003, 2024):
        if season == 2020:
            continue
        test = generateBracket(season, True, datapath=datapath)
        df0, df1 = getPossMatches(feats, season, datapath=datapath)
        if model_type is 'torch':
            df0_tensor = torch.tensor(df0.values, dtype=torch.float32, device=model.device)
            df1_tensor = torch.tensor(df1.values, dtype=torch.float32, device=model.device)
            mdl_res = model(df0_tensor, df1_tensor)
            results = pd.DataFrame(index=df0.index, columns=['Res'], data=mdl_res.cpu().data.numpy())
        else:
            mdl_res = model(df0, df1)
            results = pd.DataFrame(index=df0.index, columns=['Res'], data=mdl_res)
        truth_br = generateBracket(season, True, datapath=datapath)
        res = list()
        for _ in tqdm(range(runs)):
            test = applyResultsToBracket(test, results, select_random=True, random_limit=1.)
            res.append(scoreBracket(test, truth_br))
        mu = np.mean(res)
        std = np.std(res)
        print(f'Mu is {mu} and std is {std} for {season}')

if __name__ == '__main__':
    with open('./run_params.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    # Grab predictor model
    model = Predictor.load_from_checkpoint(Path(f"{config['model']['training']['weights_path']}/{config['model']['name']}.ckpt"))
    datapath = config['dataloader']['datapath']

    feats = pd.read_csv(Path(f'{datapath}\\MEncodedData.csv')).set_index(['season', 'tid'])
    runCalcs(100, feats, datapath, model)




