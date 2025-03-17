from glob import glob
from pathlib import Path
import pandas as pd
import torch
from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, StochasticWeightAveraging, ModelCheckpoint
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier, kernels

from bracket import generateBracket, applyResultsToBracket, scoreBracket
from dataloader import GameDataModule
from load_data import prepFrame, getMatches, load, getPossMatches
from model import GameSequencePredictor
import numpy as np
from tqdm import tqdm
import yaml

if __name__ == '__main__':
    torch.set_float32_matmul_precision('medium')
    gpu_num = 0
    device = f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu'
    seed_everything(np.random.randint(1, 2048), workers=True)

    with open('./run_params.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    datapath = config['dataloader']['datapath']

    tdf = pd.read_csv(Path(f'{datapath}/MNCAATourneyCompactResults.csv'))
    tdf = pd.concat((tdf, pd.read_csv(Path(f'{datapath}/WNCAATourneyCompactResults.csv'))),
                              ignore_index=True)
    sdf = pd.read_csv(Path(f'{datapath}/MSecondaryTourneyCompactResults.csv'))
    sdf = pd.concat((sdf, pd.read_csv(Path(f'{datapath}/WSecondaryTourneyCompactResults.csv'))),
                              ignore_index=True)
    avodf = load('SimpleAverages', datapath)
    tdf = prepFrame(pd.concat((tdf, sdf), ignore_index=True)).sort_index()
    tdf = tdf.loc(axis=0)[:, 2010:]

    t0 = getMatches(tdf, avodf, diff=True)
    labels = tdf['t_score'] - tdf['o_score'] > 0

    rfc = RandomForestClassifier(n_estimators=250, criterion='log_loss', max_features=None)
    # rfc = GaussianProcessClassifier(kernel=kernels.RBF(100.))
    rfc.fit(t0, labels)

    submission = pd.read_csv(f'{datapath}/SampleSubmissionStage2.csv')
    sub_df = pd.DataFrame(index=submission.index, columns=['season', 'tid', 'oid'])
    for i, idx in enumerate(submission['ID'].values):
        true_id = [int(n) for n in idx.split('_')]
        sub_df.loc[i, ['season', 'tid', 'oid']] = true_id
    sub_df['gid'] = np.arange(sub_df.shape[0])
    sub_df = sub_df.set_index(['gid', 'season', 'tid', 'oid'])
    g0 = getMatches(sub_df, avodf, diff=True)
    results = rfc.predict_proba(g0)[:, 1]
    submission['Pred'] = results

    submission.to_csv(f'{datapath}/submission_rfc_simple.csv', index=False)

    p0 = getPossMatches(avodf, 2025, diff=True, datapath=datapath)
    preds = rfc.predict_proba(p0)
    preds = pd.DataFrame(data=preds[:, 0], index=p0.index, columns=['Res'])
    br = generateBracket(2025, use_results=True, datapath=datapath)
    br = applyResultsToBracket(br, preds, select_random=True, random_limit=1.)

