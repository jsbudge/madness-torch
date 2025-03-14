from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import TruncatedSVD, KernelPCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier, kernels
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from tqdm import tqdm

from bracket import generateBracket, applyResultsToBracket, scoreBracket
from load_data import getMatches, getPossMatches, prepFrame
from wrappers import SKLearnWrapper


def runEstimator(feats, feat_name, df):
    s0, s1 = getMatches(df, feats)
    res = [[], [], []]
    for season in list(set(s0.index.get_level_values(1))):
        Xt0 = s0.loc[s0.index.get_level_values(1) != season]
        Xt1 = s1.loc[s0.index.get_level_values(1) != season]
        Xs0 = s0.loc[s0.index.get_level_values(1) == season]
        Xs1 = s1.loc[s0.index.get_level_values(1) == season]
        yt = df.loc[s0.index.get_level_values(1) != season, 'Truth']

        rfc = SKLearnWrapper(RandomForestClassifier(n_estimators=1000, criterion='log_loss'))
        rfc.fit(Xt0 - Xt1, yt)
        df.loc[Xs0.index, f'RFC_{feat_name}'] = rfc(Xs0 - Xs1)
        gpc = SKLearnWrapper(GaussianProcessClassifier(kernel=kernels.RBF(150.)))
        gpc.fit(Xt0 - Xt1, yt)
        df.loc[Xs0.index, f'GPC_{feat_name}'] = gpc(Xs0 - Xs1)
        ada = SKLearnWrapper(AdaBoostClassifier(n_estimators=1000))
        ada.fit(Xt0 - Xt1, yt)
        df.loc[Xs0.index, f'ADA_{feat_name}'] = ada(Xs0 - Xs1)
        truth_br = generateBracket(season, True, datapath=datapath)
        test = generateBracket(season, True, datapath=datapath)
        ps = getPossMatches(feats, season, diff=True, datapath=datapath)
        rfc_results = pd.DataFrame(index=ps.index, columns=['Res'], data=rfc(ps))
        gpc_results = pd.DataFrame(index=ps.index, columns=['Res'], data=gpc(ps))
        ada_results = pd.DataFrame(index=ps.index, columns=['Res'], data=ada(ps))
        for r in range(1000):
            test = applyResultsToBracket(test, rfc_results, select_random=True, random_limit=.1)
            res[0].append(scoreBracket(test, truth_br))
            test = applyResultsToBracket(test, gpc_results, select_random=True, random_limit=.1)
            res[1].append(scoreBracket(test, truth_br))
            test = applyResultsToBracket(test, ada_results, select_random=True, random_limit=.1)
            res[2].append(scoreBracket(test, truth_br))
    print(f'RFC_{feat_name} overall had a mean score of {np.mean(res[0])} and std of {np.std(res[0])}')
    print(f'GPC_{feat_name} overall had a mean score of {np.mean(res[1])} and std of {np.std(res[1])}')
    print(f'ADA_{feat_name} overall had a mean score of {np.mean(res[2])} and std of {np.std(res[2])}')
    return df

if __name__ == '__main__':
    with open('./run_params.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    datapath = config['dataloader']['datapath']
    tdata = pd.read_csv(Path(f'{datapath}/MNCAATourneyCompactResults.csv'))
    tdata = pd.concat((tdata, pd.read_csv(Path(f'{datapath}/WNCAATourneyCompactResults.csv'))), ignore_index=True)
    tdata = prepFrame(tdata).loc(axis=0)[:, 2010:]
    method_results = pd.DataFrame(index=tdata.index, columns=['Truth'], data=tdata['t_score'] - tdata['o_score'] > 0).astype(np.float32)

    for method in ['Simple', 'Gauss', 'Elo', 'Recent']:
        for prenorm in [True, False]:
            fnme = f'Normalized{method}Averages' if prenorm else f'{method}Averages'
            print(f'Running {fnme}...')
            feats = pd.read_csv(Path(f'{datapath}/{fnme}.csv')).set_index(['season', 'tid'])

            method_results = runEstimator(feats, fnme, method_results)
            method_results.to_csv(Path(f'{datapath}/NCAASklearnResults.csv'))

    method_results.to_csv(Path(f'{datapath}/NCAASklearnResults.csv'))

