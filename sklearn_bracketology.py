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
from load_data import getMatches, getPossMatches, prepFrame, getSeeds
from wrappers import SKLearnWrapper


def runEstimator(feats, feat_name, df):
    s0 = getMatches(df, feats, diff=True)
    res = [[], []]
    for season in list(set(s0.index.get_level_values(1))):
        Xt0 = s0.loc[s0.index.get_level_values(1) != season]
        Xs0 = s0.loc[s0.index.get_level_values(1) == season]
        yt = df.loc[s0.index.get_level_values(1) != season, 'Truth']

        # Augment sampling with weird outliers
        sd = getSeeds(Xt0, datapath)
        sd_mask = np.logical_and(sd['t_seed'] > sd['o_seed'], yt).values.astype(bool)
        Xt0 = pd.concat([Xt0, Xt0.loc[sd_mask]])
        yt = pd.concat([yt, yt.loc[sd_mask]])

        rfc = SKLearnWrapper(RandomForestClassifier(n_estimators=250, criterion='log_loss', max_features=None))
        rfc.fit(Xt0, yt)
        df.loc[Xs0.index, f'RFC_{feat_name}'] = rfc(Xs0)
        gpc = SKLearnWrapper(GaussianProcessClassifier(kernel=kernels.RBF(100.)))
        gpc.fit(Xt0, yt)
        df.loc[Xs0.index, f'GPC_{feat_name}'] = gpc(Xs0)
        truth_br = generateBracket(season, True, datapath=datapath)
        test = generateBracket(season, True, datapath=datapath)
        ps = getPossMatches(feats, season, diff=True, datapath=datapath)
        rfc_results = pd.DataFrame(index=ps.index, columns=['Res'], data=rfc(ps))
        gpc_results = pd.DataFrame(index=ps.index, columns=['Res'], data=gpc(ps))
        for r in range(200):
            test = applyResultsToBracket(test, rfc_results, select_random=True, random_limit=.05)
            res[0].append(scoreBracket(test, truth_br))
            test = applyResultsToBracket(test, gpc_results, select_random=True, random_limit=.05)
            res[1].append(scoreBracket(test, truth_br))
    print(f'RFC_{feat_name} overall had a mean score of {np.mean(res[0])} and std of {np.std(res[0])}')
    print(f'GPC_{feat_name} overall had a mean score of {np.mean(res[1])} and std of {np.std(res[1])}')
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

