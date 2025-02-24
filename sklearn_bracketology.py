from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import TruncatedSVD, KernelPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from load_data import getMatches, getPossMatches, prepFrame

def runEstimator(feats, feat_name, df):
    t0, t1 = getMatches(data, feats)
    s0, s1 = getMatches(tdata, feats)
    Xt0, Xs0, Xt1, Xs1, yt, ys = train_test_split(t0, t1, results, test_size=.3)

    rfc = RandomForestClassifier()
    rfc.fit(Xt0 - Xt1, yt)
    df.loc[s0.index, f'RFC_{feat_name}'] = rfc.predict_proba(s0 - s1)[:, 0]
    return df

if __name__ == '__main__':
    with open('./run_params.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    datapath = config['dataloader']['datapath']
    data = pd.read_csv(Path(f'{datapath}\\MGameDataBasic.csv')).set_index(['gid', 'season', 'tid', 'oid'])
    data = data.loc(axis=0)[:, 2004:]
    results = data['t_score'] - data['o_score'] > 0
    tdata = pd.read_csv(Path(f'{datapath}\\MNCAATourneyCompactResults.csv'))
    tdata = prepFrame(tdata, False).loc(axis=0)[:, 2004:]
    method_results = pd.DataFrame(index=tdata.index, columns=['Truth'], data=tdata['t_score'] - tdata['o_score'] > 0).astype(np.float32)

    for method in ['Simple', 'Gauss', 'Elo', 'Rank']:
        for prenorm in [True, False]:
            fnme = f'MNormalized{method}Averages' if prenorm else f'M{method}Averages'
            print(f'Running {fnme}...')
            feats = pd.read_csv(Path(f'{datapath}\\{fnme}.csv')).set_index(['season', 'tid'])

            method_results = runEstimator(feats, fnme[1:], method_results)
            method_results.to_csv(Path(f'{datapath}\\MNCAASklearnResults.csv'))

            # Truncated SVD
            tsvd = TruncatedSVD(n_components=60)
            trans_feats = pd.DataFrame(data=tsvd.fit_transform(feats), index=feats.index)

            method_results = runEstimator(trans_feats, f'TSVD_{fnme[1:]}', method_results)
            method_results.to_csv(Path(f'{datapath}\\MNCAASklearnResults.csv'))

            fnme = f'MNormalized{method}EncodedData' if prenorm else f'M{method}EncodedData'
            print(f'Running {fnme}...')
            feats = pd.read_csv(Path(f'{datapath}\\{fnme}.csv')).set_index(['season', 'tid'])
            method_results = runEstimator(feats, fnme[1:], method_results)
            method_results.to_csv(Path(f'{datapath}\\MNCAASklearnResults.csv'))

    method_results.to_csv(Path(f'{datapath}\\MNCAASklearnResults.csv'))

