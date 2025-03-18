from pathlib import Path
from sklearn.model_selection import ParameterGrid, GridSearchCV
import numpy as np
import pandas as pd
import yaml
from wrappers import SeasonalSplit
from load_data import prepFrame, getMatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern, RBF, RationalQuadratic
from sklearn.metrics import brier_score_loss

if __name__ == '__main__':
    with open('./run_params.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    fnme = 'GaussAverages'

    datapath = config['dataloader']['datapath']
    tdata = pd.read_csv(Path(f'{datapath}/MNCAATourneyCompactResults.csv'))
    tdata = pd.concat((tdata, pd.read_csv(Path(f'{datapath}/WNCAATourneyCompactResults.csv'))), ignore_index=True)
    tdata = prepFrame(tdata).loc(axis=0)[:, 2010:]
    feats = pd.read_csv(Path(f'{datapath}/{fnme}.csv')).set_index(['season', 'tid'])
    tmatch = getMatches(tdata, feats, diff=True)
    labels = (tdata['t_score'] - tdata['o_score']) > 0
    cv = SeasonalSplit()
    gpc = GaussianProcessClassifier()

    params = {'kernel': [Matern(nu=1.5), Matern(nu=2.5), Matern(), Matern(nu=25), RBF(1.0), RBF(100.), RBF(10.), RBF(1000.), RationalQuadratic()]}
    # pgrid = ParameterGrid(params)

    gscv = GridSearchCV(gpc, param_grid=params, cv=cv.split(tmatch, labels), verbose=2, scoring='neg_brier_score')
    gscv.fit(tmatch, labels)

    '''for xs, xt in cv.split(tdata, labels):
        rfc.fit(tmatch.loc[xt], labels[xt])
        print(f'Score is {rfc.score(tmatch.loc[xs], labels[xs])}')'''

