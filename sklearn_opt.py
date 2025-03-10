from pathlib import Path
from sklearn.model_selection import ParameterGrid, GridSearchCV
import numpy as np
import pandas as pd
import yaml
from wrappers import SeasonalSplit
from load_data import prepFrame, getMatches
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':
    with open('./run_params.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    fnme = 'SimpleAverages'

    datapath = config['dataloader']['datapath']
    tdata = pd.read_csv(Path(f'{datapath}/MNCAATourneyCompactResults.csv'))
    tdata = pd.concat((tdata, pd.read_csv(Path(f'{datapath}/WNCAATourneyCompactResults.csv'))), ignore_index=True)
    tdata = prepFrame(tdata).loc(axis=0)[:, 2010:]
    feats = pd.read_csv(Path(f'{datapath}/{fnme}.csv')).set_index(['season', 'tid'])
    tmatch = getMatches(tdata, feats, diff=True)
    labels = (tdata['t_score'] - tdata['o_score']) > 0
    cv = SeasonalSplit()
    rfc = RandomForestClassifier()

    params = {'n_estimators': [10, 50, 100, 250, 1000], 'criterion': ['gini', 'log_loss'],
              'max_features': [15, 'sqrt', 'log2', None]}
    # pgrid = ParameterGrid(params)

    gscv = GridSearchCV(rfc, param_grid=params, cv=cv.split(tmatch, labels))
    gscv.fit(tmatch, labels)

    '''for xs, xt in cv.split(tdata, labels):
        rfc.fit(tmatch.loc[xt], labels[xt])
        print(f'Score is {rfc.score(tmatch.loc[xs], labels[xs])}')'''

