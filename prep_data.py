from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import yaml

from load_data import getMatches

if __name__ == '__main__':

    with open('./run_params.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    datapath = config['dataloader']['datapath']
    data = pd.read_csv(Path(f'{datapath}\\MGameDataBasic.csv')).set_index(['gid', 'season', 'tid', 'oid'])
    feats = pd.read_csv(Path(f'{datapath}\\MEncodedData.csv')).set_index(['season', 'tid'])

    t0, t1 = getMatches(data, feats)
    results = data['t_score'] - data['o_score'] > 0

    Xs0, Xt0, Xs1, Xt1, ys, yt = train_test_split(t0, t1, results, test_size=.3)

    rfc = RandomForestClassifier()
    rfc.fit(Xs0 - Xs1, ys.values.ravel())

    test_res = rfc.predict(Xt0 - Xt1)
    res_acc = sum([x == y for x, y in zip(yt.values.flatten(), test_res)]) / Xt0.shape[0]