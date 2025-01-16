import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import yaml

if __name__ == '__main__':

    with open('./run_params.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    datapath = config['dataloader']['datapath']
    t0 = pd.read_csv(f'{datapath}/MTrainingData_0.csv').set_index(['gid', 'season', 'tid', 'oid'])
    t1 = pd.read_csv(f'{datapath}/MTrainingData_1.csv').set_index(['gid', 'season', 'tid', 'oid'])
    results = pd.read_csv(f'{datapath}/MTrainingData_label.csv').set_index(['gid', 'season', 'tid', 'oid'])

    Xs0, Xt0, Xs1, Xt1, ys, yt = train_test_split(t0, t1, results, test_size=.3)

    rfc = RandomForestClassifier()
    rfc.fit(Xs0 - Xs1, ys.values.ravel())

    test_res = rfc.predict(Xt0 - Xt1)
    res_acc = sum([x == y for x, y in zip(yt.values.flatten(), rfc.predict(Xt0 - Xt1))]) / Xt0.shape[0]