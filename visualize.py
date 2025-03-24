from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from load_data import prepFrame, getMatches, load
from prep_data import loadFramesForTorch
import plotly.io as pio
import plotly.graph_objects as go
pio.renderers.default = 'browser'

if __name__ == '__main__':

    with open('./run_params.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    datapath = config['dataloader']['datapath']
    avodf = load('NormalizedGaussAverages', datapath)

    tdata = pd.read_csv(Path(f'{datapath}/MNCAATourneyCompactResults.csv'))
    tdata = pd.concat((tdata, pd.read_csv(Path(f'{datapath}/WNCAATourneyCompactResults.csv'))), ignore_index=True)
    tdata = prepFrame(tdata).loc(axis=0)[:, 2010:]

    s0 = getMatches(tdata, avodf, diff=True)
    method_results = pd.DataFrame(index=tdata.index, columns=['Truth'],
                                  data=tdata['t_score'] - tdata['o_score'] > 0).astype(np.float32)
    rfc = RandomForestClassifier(n_estimators=250, criterion='log_loss')

    rfc.fit(s0, method_results)
    rfe = RFE(rfc, verbose=2)
    rfe.fit(s0, method_results)

    important_features = rfe.feature_names_in_[rfe.ranking_ == 1]

    one = avodf.loc(axis=0)[2025, 1140][important_features]
    two = avodf.loc(axis=0)[2025, 1433][important_features]
    theta = np.linspace(0, 360, len(one))

    fig = px.line_polar(r=one.values, theta=theta, line_close=True)
    fig.update_traces(fill='toself')
    fig.update_layout(polar=dict(
        angularaxis = dict(tickmode='array', tickvals=theta, ticktext=important_features),
    ))
    fig.show()






