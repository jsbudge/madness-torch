from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from load_data import addSeasonalStatsToFrame, normalize


def gauss_weight(df, col, sigma=None):
    sigma = df[f't_{col}'].std() if sigma is None else sigma
    df_weight = np.exp(-(df[f't_{col}'] - df[f'o_{col}']) ** 2 / (2 * sigma ** 2))
    return df.mul(df_weight, axis=0).groupby(['season', 'tid']).sum().mul(
        1 / df_weight.groupby(['season', 'tid']).sum(), axis=0)


def col_weight(df, col):
    df_weight = df[col]
    return df.mul(df_weight, axis=0).groupby(['season', 'tid']).sum().mul(
        1 / df_weight.groupby(['season', 'tid']).sum(), axis=0)

if __name__ == '__main__':

    with open('./run_params.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    datapath = config['dataloader']['datapath']
    adf = pd.read_csv(Path(f'{datapath}\\MGameDataAdv.csv')).set_index(['gid', 'season', 'tid', 'oid'])
    adf = adf.loc(axis=0)[:, 2004:]

    # Averaging using various methods over whole dataset, not averages
    # Save these as different datasets for training
    for method in ['Simple', 'Gauss', 'Elo', 'Rank']:
        if method == 'Simple':
            avdf = adf.groupby(['season', 'tid']).mean()
        elif method == 'Gauss':
            avdf = gauss_weight(adf, 'elo')
        elif method == 'Elo':
            avdf = col_weight(adf, 'o_elo')
        elif method == 'Rank':
            avdf = gauss_weight(adf, 'rank')
        avdf = addSeasonalStatsToFrame(adf, avdf, True)
        avdf = avdf.drop(columns=['numot'])
        avdf_norm = normalize(avdf, to_season=True)
        avdf_norm.to_csv(Path(f'{config["load_data"]["save_path"]}/M{method}Averages.csv'))

    # Averaging using various methods over normalized data
    # Is this significant? Dunno.
    nadf = normalize(adf, to_season=True)
    for method in ['Simple', 'Gauss', 'Elo', 'Rank']:
        if method == 'Simple':
            avdf = nadf.groupby(['season', 'tid']).mean()
        elif method == 'Gauss':
            avdf = gauss_weight(nadf, 'elo')
        elif method == 'Elo':
            avdf = col_weight(nadf, 'o_elo')
        elif method == 'Rank':
            avdf = gauss_weight(nadf, 'rank')
        avdf = addSeasonalStatsToFrame(adf, avdf, True)
        avdf = avdf.drop(columns=['numot'])
        avdf_norm = normalize(avdf, to_season=True)
        avdf_norm.to_csv(Path(f'{config["load_data"]["save_path"]}/MNormalized{method}Averages.csv'))