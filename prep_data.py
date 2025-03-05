import os
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm
from load_data import addSeasonalStatsToFrame, normalize, prepFrame
import torch


def gauss_weight(df, col, sigma=None):
    sigma = df[f't_{col}'].std() if sigma is None else sigma
    df_weight = np.exp(-(df[f't_{col}'] - df[f'o_{col}']) ** 2 / (2 * sigma ** 2))
    return df.mul(df_weight, axis=0).groupby(['season', 'tid']).sum().mul(
        1 / df_weight.groupby(['season', 'tid']).sum(), axis=0)


def col_weight(df, col):
    df_weight = df[col]
    return df.mul(df_weight, axis=0).groupby(['season', 'tid']).sum().mul(
        1 / df_weight.groupby(['season', 'tid']).sum(), axis=0)


def date_weight(df, dates):
    df_weight = dates['daynum']
    return df.mul(df_weight, axis=0).groupby(['season', 'tid']).sum().mul(
        1 / df_weight.groupby(['season', 'tid']).sum(), axis=0)

if __name__ == '__main__':

    with open('./run_params.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    datapath = config['dataloader']['datapath']
    print('Loading dataframes...')
    adf = pd.read_csv(Path(f'{datapath}\\GameDataAdv.csv')).set_index(['gid', 'season', 'tid', 'oid'])
    bdf = pd.read_csv(Path(f'{datapath}\\GameDataBasic.csv')).set_index(['gid', 'season', 'tid', 'oid'])[['gloc', 'daynum', 't_score', 'o_score']]
    av_other_df = pd.read_csv(Path(f'{datapath}\\Averages.csv')).set_index(['season', 'tid'])
    avodf = av_other_df[[c for c in av_other_df.columns if c not in adf.columns]]
    adf = adf.loc(axis=0)[:, 2004:].drop(columns=['numot']).fillna(0)
    bdf = bdf.loc(axis=0)[:, 2004:]
    print('dataframes loaded.')

    # Averaging using various methods over whole dataset, not averages
    # Save these as different datasets for training
    for method in ['Simple', 'Gauss', 'Elo', 'Recent']:
        if method == 'Simple':
            avdf = adf.groupby(['season', 'tid']).mean()
        elif method == 'Gauss':
            avdf = gauss_weight(adf, 'elo')
        elif method == 'Elo':
            avdf = col_weight(adf, 'o_elo')
        elif method == 'Recent':
            avdf = date_weight(adf, bdf)
        avdf_norm = normalize(avdf, to_season=True)
        avdf_norm = avdf_norm.merge(avodf, left_index=True, right_index=True)
        avdf_norm.to_csv(Path(f'{config["load_data"]["save_path"]}/{method}Averages.csv'))
        print(f'Saved {method}')

    # Averaging using various methods over normalized data
    # Is this significant? Dunno.
    nadf = normalize(adf, to_season=True)
    for method in ['Simple', 'Gauss', 'Elo', 'Recent']:
        if method == 'Simple':
            avdf = nadf.groupby(['season', 'tid']).mean()
        elif method == 'Gauss':
            avdf = gauss_weight(nadf, 'elo')
        elif method == 'Elo':
            avdf = col_weight(nadf, 'o_elo')
        elif method == 'Recent':
            avdf = date_weight(adf, bdf)
        avdf_norm = normalize(avdf, to_season=True)
        avdf_norm = avdf_norm.merge(avodf, left_index=True, right_index=True)
        avdf_norm.to_csv(Path(f'{config["load_data"]["save_path"]}/Normalized{method}Averages.csv'))
        print(f'Saved {method}')

    # Run SVD to get noise out of the values
    adf = normalize(adf)
    check_tsvd = TruncatedSVD(n_components=adf.shape[1])
    check_tsvd.fit(adf)
    n_true = sum(check_tsvd.singular_values_ > 1e-3)
    tsvd = TruncatedSVD(n_components=n_true)
    adf = normalize(pd.DataFrame(index=adf.index, data=tsvd.fit_transform(adf)))

    # Build dataset for training averaging method using last 5 games
    onehot = OneHotEncoder(sparse_output=False)
    han = onehot.fit_transform(bdf[['gloc']])
    adf['daynum'] = bdf['daynum']
    adf[['home', 'away', 'neutral']] = han
    adf = adf.reset_index().set_index(['gid', 'season', 'tid', 'oid', 'daynum']).sort_index()
    bdf = bdf.reset_index().set_index(['gid', 'season', 'tid', 'oid', 'daynum']).sort_index()
    # target = adf['t_score'] - adf['o_score'] > 0
    for season in range(adf.index.get_level_values(1).min(), adf.index.get_level_values(1).max() + 1):
        season_path = Path(f'{config["load_data"]["save_path"]}/{season}')
        if not season_path.exists():
            os.mkdir(season_path)
        sadf = adf.loc(axis=0)[:, season]
        for idx, row in tqdm(sadf.iterrows()):
            res = bdf.loc[idx]
            t_games = sadf.loc(axis=0)[:, :, idx[2], :, :idx[4]]
            o_games = sadf.loc(axis=0)[:, :, idx[3], :, :idx[4]]
            if t_games.shape[0] < 6 or o_games.shape[0] < 6:
                continue
            t_hist = torch.tensor(t_games.iloc[-6:-1].values, dtype=torch.float32)
            o_hist = torch.tensor(o_games.iloc[-6:-1].values, dtype=torch.float32)
            target_hist = torch.tensor(row[['home', 'away', 'neutral']].values, dtype=torch.float32)
            if torch.any(torch.isnan(t_hist)) or torch.any(torch.isnan(o_hist)):
                continue
            torch.save([t_hist, o_hist, target_hist, np.float32(res['t_score'] > res['o_score'])], f'{season_path}/{idx[0]}_{idx[2]}_{idx[3]}.pt')

    # Apply same logic to tournament data
    tdf = pd.read_csv(Path(f'{datapath}\\MNCAATourneyCompactResults.csv'))
    tdf = prepFrame(pd.concat((tdf, pd.read_csv(Path(f'{datapath}\\WNCAATourneyCompactResults.csv'))), ignore_index=True))
    for season in range(adf.index.get_level_values(1).min(), adf.index.get_level_values(1).max() + 1):
        season_path = Path(f'{config["load_data"]["save_path"]}/t{season}')
        if not season_path.exists():
            os.mkdir(season_path)
        try:
            stdf = tdf.loc(axis=0)[:, season]
        except KeyError:
            print('Missing a season.')
            continue
        for idx, row in tqdm(stdf.iterrows()):
            t_games = adf.loc(axis=0)[:, season, idx[2]]
            o_games = adf.loc(axis=0)[:, season, idx[3]]
            if t_games.shape[0] < 6 or o_games.shape[0] < 6:
                continue
            t_hist = torch.tensor(t_games.iloc[-5:].values, dtype=torch.float32)
            o_hist = torch.tensor(o_games.iloc[-5:].values, dtype=torch.float32)
            target_hist = torch.tensor([0., 0., 1.], dtype=torch.float32)
            if torch.any(torch.isnan(t_hist)) or torch.any(torch.isnan(o_hist)):
                continue
            torch.save([t_hist, o_hist, target_hist, np.float32(row['t_score'] > row['o_score'])], f'{season_path}/{idx[0]}_{idx[2]}_{idx[3]}.pt')



