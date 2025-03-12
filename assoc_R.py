from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from glob import glob
from thefuzz import process
from load_data import load

if __name__ == '__main__':
    with open('./run_params.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    datapath = config['dataloader']['datapath']
    spellings = pd.read_csv(f'{datapath}/MTeamSpellings.csv').set_index('TeamNameSpelling')
    adf = load('adv', datapath=datapath)
    for idx, row in spellings.iterrows():
        if ' ' in idx:
            spellings.loc[idx.replace(' ', '_')] = row['TeamID']

    tspells = spellings.index.values

    # Load in the csv files for teams and try and associate names
    fdir = Path(f'{datapath}/2023-24/box_scores')
    test = [Path(g).stem for g in glob(f'{fdir}/*')]
    loc_ids = {}
    for t in test:
        res = process.extractOne(t.lower(), tspells)
        print(f'Result for {t} is {res}')
        if res[1] >= 90:
            loc_ids[t] = spellings.loc[res[0], 'TeamID']

    # Associate games with teams
    fdir = Path(f'{datapath}/2023-24/schedules')
    datafiles = [Path(g) for g in glob(f'{fdir}/*.csv')]
    gid_ids = {}
    for f in datafiles:
        team_name = str(f.stem).replace('_schedule', '')
        sched = pd.read_csv(f'{f}')
        # Get scheduled team's tid first
        tid = loc_ids[team_name]
        for i in range(sched.shape[0]):
            row = sched.iloc[i]
            oid = loc_ids[row['opponent']]
            poss_games = adf.loc(axis=0)[:, 2024, tid, oid]
            fg = poss_games.loc[np.logical_and(row['team_score'] == poss_games['t_score'],
                                          row['opponent_score'] == poss_games['o_score'])]
            gid_ids[row['game_id']] = fg.index[0]



