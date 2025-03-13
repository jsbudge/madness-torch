from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from glob import glob
from thefuzz import process
from load_data import load, loadTeamNames
from tqdm import tqdm
from constants import state_abbreviations
import re

def replace_substrings(text, dictionary):
    pattern = re.compile("|".join(re.escape(key) for key in dictionary))
    return pattern.sub(lambda match: dictionary[match.group(0)], text)


def findNameAndGame(season_start, tid, adf, row, spellings, year):
    poss_games = adf.loc(axis=0)[:, year, tid]
    poss_dates = np.array([(season_start[year] + timedelta(days=d)).strftime('%Y-%m-%d') for d in poss_games['daynum']])
    poss_games = poss_games.loc[poss_dates == row['date']]
    poss_games = poss_games.loc[np.logical_and(row['team_score'] == poss_games['t_score'],
                                               row['opp_score'] == poss_games['o_score'])]
    if poss_games.shape[0] > 1:
        res = process.extractOne(row['opponent'].lower(), tspells)
        if res[1] >= 90:
            oid = spellings.loc[res[0], 'TeamID']
            poss_games = adf.loc(axis=0)[:, year, tid, oid]
            return poss_games.index.get_level_values(0)[0], poss_games.index.get_level_values(3)[0]
    elif poss_games.shape[0] == 1:
        return poss_games.index.get_level_values(0)[0], poss_games.index.get_level_values(3)[0]
    else:
        return None, None


def findGame(poss_games, season_start, row, year):
    poss_dates = np.array([(season_start[year] + timedelta(days=d)).strftime('%Y-%m-%d') for d in poss_games['daynum']])
    poss_games = poss_games.loc[poss_dates == row['date']]
    if poss_games.shape[0] == 1:
        return poss_games.index.get_level_values(0)[0], poss_games.index.get_level_values(3)[0]
    elif poss_games.shape[0] == 0:
        poss_dates = np.array(
            [(season_start[year] + timedelta(days=d - 1)).strftime('%Y-%m-%d') for d in poss_games['daynum']])
        poss_games = poss_games.loc[poss_dates == row['date']]
        if poss_games.shape[0] == 0:
            poss_dates = np.array(
                [(season_start[year] + timedelta(days=d + 1)).strftime('%Y-%m-%d') for d in poss_games['daynum']])
            poss_games = poss_games.loc[poss_dates == row['date']]
    if poss_games.shape[0] > 1:
        poss_games = poss_games.loc[np.logical_and(row['team_score'] == poss_games['t_score'],
                                                   row['opp_score'] == poss_games['o_score'])]
        if poss_games.shape[0] > 1:
            print('This is wild.')
        else:
            return poss_games.index.get_level_values(0)[0], poss_games.index.get_level_values(3)[0]
    return None, None


if __name__ == '__main__':
    with open('./run_params.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)


    datapath = config['dataloader']['datapath']
    abbrev_to_us_state = dict(map(reversed, state_abbreviations.items()))
    abbrev_to_us_state = dict(zip([a.lower() for a in abbrev_to_us_state.keys()],
                                  [a.lower() for a in abbrev_to_us_state.values()]))
    card_directions = {'north': 'n', 'east': 'e', 'west': 'w', 'south': 's'}
    card_movements = {'northern': 'n', 'eastern': 'e', 'western': 'w', 'southern': 's'}
    names = loadTeamNames(datapath)
    spellings = pd.read_csv(f'{datapath}/MTeamSpellings.csv').set_index('TeamNameSpelling')
    seasonal_info = pd.read_csv(f'{datapath}/MSeasons.csv').set_index('Season')
    season_start = dict(zip(seasonal_info.index.values,
                            [datetime.strptime(s, '%m/%d/%Y') for s in seasonal_info['DayZero']]))
    adf = load('basic', datapath=datapath)
    for idx, row in spellings.iterrows():
        if ' ' in idx:
            spellings.loc[idx.replace(' ', '_')] = row['TeamID']
        spellings.loc[replace_substrings(idx, abbrev_to_us_state)] = row['TeamID']
        spellings.loc[replace_substrings(idx, card_directions)] = row['TeamID']
        spellings.loc[replace_substrings(idx, card_movements)] = row['TeamID']

    spellings.loc['cal', 'TeamID'] = 1143
    spellings.loc['cent_conn_st', 'TeamID'] = 1148
    spellings.loc['ecu', 'TeamID'] = 1187
    spellings.loc['fair_dickinson', 'TeamID'] = 1192
    spellings.loc['fau', 'TeamID'] = 1194
    spellings.loc['jmu', 'TeamID'] = 1241
    # spellings.loc['la_tech', 'TeamID'] = 1256
    spellings.loc['pv_a&m', 'TeamID'] = 1341
    spellings.loc['tenn_tech', 'TeamID'] = 1399
    spellings.loc['usf', 'TeamID'] = 1378
    spellings.loc['loyola_mary', 'TeamID'] = 1258
    spellings.loc['mid_tennessee', 'TeamID'] = 1292
    spellings.loc['miss_st', 'TeamID'] = 1280
    spellings.loc['st._thomas_-_minnesota', 'TeamID'] = 1472


    tspells = spellings.index.values
    year = 2023
    yr_str = f'{year - 1}-{year-2000}'

    # Load in the csv files for teams and try and associate names
    fdir = Path(f'{datapath}/{yr_str}/box_scores')
    test = [Path(g).stem for g in glob(f'{fdir}/*')]
    loc_ids = {'C._Carolina': 1157, 'Cent_Conn_St': 1148}
    unknowns = []
    for t in test:
        san_t = t.lower().replace('state', 'st')# .replace('cent', 'central')
        if 'cent' in san_t and 'central' not in san_t:
            san_t = san_t.replace('cent', 'central')
        res = process.extractOne(san_t, tspells)
        print(f'Result for {t} is {res}')
        if res[1] >= 90:
            loc_ids[t] = spellings.loc[res[0], 'TeamID']
        else:
            unknowns.append(t)

    # Associate games with teams
    fdir = Path(f'{datapath}/{yr_str}/schedules')
    datafiles = [Path(g) for g in glob(f'{fdir}/*.csv')]
    gid_ids = {}
    for f in tqdm(datafiles):
        team_name = str(f.stem).replace('_schedule', '')
        sched = pd.read_csv(f'{f}')
        # Get scheduled team's tid first
        if team_name not in loc_ids.keys():
            san_t = team_name.lower().replace('state', 'st')  # .replace('cent', 'central')
            if 'cent' in san_t and 'central' not in san_t:
                san_t = san_t.replace('cent', 'central')
            res = process.extractOne(san_t, tspells)
            print(f'Result for {team_name} is {res}')
            if res[1] >= 90:
                loc_ids[team_name] = spellings.loc[res[0], 'TeamID']
        tid = loc_ids[team_name]

        for i in range(sched.shape[0]):
            row = sched.iloc[i]
            if datetime.strptime(row['date'], '%Y-%m-%d') > (season_start[year] + timedelta(days=132)):
                continue
            if row['opponent'] in loc_ids.keys():
                oid = loc_ids[row['opponent']]
                try:
                    poss_games = adf.loc(axis=0)[:, year, tid, oid]
                except KeyError:
                    print(f'Error with {names[tid]} vs. {names[oid]}')
                    continue
                if poss_games.shape[0] > 1:
                    gid, oid = findGame(poss_games, season_start, row, year)
                elif poss_games.shape[0] == 1:
                    gid = poss_games.index.get_level_values(0)[0]
                else:
                    gid = None
            else:
                gid, oid = findNameAndGame(season_start, tid, adf, row, spellings, year)
            if oid is not None:
                loc_ids[row['opponent']] = oid
            if gid is not None:
                gid_ids[row['game_id']] = gid



