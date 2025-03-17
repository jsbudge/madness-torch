from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from glob import glob
from thefuzz import process
from load_data import load, loadTeamNames
from tqdm import tqdm
from constants import state_abbreviations, position_mapping, exp_mapping
import re

def replace_substrings(text, dictionary):
    pattern = re.compile("|".join(re.escape(key) for key in dictionary))
    return pattern.sub(lambda match: dictionary[match.group(0)], text)


def checkName(team_name, tspells, spellings):
    san_t = team_name.lower().replace('state', 'st')  # .replace('cent', 'central')
    if 'cent' in san_t and 'central' not in san_t:
        san_t = san_t.replace('cent', 'central')
    res = process.extractOne(san_t, tspells)
    print(f'Result for {team_name} is {res}')
    if res[1] >= 90:
        return spellings.loc[res[0], 'TeamID']
    return None


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


def createAdvancedStatsFrame(df):
    ndf = pd.DataFrame(index=df.index)
    ndf['efg%'] = (df['FGM'] + .5 * df['3PTM']) / (1e-12 + df['FGA'])
    ndf['fg3%'] = df['3PTM'] / (1e-12 + df['3PTA'])
    ndf['ast/to'] = df['AST'] - df['TO']
    ndf['pos_play'] = (df['FGM'] + df['AST'] + df['STL'] + df['BLK'] - df['TO'] - df['PF']) / (1e-12 + df['MIN'])
    ndf['stock'] = df['STL'] + df['BLK']
    ndf['econ'] = df['AST'] + df['STL'] - df['TO']
    ndf['height'] = df['height']
    ndf['weight'] = df['weight']
    ndf['exp'] = df['class']
    ndf['pos'] = df['position']
    ndf['bmi'] = 703. * df['weight'] / df['height']**2

    # These stats require the whole team's stats
    tdf = df.merge(df.groupby(['gid', 'tid']).sum(), on=['gid', 'tid'], suffixes=('', '_team'))
    tdf.index = df.index
    ndf['usage'] = (((df['FGA'] + .44 * df['FTA'] + df['TO']) * (tdf['MIN_team'] / 5)) /
                    (1e-12 + df['MIN'] * (tdf['FGA_team'] + .44 * tdf['FTA_team'] + tdf['TO_team'])))
    ndf['min%'] = df['MIN'] / 40.
    ndf['pts%'] = (df['PTS'] + df['AST'] * 2) / (1e-12 + tdf['PTS_team'])

    return ndf



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
    spellings.loc['queens_university', 'TeamID'] = 1474
    spellings.loc["saint_joe's", 'TeamID'] = 1386
    spellings.loc['ul_monroe', 'TeamID'] = 1419
    spellings.loc['uri', 'TeamID'] = 1348
    spellings.loc['ut_rio_grande', 'TeamID'] = 1410


    tspells = spellings.index.values
    loc_ids = {'C._Carolina': 1157, 'Cent_Conn_St': 1148}
    unknowns = []
    player_ids = {}
    pid_idx = 0
    for year in range(2010, 2024):
        yr_str = f'{year - 1}-{year-2000}'

        # Load in the csv files for teams and try and associate names
        fdir = Path(f'{datapath}/{yr_str}/box_scores')
        boxscore_teamnames = [Path(g).stem for g in glob(f'{fdir}/*')]
        for t in boxscore_teamnames:
            if t not in loc_ids.keys():
                poss_id = checkName(t, tspells, spellings)
                if poss_id is not None:
                    loc_ids[t] = poss_id
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
                poss_id = checkName(team_name, tspells, spellings)
                if poss_id is not None:
                    loc_ids[team_name] = poss_id
            tid = loc_ids[team_name]

            for i in range(sched.shape[0]):
                row = sched.iloc[i]
                if np.isnan(row['team_score']):
                    print(f'Found NaN score for {team_name} vs. {row["opponent"]}')
                    continue
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
                    loc_ids[row['opponent']] = int(oid)
                if gid is not None:
                    gid_ids[row['game_id']] = int(gid)

        # Merge everything together with rosters and box scores
        roster_df = pd.DataFrame()
        fdir = Path(f'{datapath}/{yr_str}/rosters')
        datafiles = [Path(g) for g in glob(f'{fdir}/*.csv')]
        for d in tqdm(datafiles):
            roster_teamname = d.stem.replace('_roster', '')
            if roster_teamname not in loc_ids.keys():
                poss_id = checkName(roster_teamname, tspells, spellings)
                if poss_id is not None:
                    loc_ids[roster_teamname] = poss_id
                else:
                    continue
                tid = poss_id
            else:
                tid = loc_ids[d.stem.replace('_roster', '')]
            dt = pd.read_csv(f'{d}')
            if dt.shape[0] == 0:
                continue
            if 'player_id' not in dt.columns:
                for idx, row in dt.iterrows():
                    player_ids[(row['name'], tid)] = pid_idx
                    dt.loc[idx, 'player_id'] = pid_idx
                    pid_idx += 1
            dt = dt[['player_id', 'position', 'height', 'weight', 'class']]
            dt['tid'] = tid
            dt['season'] = year
            roster_df = pd.concat((roster_df, dt))
        # Do some mapping of things to numbers
        roster_df['position'] = roster_df['position'].map(position_mapping)
        roster_df['class'] = roster_df['class'].map(exp_mapping)


        def height_map(x):
            try:
                if x == '--':
                    return 72
                ft, inch = x.strip().split("'")
                return int(ft) * 12 + int(inch[1:-1])
            except:
                return 72

        roster_df['height'] = roster_df['height'].fillna('6\' 3"')
        roster_df['height'] = roster_df['height'].apply(height_map)
        roster_df['weight'] = roster_df['weight'].fillna('190 lbs')
        roster_df['weight'] = roster_df['weight'].apply(lambda x: int(x[:-4]) if x != '--' else 190)

        fdir = Path(f'{datapath}/{yr_str}/box_scores')
        datafiles = [Path(g) for g in glob(f'{fdir}/*/*.csv')]
        boxscore_df = pd.DataFrame()
        for d in tqdm(datafiles):
            game_id = int(d.stem)
            if game_id in gid_ids.keys():
                dt = pd.read_csv(f'{d}').iloc[:-1]
                dt['gid'] = gid_ids[game_id]
                dt['tid'] = loc_ids[d.parts[-2]]
                dt['oid'] = loc_ids[dt.iloc[0]['opponent']]
                dt['season'] = year
                for dtidx in range(dt.shape[0]):
                    if (dt.loc[dtidx, 'player'], loc_ids[d.parts[-2]]) not in player_ids.keys():
                        player_ids[(dt.loc[dtidx, 'player'], loc_ids[d.parts[-2]])] = pid_idx
                        dt.loc[dtidx, 'player_id'] = pid_idx
                        pid_idx += 1
                    else:
                        dt.loc[dtidx, 'player_id'] = player_ids[(dt.loc[dtidx, 'player'], loc_ids[d.parts[-2]])]
                if 'team' in dt.columns:
                    dt = dt.drop(columns=['team'])
                if 'home' in dt.columns:
                    dt = dt.drop(columns=['home'])
                boxscore_df = pd.concat((boxscore_df,
                                         dt.drop(columns=['position', 'opponent', 'starter', 'date', 'location', 'player'])))

        boxscore_df = boxscore_df.merge(roster_df, on=['player_id', 'tid', 'season'])
        boxscore_df = createAdvancedStatsFrame(boxscore_df.set_index(['gid', 'season', 'tid', 'player_id']))

        if config['assoc_R']['save_files']:
            boxscore_df.to_csv(f'{datapath}/PlayerData{year}.csv')






