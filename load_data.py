from itertools import permutations

import numpy as np
import pandas as pd
from pathlib import Path
from pandas import DataFrame
from scipy.optimize import minimize, basinhopping
from tqdm import tqdm
from sklearn.linear_model import Ridge
import yaml


def normalize(df: DataFrame, transform = None, to_season: bool = False):
    """
    Normalize a frame to have a mean of zero and standard deviation of one.
    :param transform: SKlearn transformer.
    :param df: Frame of seasonal data.
    :param to_season: if True, calculates norms based on season instead of overall.
    :return: Frame of normalized seasonal data.
    """
    rdf = df.copy()
    if transform is not None:
        if to_season:
            for idx, grp in rdf.groupby(['season']):
                rdf.loc[grp.index] = transform.fit_transform(grp)
        else:
            rdf = pd.DataFrame(index=rdf.index, columns=rdf.columns, data=transform.fit_transform(rdf))
    else:
        if to_season:
            mu_df = df.groupby(['season']).mean()
            std_df = df.groupby(['season']).std()
            for idx, grp in rdf.groupby(['season']):
                rdf.loc[grp.index] = (grp - mu_df.loc[idx]) / std_df.loc[idx]
        else:
            rdf = (rdf - rdf.mean()) / rdf.std()
    return rdf


def getMatches(gids: DataFrame, team_feats: DataFrame, season: int = None, diff: bool = False, sort: bool = True):
    """
    Given a set of games, creates a frame with the chosen stats for predicting purposes.
    :param sort:
    :param gids: frame of games to get matches for. Only uses index. Should have index of [GameID, Season, TID, OID].
    :param team_feats: frame of features to use for matches. Should have index of [Season, TID].
    :param season: Season(s) for which data is wanted. If None, gets it all.
    :param diff: if True, returns differences of features. If False, returns two frames with features.
    :return: Returns either one frame or two, based on diff parameter, of game features.
    """
    if season is not None:
        g = gids.loc(axis=0)[:, season, :, :]
    else:
        g = gids.copy()
    ids = ['gid', 'season', 'tid', 'oid']
    gsc = g.reset_index()[ids]
    g1 = gsc.merge(team_feats, on=['season', 'tid']).set_index(ids)
    g2 = gsc.merge(team_feats, left_on=['season', 'oid'],
                   right_on=['season', 'tid']).set_index(ids)
    if diff:
        return (g1 - g2).sort_index() if sort else (g1 - g2)
    else:
        if sort:
            return g1.sort_index(), g2.sort_index()
        else:
            return g1, g2


def getPossMatches(team_feats, season, diff=False, use_seed=True, datapath=None):
    """
    Gets the possible matches in any season of all the teams in the tournament.
    :param use_seed: If True, only gets the tournament participants. Otherwise uses every team that's played that season
    :param team_feats: Frame of features wanted for hypothetical matchups. Should have index of [Season, TID]
    :param season: Season for which data is wanted.
    :param diff: if True, returns differences of features. If False, returns two frames with features.
    :return: Returns either one frame or two, based on diff parameter, of game features.
    """
    if use_seed:
        sd = pd.read_csv(f'{datapath}/MNCAATourneySeeds.csv')
    else:
        sd = pd.read_csv(f'{datapath}/MRegularSeasonCompactResults.csv')
    sd = sd.loc[sd['Season'] == season]['TeamID'].values
    teams = list(set(sd))
    matches = [[x, y] for (x, y) in permutations(teams, 2)]
    poss_games = pd.DataFrame(data=matches, columns=['tid', 'oid'])
    poss_games['season'] = season
    poss_games['gid'] = np.arange(poss_games.shape[0])
    gsc = poss_games.set_index(['gid', 'season'])
    g1 = gsc.merge(team_feats, left_on=['season', 'tid'],
                   right_on=['season', 'tid'], right_index=True).sort_index()
    g1 = g1.reset_index().set_index(['gid', 'season', 'tid', 'oid'])
    g2 = gsc.merge(team_feats, left_on=['season', 'oid'],
                   right_on=['season', 'tid'],
                   right_index=True).sort_index()
    g2 = g2.reset_index().set_index(['gid', 'season', 'tid', 'oid'])
    if diff:
        return g1 - g2
    else:
        return g1, g2

def prepFrame(df: DataFrame, full_frame: bool = True) -> DataFrame:
    df = df.rename(columns={'WLoc': 'gloc'})
    df['gloc'] = df['gloc'].map({'A': -1, 'N': 0, 'H': 1})

    # take the frame and convert it into team/opp format
    iddf = df[[c for c in df.columns if c[0] != 'W' and c[0] != 'L']]
    iddf = iddf.rename(columns=dict([(c, c.lower()) for c in iddf.columns]))
    iddf['gid'] = iddf.index.values
    wdf = df[[c for c in df.columns if c[0] == 'W']]
    ldf = df[[c for c in df.columns if c[0] == 'L']]

    # First the winners get to be the team
    tdf = wdf.rename(columns=dict([(c, f't_{c[1:].lower()}') for c in wdf.columns]))
    odf = ldf.rename(columns=dict([(c, f'o_{c[1:].lower()}') for c in ldf.columns]))
    fdf = tdf.merge(odf, left_index=True, right_index=True).merge(iddf, left_index=True, right_index=True)

    # then the losers get to do it
    if full_frame:
        tdf = ldf.rename(columns=dict([(c, f't_{c[1:].lower()}') for c in ldf.columns]))
        odf = wdf.rename(columns=dict([(c, f'o_{c[1:].lower()}') for c in wdf.columns]))
        full_ldf = tdf.merge(odf, left_index=True, right_index=True).merge(iddf, left_index=True, right_index=True)
        full_ldf['gloc'] = -full_ldf['gloc']
        fdf = pd.concat([fdf, full_ldf])

    # Final clean up of column names and ID setting
    fdf = fdf.rename(columns={'t_teamid': 'tid', 'o_teamid': 'oid'})
    fdf = fdf.set_index(['gid', 'season', 'tid', 'oid']).sort_index()

    return fdf


def addAdvStatstoFrame(df: DataFrame, add_to_frame: bool = False) -> DataFrame:
    """
    Given a properly indexed dataframe, adds some advanced stats and returns it
    :param df: DataFrame with index keys of [gid, season, tid, oid] and the necessary columns
    :param add_to_frame: if True, adds it to df. Otherwise returns a new DataFrame with the columns.
    :return: DataFrame with new stats. Same index.
    """
    out_df = pd.DataFrame(index=df.index)
    # First order derived stats
    out_df['t_fg%'] = df['t_fgm'] / df['t_fga']
    out_df['t_fg2%'] = (df['t_fgm'] - df['t_fgm3']) / (df['t_fga'] - df['t_fga3'])
    out_df['t_fg3%'] = df['t_fgm3'] / df['t_fga3']
    out_df['t_efg%'] = (df['t_fgm'] + .5 * df['t_fgm3']) / df['t_fga']
    out_df['t_ts%'] = df['t_score'] / (2 * (df['t_fga'] + .44 * df['t_fta']))
    out_df['t_econ'] = df['t_ast'] + df['t_stl'] - df['t_to']
    out_df['t_poss'] = .96 * (df['t_fga'] - df['t_or'] + df['t_to'] + .44 * df['t_fta'])
    out_df['t_offrat'] = df['t_score'] * 100 / out_df['t_poss']
    out_df['t_r%'] = (df['t_or'] + df['t_dr']) / (df['t_or'] + df['t_dr'] + df['o_or'] + df['o_dr'])
    out_df['t_ast%'] = df['t_ast'] / df['t_fgm']
    out_df['t_3two%'] = df['t_fga3'] / df['t_fga']
    out_df['t_ft/a'] = df['t_fta'] / (df['t_fga'] * 2 + df['t_fga3'])
    out_df['t_ft%'] = df['t_ftm'] / df['t_fta']
    out_df['t_to%'] = df['t_to'] / out_df['t_poss']
    out_df['t_extraposs'] = df['t_or'] + df['t_stl'] + df['o_pf']
    out_df['t_mov'] = df['t_score'] - df['o_score']
    out_df['t_rmar'] = (df['t_or'] + df['t_dr']) - (df['o_or'] + df['o_dr'])
    out_df['t_tomar'] = df['t_to'] - df['o_to']
    out_df['t_a/to'] = df['t_ast'] - df['t_to']
    out_df['t_blkperp'] = df['t_blk'] / out_df['t_poss']
    out_df['t_domf'] = (df['t_or'] - df['o_or']) * 1.2 + (df['t_dr'] - df['o_dr']) * 1.07 + \
                       (df['o_to'] - df['t_to']) * 1.5
    out_df['t_score%'] = (df['t_fgm'] + df['t_fgm3'] * .5 + df['t_ftm'] * .3 + df['t_pf'] * .5) / (
            df['t_fga'] + df['t_fta'] * .3 + df['t_to'])
    out_df['t_pie'] = df['t_score'] + df['t_fgm'] + df['t_ftm'] - df['t_fga'] - df['t_fta'] + \
                      df['t_dr'] + (.5 * df['t_or']) + df['t_ast'] + df['t_stl'] + .5 * df['t_blk'] - \
                      df['t_pf'] - df['t_to']
    out_df['o_pie'] = df['o_score'] + df['o_fgm'] + df['o_ftm'] - df['o_fga'] - df['o_fta'] + \
                      df['o_dr'] + (.5 * df['o_or']) + df['o_ast'] + df['o_stl'] + .5 * df['o_blk'] - \
                      df['o_pf'] - df['o_to']
    out_df['t_or%'] = df['t_or'] / (df['t_fga'] - df['t_fgm'])

    out_df['o_fg%'] = df['o_fgm'] / df['o_fga']
    out_df['o_fg2%'] = (df['o_fgm'] - df['o_fgm3']) / (df['o_fga'] - df['o_fga3'])
    out_df['o_fg3%'] = df['o_fgm3'] / df['o_fga3']
    out_df['o_efg%'] = (df['o_fgm'] + .5 * df['o_fgm3']) / df['o_fga']
    out_df['o_ts%'] = df['o_score'] / (2 * (df['o_fga'] + .44 * df['o_fta']))
    out_df['o_econ'] = df['o_ast'] + df['o_stl'] - df['o_to']
    out_df['o_poss'] = .96 * (df['o_fga'] - df['o_or'] + df['o_to'] + .44 * df['o_fta'])
    out_df['o_offrat'] = df['o_score'] * 100 / out_df['o_poss']
    out_df['o_r%'] = 1 - out_df['t_r%']
    out_df['o_ast%'] = df['o_ast'] / df['o_fgm']
    out_df['o_3two%'] = df['o_fga3'] / df['o_fga']
    out_df['o_ft/a'] = df['o_fta'] / (df['o_fga'] * 2 + df['o_fga3'])
    out_df['o_ft%'] = df['o_ftm'] / df['o_fta']
    out_df['o_to%'] = df['o_to'] / out_df['o_poss']
    out_df['o_extraposs'] = df['o_or'] + df['o_stl'] + df['t_pf']
    out_df['o_mov'] = df['o_score'] - df['t_score']
    out_df['o_rmar'] = (df['o_or'] + df['o_dr']) - (df['t_or'] + df['t_dr'])
    out_df['o_tomar'] = df['o_to'] - df['t_to']
    out_df['o_a/to'] = df['o_ast'] - df['o_to']
    out_df['o_blkperp'] = df['o_blk'] / out_df['o_poss']
    out_df['o_domf'] = (df['o_or'] - df['t_or']) * 1.2 + (df['o_dr'] - df['t_dr']) * 1.07 + \
                       (df['t_to'] - df['o_to']) * 1.5
    out_df['o_score%'] = (df['o_fgm'] + df['o_fgm3'] * .5 + df['o_ftm'] * .3 + df['o_pf'] * .5) / (
            df['o_fga'] + df['o_fta'] * .3 + df['o_to'])
    out_df['o_or%'] = df['o_or'] / (df['o_fga'] - df['o_fgm'])

    # Second order derived stats
    out_df['t_defrat'] = out_df['o_offrat']
    out_df['o_defrat'] = out_df['t_offrat']
    out_df['t_gamescore'] = 40 * out_df['t_efg%'] + 20 * out_df['t_r%'] + \
                            15 * out_df['t_ft/a'] + 25 - 25 * out_df['t_to%']
    out_df['o_gamescore'] = 40 * out_df['o_efg%'] + 20 * out_df['o_r%'] + \
                            15 * out_df['o_ft/a'] + 25 - 25 * out_df['o_to%']
    out_df['t_prodposs'] = out_df['t_poss'] - df['t_to'] - (df['t_fga'] - df['t_fgm'] + .44 * df['t_ftm'])
    out_df['o_prodposs'] = out_df['o_poss'] - df['o_to'] - (df['o_fga'] - df['o_fgm'] + .44 * df['o_ftm'])
    out_df['t_prodposs%'] = out_df['t_prodposs'] / out_df['t_poss']
    out_df['o_prodposs%'] = out_df['o_prodposs'] / out_df['o_poss']
    out_df['t_gamecontrol'] = out_df['t_poss'] / (out_df['o_poss'] + out_df['t_poss'])
    out_df['o_gamecontrol'] = 1 - out_df['t_gamecontrol']
    out_df['t_sos'] = df['t_score'] / out_df['t_poss'] - df['o_score'] / out_df['o_poss']
    out_df['o_sos'] = df['o_score'] / out_df['o_poss'] - df['t_score'] / out_df['t_poss']
    out_df['t_tie'] = out_df['t_pie'] / (out_df['t_pie'] + out_df['o_pie'])
    out_df['o_tie'] = out_df['o_pie'] / (out_df['t_pie'] + out_df['o_pie'])

    # third order derived stats
    eff_model = np.polyfit(out_df['t_efg%'], out_df['t_offrat'], 1)
    out_df['t_offeff'] = out_df['t_offrat'] - np.poly1d(eff_model)(out_df['t_efg%'])
    out_df['o_offeff'] = out_df['o_offrat'] - np.poly1d(eff_model)(out_df['o_efg%'])

    return df.merge(out_df, right_index=True, left_index=True) if add_to_frame else out_df


def addSeasonalStatsToFrame(sdf: DataFrame, df: DataFrame, add_to_frame: bool = True, pyth_exp: float = 13.91):
    """
    Adds some end-of-season stats to a dataframe.
    :param sdf: Frame with seasonal stats. Needs the stats listed in the function or it will error.
    :param df: Frame with team stats, id of [season, tid]
    :param add_to_frame: if True, adds the stats to df. Otherwise returns a new frame.
    :param pyth_exp: Float with the pythagorean win exponential. Generally accepted to be 13.91, but can be changed if desired.
    :return: Either df with the new columns or the new dataframe.
    """
    out_df = pd.DataFrame(index=df.index)
    dfapp = sdf.groupby(['season', 'tid'])
    out_df['t_closegame%'] = dfapp.apply(lambda x: sum(np.logical_or(abs(x['t_mov']) < 4, x['numot'] > 0)) / x.shape[0])
    out_df['t_win%'] = dfapp.apply(lambda x: sum(x['t_mov'] > 0) / x.shape[0])
    out_df['t_pythwin%'] = dfapp.apply(
        lambda grp: sum(grp['t_score'] ** pyth_exp) / sum(grp['t_score'] ** pyth_exp + grp['o_score'] ** pyth_exp))
    out_df['t_owin%'] = sdf.reset_index().merge(out_df['t_win%'].reset_index(),
                                            left_on=['season', 'oid'],
                                            right_on=['season', 'tid']).groupby(['season',
                                                                                 'tid_x']).mean()['t_win%'].values
    # Opponents' opponent win percentage calculations, for RPI
    oo_win = sdf.reset_index().merge(out_df['t_owin%'], left_on=['season', 'oid'], right_on=['season', 'tid']).groupby(
        ['season', 'tid']).mean()['t_owin%']
    out_df['t_rpi'] = .25 * out_df['t_win%'] + .5 * out_df['t_owin%'] + .25 * oo_win
    out_df['t_opythwin%'] = sdf.reset_index().merge(out_df['t_pythwin%'].reset_index(),
                                                left_on=['season', 'oid'],
                                                right_on=['season', 'tid']).groupby(['season',
                                                                                     'tid_x']).mean()[
        't_pythwin%'].values
    # Opponents' opponent win percentage calculations, for RPI
    oo_win = \
        sdf.reset_index().merge(out_df['t_opythwin%'], left_on=['season', 'oid'], right_on=['season', 'tid']).groupby(
            ['season', 'tid']).mean()['t_opythwin%']
    out_df['t_pythrpi'] = .25 * out_df['t_pythwin%'] + .5 * out_df['t_opythwin%'] + .25 * oo_win
    out_df['t_expwin%'] = dfapp.apply(lambda x: sum(x['t_elo'] > x['o_elo']) / x.shape[0])
    out_df['t_luck'] = out_df['t_win%'] - out_df['t_pythwin%']

    return df.merge(out_df, right_index=True, left_index=True) if add_to_frame else out_df

def loadTeamNames(datapath: str = './data'):
    """
    Create a dict of team names and ids so we know who's behind the numbers.
    :return: Dict of teamIDs and names.
    """
    df = pd.concat([pd.read_csv(f'{datapath}/MTeams.csv'), pd.read_csv(f'{datapath}/WTeams.csv')]).sort_index()
    ret = {}
    for idx, row in df.iterrows():
        ret[row['TeamID']] = row['TeamName']
        ret[row['TeamName']] = row['TeamID']
    return ret

if __name__ == '__main__':

    with open('./run_params.yaml', 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    m_season_data_fnme = Path(f"{config['load_data']['data_path']}/MRegularSeasonDetailedResults.csv")
    msdf = pd.read_csv(m_season_data_fnme)

    sdf = prepFrame(msdf)
    adf = addAdvStatstoFrame(sdf)
    avdf = adf.groupby(['season', 'tid']).mean()
    # countdf = adf.groupby(['season', 'tid']).count()
    stddf = adf.groupby(['season', 'tid']).std()

    o_cols = np.sort([c for c in adf.columns if c[:2] == 'o_'])
    t_cols = np.sort([c for c in adf.columns if c[:2] == 't_'])

    infdf = pd.DataFrame(index=sdf.index)

    print('Running influence calculations...')
    for tidx, tgrp in tqdm(adf.groupby(['tid'])):
        # Use median to reduce outlier influence in calculation
        o_av = adf.loc[adf.index.get_level_values(2) != tidx].groupby(['season', 'tid']).mean()
        # Join each game's o_fg% with that team's average fg%, excluding tidx (to remove bias)
        nchck = tgrp[o_cols].merge(o_av[t_cols], left_on=['season', 'oid'], right_on=['season', 'tid'])
        # Join with the std so we can calculate the # of stds each game was affected by the team
        nstd = tgrp[o_cols].merge(stddf[t_cols], left_on=['season', 'oid'], right_on=['season', 'tid'])
        # Get the number of standard deviations away from the mean this team made its opponent go
        inf_data = (nchck[o_cols].values - nchck[t_cols].values) / nstd[t_cols].values

        # These stats are how the team affects its opponents, not how the team is affected
        infdf.loc[tgrp.index, [f'{c}_inf' for c in nstd.columns if c[:2] == 't_']] = inf_data

    avdf[infdf.columns] = infdf.groupby(['season', 'tid']).mean().values

    # Get resiliency stats - how variable the team is compared to the rest of the world
    # Formulated so that a higher resiliency score means you have less variance than the average team
    avdf[[f'{c}_res' for c in stddf.columns]] = stddf - adf.groupby(['season', 'tid']).std().groupby(['season']).mean()

    print('Running skill stats...')
    avdf_norm = normalize(avdf, to_season=True)
    # Add new stats based on specific areas of the game
    # PASSING
    # stats that affect passing - ast, ast%, a/to, to, to%, econ
    # We'll connect them here to normalized resiliency
    ridge = Ridge()
    ridge.fit(avdf[['t_ast%', 't_a/to', 't_to%', 't_econ']], avdf_norm['t_ast%_res'])
    passer_rating = ridge.predict(adf[['t_ast%', 't_a/to', 't_to%', 't_econ']])
    avdf['t_passrtg'] = pd.DataFrame(index=adf.index, columns=['t_passrtg'],
                                     data=ridge.predict(adf[['t_ast%', 't_a/to', 't_to%', 't_econ']])).groupby(
        ['season', 'tid']).mean()
    avdf['o_passrtg'] = pd.DataFrame(index=adf.index, columns=['o_passrtg'],
                                     data=ridge.predict(adf[['o_ast%', 'o_a/to', 'o_to%', 'o_econ']].values)).groupby(
        ['season', 'tid']).mean()

    # RIM PROTECTION
    # stats that affect this - blk%, 3/two%_inf, fg2%_inf
    # I'm going to regress this against normalized opponent fg2%
    ridge = Ridge()
    ridge.fit(avdf[['t_blkperp', 'o_3two%', 'o_fg2%']], avdf_norm['t_fg2%_inf'])
    avdf['t_rimprot'] = pd.DataFrame(index=adf.index, columns=['t_rimprot'],
                                     data=ridge.predict(adf[['t_blkperp', 'o_3two%', 'o_fg2%']])).groupby(
        ['season', 'tid']).mean()
    avdf['o_rimprot'] = pd.DataFrame(index=adf.index, columns=['o_rimprot'],
                                     data=ridge.predict(adf[['o_blkperp', 't_3two%', 't_fg2%']].values)).groupby(
        ['season', 'tid']).mean()

    # PERIMETER DEFENSE
    # stats that affect this - 3/two%_inf, fg3%_inf, ast%_inf, to%_inf
    ridge = Ridge()
    ridge.fit(avdf[['o_ast%', 'o_3two%', 'o_fg3%', 'o_to%']], avdf_norm['t_fg3%_inf'])
    avdf['t_perimdef'] = pd.DataFrame(index=adf.index, columns=['t_perimdef'],
                                     data=ridge.predict(adf[['o_ast%', 'o_3two%', 'o_fg3%', 'o_to%']])).groupby(
        ['season', 'tid']).mean()
    avdf['o_perimdef'] = pd.DataFrame(index=adf.index, columns=['o_perimdef'],
                                     data=ridge.predict(adf[['t_ast%', 't_3two%', 't_fg3%', 't_to%']].values)).groupby(
        ['season', 'tid']).mean()

    # Run elo ratings
    print('Running elo ratings...')
    m_cond_data_fnme = Path(f"{config['load_data']['data_path']}/MRegularSeasonCompactResults.csv")
    mcdf = pd.read_csv(m_cond_data_fnme)
    mcdf = mcdf.loc[mcdf['Season'] > 2001]

    # Don't duplicate for the losers because we want to play each game once
    scdf = prepFrame(mcdf, False)
    tids = list(set(scdf.index.get_level_values(2)))

    # curr_elo = np.ones(max(tids) + 1) * 1500
    scdf = scdf.sort_values(by=['season', 'daynum'])
    scdf['mov'] = scdf['t_score'] - scdf['o_score']
    scdf['t_elo'] = 1500.
    scdf['o_elo'] = 1500.

    def runElo(x):
        scarray = scdf.reset_index().values
        curr_elo = np.ones(max(tids) + 1) * 1500
        curr_seas = 2002
        mu_reg = x[0]
        margin = x[1]
        k = x[2]
        for n in range(scarray.shape[0]):
            if curr_seas != scarray[n, 1]:
                # Regress everything to the mean
                for val in curr_elo:
                    val += ((1 - mu_reg) * val + mu_reg * 1500 - val)
            curr_seas = scarray[n, 1]
            t_elo = curr_elo[int(scarray[n, 2])]
            o_elo = curr_elo[int(scarray[n, 3])]
            scarray[n, -2:] = [t_elo, o_elo]
            hc_adv = x[3] * scarray[n, 7]
            elo_diff = max(t_elo + hc_adv - o_elo, -400)
            elo_shift = 1. / (10. ** (-elo_diff / 400.) + 1.)
            exp_margin = margin + 0.006 * elo_diff
            final_elo_update = k * ((abs(scarray[n, 9]) + 3.) ** 0.8) / exp_margin * (1 - elo_shift) * np.sign(scarray[n, 9])
            curr_elo[int(scarray[n, 2])] += final_elo_update
            curr_elo[int(scarray[n, 3])] -= final_elo_update
        return scarray

    def optElo(x):
        sc_x = runElo(x)
        return 1 - np.logical_and((sc_x[:, 10] - sc_x[:, 11] > 0), sc_x[:, 9] > 0).sum() / sc_x.shape[0]

    elo_params = np.array([0.3896076731384477, 6.51988202753904, 34.11927604457895, 0.17251109126016217])
    if config['load_data']['run_elo_opt']:
        print('Optimizing elo...')
        opt_res = basinhopping(optElo, elo_params,
                               minimizer_kwargs=dict(bounds=[(0.1, 10.), (0.1, 100.), (.1, 100), (-10., 10.)]))
        # Add them to the adv frame (make sure the game ids are the same, though)
        sc_out = runElo(opt_res['x'])
    else:
        print('Not optimizing elo.')
        sc_out = runElo(elo_params)
    scdf = pd.DataFrame(index=scdf.index, columns=scdf.columns, data=sc_out[:, 4:])
    joiner_df = sdf.reset_index()[['season', 'tid', 'oid', 'daynum', 'gid']].merge(
        scdf.reset_index()[['season', 'tid', 'oid', 'daynum', 't_elo', 'o_elo']],
        on=['season', 'tid', 'oid', 'daynum'])
    joiner_df = joiner_df.set_index(['gid', 'season', 'tid', 'oid'])

    adf.loc[joiner_df.index, ['t_elo', 'o_elo']] = joiner_df[['t_elo', 'o_elo']]
    adf.loc[np.isnan(adf['t_elo']), ['t_elo', 'o_elo']] = joiner_df[['o_elo', 't_elo']].values
    # Run Glicko ratings

    # Consolidate massey ordinals in a logical way
    ord_fnme = Path(f"{config['load_data']['data_path']}/MMasseyOrdinals.csv")
    ord_df = pd.read_csv(ord_fnme)
    ord_df = ord_df.pivot_table(index=['Season', 'TeamID', 'RankingDayNum'], columns=['SystemName'])
    ord_df.columns = ord_df.columns.droplevel(0)

    ord_id = sdf.reset_index()[['season', 'tid', 'oid', 'daynum', 'gid']].rename(
        columns={'season': 'Season', 'tid': 'TeamID', 'daynum': 'RankingDayNum'})
    ord_id = ord_id[ord_id['Season'] > 2002]
    adf[['t_rank', 'o_rank']] = 0.

    print('Running ranking consolidation...')
    for t in tqdm(tids):
        t_ords = ord_df.loc[:, t, :]
        ord_id_local = ord_id.loc[np.logical_or(ord_id['TeamID'] == t, ord_id['oid'] == t)].set_index(['Season', 'RankingDayNum'])
        check = ord_id_local.join(t_ords, how='left', lsuffix='_left', rsuffix='_right')
        check = check.loc[check['TeamID'] == t]
        check['t_rank'] = check.reset_index().ffill().drop(
            columns=['Season', 'RankingDayNum', 'TeamID', 'gid', 'oid']).mean(axis=1, skipna=True).values
        check = check.reset_index().rename(columns={'Season': 'season', 'TeamID': 'tid'}).set_index(['gid', 'season', 'tid', 'oid'])[['t_rank']]
        check = check.fillna(400.)
        adf.loc[check.index, 't_rank'] = check['t_rank']

    ind_0 = adf.reset_index().drop_duplicates(subset=['gid'], keep='first').set_index(['gid', 'season', 'tid', 'oid']).index
    ind_1 = adf.reset_index().drop_duplicates(subset=['gid'], keep='last').set_index(['gid', 'season', 'tid', 'oid']).index
    adf.loc[ind_0, 'o_rank'] = adf.loc[ind_1, 't_rank'].values
    adf.loc[ind_1, 'o_rank'] = adf.loc[ind_0, 't_rank'].values

    # Add in seasonal stats to the avdf frame
    print('Adding seasonal stats to frame...')
    adf[['t_score', 'o_score', 'numot']] = sdf[['t_score', 'o_score', 'numot']]
    avdf = addSeasonalStatsToFrame(adf, avdf, True)

    # Save out the files so we can use them later
    if config['load_data']['save_files']:
        adf.to_csv(f'{config["load_data"]["save_path"]}/MGameDataAdv.csv')
        sdf.to_csv(f'{config["load_data"]["save_path"]}/MGameDataBasic.csv')

    # Create a dataframe of the tournament results with average data
    ncaa_fnme = f'{config["load_data"]["data_path"]}/MNCAATourneyCompactResults.csv'
    ncaa_tdf = pd.read_csv(ncaa_fnme)

    ncaa_tdf = prepFrame(ncaa_tdf)

    # Add in secondary tourney results
    sec_fnme = f'{config["load_data"]["data_path"]}/MSecondaryTourneyCompactResults.csv'
    sc_tdf = pd.read_csv(sec_fnme)
    ncaa_tdf = pd.concat([ncaa_tdf, prepFrame(sc_tdf)])
    ncaa_tdf['t_win'] = ncaa_tdf['t_score'] - ncaa_tdf['o_score'] > 0

    # merge information with teams
    print('Generating tournament training data...')
    avdf_norm = normalize(avdf, to_season=True)
    tdf, odf = getMatches(ncaa_tdf, avdf_norm)
    results_df = ncaa_tdf.loc[tdf.index, ['t_win']]

    if config['load_data']['save_files']:
        avdf_norm.to_csv(f'{config["load_data"]["save_path"]}/MAverages.csv')
        tdf.to_csv(f'{config["load_data"]["save_path"]}/MTrainingData_0.csv')
        odf.to_csv(f'{config["load_data"]["save_path"]}/MTrainingData_1.csv')
        results_df.to_csv(f'{config["load_data"]["save_path"]}/MTrainingData_label.csv')















