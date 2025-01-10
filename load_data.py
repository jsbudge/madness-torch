import numpy as np
import pandas as pd
from pathlib import Path
from pandas import DataFrame
from tqdm import tqdm
from sklearn.linear_model import Ridge


def prepFrame(df: DataFrame) -> DataFrame:
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
    tdf = ldf.rename(columns=dict([(c, f't_{c[1:].lower()}') for c in ldf.columns]))
    odf = wdf.rename(columns=dict([(c, f'o_{c[1:].lower()}') for c in wdf.columns]))
    fdf = pd.concat([fdf, tdf.merge(odf, left_index=True, right_index=True).merge(iddf, left_index=True, right_index=True)])

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
    out_df['t_blk%'] = df['t_blk'] / (df['o_fga'] - df['o_fta'] / 1.4)
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
    out_df['o_blk%'] = df['o_blk'] / (df['t_fga'] - df['t_fta'] / 1.4)
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


if __name__ == '__main__':
    m_season_data_fnme = Path("C:\\Users\\Jeff\\PycharmProjects\\mmadness\\data\\MRegularSeasonDetailedResults.csv")
    msdf = pd.read_csv(m_season_data_fnme)

    sdf = prepFrame(msdf)
    adf = addAdvStatstoFrame(sdf)
    avdf = adf.groupby(['season', 'tid']).mean()
    # countdf = adf.groupby(['season', 'tid']).count()
    stddf = adf.groupby(['season', 'tid']).std()

    o_cols = np.sort([c for c in adf.columns if c[:2] == 'o_'])
    t_cols = np.sort([c for c in adf.columns if c[:2] == 't_'])

    infdf = pd.DataFrame(index=sdf.index)

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

    # Add new stats based on specific areas of the game
    # PASSING
    # stats that affect passing - ast, ast%, a/to, to, to%, econ
    # We'll connect them here to normalized resiliency
    norm_ast_res = (avdf[['t_ast%_res']] - avdf[['t_ast%_res']].groupby(['season']).mean()) / avdf[['t_ast%_res']].groupby(['season']).std()
    ridge = Ridge()
    ridge.fit(avdf[['t_ast%', 't_a/to', 't_to%', 't_econ']], norm_ast_res)
    passer_rating = ridge.predict(adf[['t_ast%', 't_a/to', 't_to%', 't_econ']])
    avdf['t_passrtg'] = pd.DataFrame(index=adf.index, columns=['t_passrtg'],
                                     data=ridge.predict(adf[['t_ast%', 't_a/to', 't_to%', 't_econ']])).groupby(
        ['season', 'tid']).mean()
    avdf['o_passrtg'] = pd.DataFrame(index=adf.index, columns=['o_passrtg'],
                                     data=ridge.predict(adf[['o_ast%', 'o_a/to', 'o_to%', 'o_econ']].values)).groupby(
        ['season', 'tid']).mean()

    # RIM PROTECTION
    # stats that affect this - blk%, 3/two%_inf, fg2%_inf



