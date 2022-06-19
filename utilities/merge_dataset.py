import re

import pandas as pd


def drop_duplicated_cols(df):
    column_to_drop = [
        'FG_PCT', 'FG3_PCT', 'FT_PCT', 'REB',
        'AST', 'PTS', 'TEAM_ABBREVIATION', 'MIN'
    ]
    return df.drop(column_to_drop, axis=1)


def get_relevant_column():
    return [
        'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM',
        'FTA', 'OREB', 'DREB', 'STL', 'BLK', 'TO', 'PF',
        'PLUS_MINUS', 'FG2M', 'FG2A', 'EFG', 'FG_MISSED', 'FT_MISSED',
        'EFFICACY', 'POSS', 'PACE', 'PIE', 'OFF_EFC', 'DEF_EFC', 'PTS_FGA'
    ]


def get_dict_to_rename_columns(team):
    cols_stats_name = get_relevant_column()
    columns_to_rename = {}
    for col_name in cols_stats_name:
        columns_to_rename[col_name] = col_name + team
    return columns_to_rename


def rename_df_columns(columns_name, df):
    return df.rename(columns_name, axis=1)


def get_index_to_move_columns(columns, team):
    regex_cols_idx = [i for i, item in enumerate(columns) if re.search('[a-zA-Z0-9]+' + team, item)]
    return regex_cols_idx[len(regex_cols_idx) - 1] + 1


def move_dependent_col_as_last(df):
    tmp = df.pop('HOME_TEAM_WINS')
    df.insert(df.shape[1], 'WINNER_TEAM', tmp)


def merge_datasets(df1, df2, left_key='HOME_TEAM_ID', right_key='TEAM_ID', team='home'):
    tmp_df = pd.merge(df1, df2,
                      left_on=['GAME_ID', left_key],
                      right_on=['GAME_ID', right_key]
                      ).drop('TEAM_ID', axis=1)
    tmp_df = rename_df_columns(get_dict_to_rename_columns('_' + team), tmp_df)
    return tmp_df


def reorder_columns(df):
    cols_ordered = ['GAME_DATE_EST',
                    'GAME_ID',
                    'HOME_TEAM_ID',
                    'HOME_TEAM_NAME',
                    'VISITOR_TEAM_ID',
                    'VISITOR_TEAM_NAME',
                    'SEASON',
                    'PTS_home',
                    'FG_PCT_home',
                    'FT_PCT_home',
                    'FG3_PCT_home',
                    'AST_home',
                    'REB_home',
                    'ELO_BEFORE_home',
                    'FGM_home',
                    'FGA_home',
                    'FG3M_home',
                    'FG3A_home',
                    'FTM_home',
                    'FTA_home',
                    'OREB_home',
                    'DREB_home',
                    'STL_home',
                    'BLK_home',
                    'TO_home',
                    'PF_home',
                    'FG2M_home',
                    'FG2A_home',
                    'FG_MISSED_home',
                    'FT_MISSED_home',
                    'EFFICACY_home',
                    'EFG_home',
                    'PTS_FGA_home',
                    'PIE_home',
                    'POSS_home',
                    'PACE_home',
                    'OFF_EFC_home',
                    'DEF_EFC_home',
                    'PTS_away',
                    'FG_PCT_away',
                    'FT_PCT_away',
                    'FG3_PCT_away',
                    'AST_away',
                    'REB_away',
                    'ELO_BEFORE_away',
                    'FGM_away',
                    'FGA_away',
                    'FG3M_away',
                    'FG3A_away',
                    'FTM_away',
                    'FTA_away',
                    'OREB_away',
                    'DREB_away',
                    'STL_away',
                    'BLK_away',
                    'TO_away',
                    'PF_away',
                    'FG2M_away',
                    'FG2A_away',
                    'FG_MISSED_away',
                    'FT_MISSED_away',
                    'EFFICACY_away',
                    'EFG_away',
                    'PTS_FGA_away',
                    'PIE_away',
                    'POSS_away',
                    'PACE_away',
                    'OFF_EFC_away',
                    'DEF_EFC_away',
                    'HOME_TEAM_WINS']
    return df[cols_ordered]


if __name__ == "__main__":
    games = pd.read_csv('../datasets/curated_data/games.csv')
    games_details_avg = pd.read_csv('../datasets/curated_data/game_details.csv')
    games_details_avg = drop_duplicated_cols(games_details_avg)
    tmp_merged_df = merge_datasets(games, games_details_avg, team='home')
    merged_df = merge_datasets(tmp_merged_df, games_details_avg, left_key='VISITOR_TEAM_ID', team='away')
    merged_df = reorder_columns(merged_df)
    merged_df.to_csv('../datasets/curated_data/full_dataset_test.csv', index=False)
