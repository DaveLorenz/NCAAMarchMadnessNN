
# Load packages

import numpy as np
import pandas as pd

# Prep other sources

all_games_df = pd.read_csv('Data/~Created data/all_games_df.csv')
all_games_df = all_games_df[['is_tourney','Season','HTeamID','RTeamID','Hwin']]
all_games_df.head()

kp_df = pd.read_csv('Data/~Created data/kp_all.csv')
kp_df.head()

regseason_df = pd.read_csv('Data/~Created data/regseason_df.csv')
regseason_df  = regseason_df[['TeamID','Season','wins_top25','PointMargin','FG','FG3']]
regseason_df.head()

massey_df = pd.read_csv('Data/Kaggle NCAA/MasseyOrdinals_thru_2019_day_128.csv')
POM_df = massey_df[massey_df['SystemName'].str.contains("POM")]
POM_end_df = POM_df.loc[POM_df['RankingDayNum'] == 128]
POM_end_df.rename(columns={'OrdinalRank': 'RankPOM'}, inplace=True)
POM_end_df = POM_end_df[['Season','TeamID','RankPOM']]
POM_end_df.head()

# Create test set

#Test set (this sets the data up in the format Kaggle needs for scoring)
df_seeds = pd.read_csv('Data/Kaggle NCAA/NCAATourneySeeds.csv')
df_seeds = df_seeds[df_seeds['Season']==2019]
df_19_tourney = df_seeds.merge(df_seeds, how='inner', on='Season')
df_19_tourney = df_19_tourney[df_19_tourney['TeamID_x'] < df_19_tourney['TeamID_y']]
df_19_tourney['ID'] = df_19_tourney['Season'].astype(str) + '_'               + df_19_tourney['TeamID_x'].astype(str) + '_'               + df_19_tourney['TeamID_y'].astype(str)
df_19_tourney['SeedInt_x'] = [int(x[1:3]) for x in df_19_tourney['Seed_x']]
df_19_tourney['SeedInt_y'] = [int(x[1:3]) for x in df_19_tourney['Seed_y']]

#Make home team lower seed (consistent with training data)
df_19_tourney.loc[df_19_tourney['SeedInt_x']<df_19_tourney['SeedInt_y'], 'HTeamID'] = df_19_tourney['TeamID_x']
df_19_tourney.loc[df_19_tourney['SeedInt_x']>df_19_tourney['SeedInt_y'], 'HTeamID'] = df_19_tourney['TeamID_y']
df_19_tourney.loc[df_19_tourney['SeedInt_x']<df_19_tourney['SeedInt_y'], 'RTeamID'] = df_19_tourney['TeamID_y']
df_19_tourney.loc[df_19_tourney['SeedInt_x']>df_19_tourney['SeedInt_y'], 'RTeamID'] = df_19_tourney['TeamID_x']
df_19_tourney.loc[df_19_tourney['SeedInt_x']==df_19_tourney['SeedInt_y'], 'HTeamID'] = df_19_tourney['TeamID_x']
df_19_tourney.loc[df_19_tourney['SeedInt_x']==df_19_tourney['SeedInt_y'], 'RTeamID'] = df_19_tourney['TeamID_y']

df_19_tourney.loc[df_19_tourney['SeedInt_x']<df_19_tourney['SeedInt_y'], 'HSeed'] = df_19_tourney['SeedInt_x']
df_19_tourney.loc[df_19_tourney['SeedInt_x']>df_19_tourney['SeedInt_y'], 'HSeed'] = df_19_tourney['SeedInt_y']
df_19_tourney.loc[df_19_tourney['SeedInt_x']<df_19_tourney['SeedInt_y'], 'RSeed'] = df_19_tourney['SeedInt_y']
df_19_tourney.loc[df_19_tourney['SeedInt_x']>df_19_tourney['SeedInt_y'], 'RSeed'] = df_19_tourney['SeedInt_x']
df_19_tourney.loc[df_19_tourney['SeedInt_x']==df_19_tourney['SeedInt_y'], 'HSeed'] = df_19_tourney['SeedInt_x']
df_19_tourney.loc[df_19_tourney['SeedInt_x']==df_19_tourney['SeedInt_y'], 'RSeed'] = df_19_tourney['SeedInt_y']

df_19_tourney['is_tourney'] = 1

df_19_tourney = df_19_tourney.drop(['Seed_x','Seed_y','TeamID_x','TeamID_y','SeedInt_x','SeedInt_y'], axis=1)
df_19_tourney.sort_index()

home_road = ['H','R']
for hr in home_road:
    df_19_tourney = pd.merge(df_19_tourney, regseason_df, left_on=['Season',hr+'TeamID'], right_on = ['Season','TeamID'], how='left')
    df_19_tourney.rename(columns={'wins_top25': hr+'wins_top25'}, inplace=True)
    df_19_tourney.rename(columns={'PointMargin': hr+'PointMargin'}, inplace=True)
    df_19_tourney.rename(columns={'FG': hr+'FG'}, inplace=True)
    df_19_tourney.rename(columns={'FG3': hr+'FG3'}, inplace=True)
    df_19_tourney = df_19_tourney.drop(['TeamID'], axis=1)

for hr in home_road:
    df_19_tourney = pd.merge(df_19_tourney, POM_end_df, left_on=['Season',hr+'TeamID'], right_on = ['Season','TeamID'], how='left')
    df_19_tourney.rename(columns={'RankPOM': hr+'RankPOM'}, inplace=True)
    df_19_tourney = df_19_tourney.drop(['TeamID'], axis=1)

efficiency_list = ['conf','adjem','adjo','adjd','luck','TeamID']
for hr in home_road:
    df_19_tourney = pd.merge(df_19_tourney, kp_df, left_on=[hr+'TeamID','Season'], right_on = ['TeamID','Season'], how='inner')
    df_19_tourney = df_19_tourney.drop(['TeamID'], axis=1)
    for metric in efficiency_list:
        df_19_tourney.rename(columns={metric: hr+metric}, inplace=True)
    if hr == 'H':
        df_19_tourney.rename(columns={'team': 'home'}, inplace=True)
    if hr == 'R':
        df_19_tourney.rename(columns={'team': 'road'}, inplace=True) 

df_19_tourney['Htourny20plus'] = 0
df_19_tourney['Rtourny20plus'] = 0

experienced_teams = ['kansas','north carolina','kentucky','duke','michigan st.','wisconsin','florida','villanova','gonzaga','louisville','arizona','xavier','connecticut','syracuse','butler','ohio st.','ucla','west virginia','texas','michigan','pittsburgh','memphis','oregon']
for team in experienced_teams:
    df_19_tourney.loc[df_19_tourney['home']==team, 'Htourny20plus'] = 1
    df_19_tourney.loc[df_19_tourney['road']==team, 'Rtourny20plus'] = 1

df_19_tourney['HBig4Conf'] = 0
df_19_tourney['RBig4Conf'] = 0
conferences = ['ACC','B10','B12','SEC']
for conf in conferences:
    df_19_tourney.loc[df_19_tourney['Hconf']==conf, 'HBig4Conf'] = 1
    df_19_tourney.loc[df_19_tourney['Rconf']==conf, 'RBig4Conf'] = 1

df_19_tourney = df_19_tourney.fillna(df_19_tourney.mean())

# Output to csv

df_19_tourney.to_csv('Data/~Created data/test_combos_df_19.csv', index=False)

