
# Load packages

import numpy as np
import pandas as pd


# Combine all tournament and reg season games

tourney_compact_df = pd.read_csv('Data/Kaggle NCAA/NCAATourneyCompactResults.csv')
tourney_compact_df['is_tourney'] = 1.0

regseason_compact_df = pd.read_csv('Data/Kaggle NCAA/RegularSeasonCompactResults.csv')
regseason_compact_df['is_tourney'] = 0.0

all_games_df = regseason_compact_df.append(tourney_compact_df)
all_games_df.head()

all_games_df.tail()


# Add spread, seeds, ranks, and Ken POM data

# Load and merge on point spread

spread_df = pd.read_csv('Data/~Created data/spreads_all.csv')
spread_df = spread_df[['Season','date','line','lineavg','road','home','rscore','hscore','WScore','LScore','HTeamID','RTeamID','WTeamID','LTeamID']]

# Merge point spread on to games

all_games_df = pd.merge(all_games_df, spread_df, left_on=['WTeamID','LTeamID','Season','WScore','LScore'], right_on = ['WTeamID','LTeamID','Season','WScore','LScore'], how='inner')
all_games_df = all_games_df[['Season','date','is_tourney','home','HTeamID','hscore','road','RTeamID','rscore','line','lineavg']]

all_games_df['Hwin'] = 0
all_games_df.loc[all_games_df['hscore']>all_games_df['rscore'], 'Hwin'] = 1      

all_games_df.head()

# Add seeds

seeds_df = pd.read_csv('Data/Kaggle NCAA/NCAATourneySeeds.csv')
seeds_df['Seed_num'] = seeds_df['Seed'].str.extract('(\d\d)', expand=True)
seeds_df['Seed_num'] = pd.to_numeric(seeds_df['Seed_num'])
seeds_df = seeds_df[['Season','TeamID','Seed_num']]
seeds_df.rename(columns={'Seed_num': 'Seed'}, inplace=True)
seeds_df.head()

#Note: we merge twice for this merge and subsequent merges, because the data are in a wide format by team (i.e., there are two variables rather than two rows for each game)
home_road = ['H','R']
for hr in home_road:
    all_games_df = pd.merge(all_games_df, seeds_df, left_on=[hr+'TeamID','Season'], right_on = ['TeamID','Season'], how='left')
    all_games_df.rename(columns={'Seed': hr+'Seed'}, inplace=True)
    all_games_df = all_games_df.drop(['TeamID'], axis=1)

all_games_df.head()


# Add Ken Pom data

kp_df = pd.read_csv('Data/~Created data/kp_all.csv')

efficiency_list = ['conf','adjem','adjo','adjd','luck']
for hr in home_road:
    all_games_df = pd.merge(all_games_df, kp_df, left_on=[hr+'TeamID','Season'], right_on = ['TeamID','Season'], how='inner')
    for metric in efficiency_list:
        all_games_df.rename(columns={metric: hr+metric}, inplace=True)
    all_games_df = all_games_df.drop(['TeamID','team'], axis=1)

all_games_df.head()

# Add Massey and Ken Pom rankings

massey_df = pd.read_csv('Data/Kaggle NCAA/MasseyOrdinals_thru_2019_day_128.csv')
massey_df19 = massey_df.loc[massey_df['Season'] == 2019]

#Ranking at the end of season versus end of tourney (as shown in KP data)
POM_df_else = massey_df[massey_df['SystemName'].str.contains("POM")]
POM_end_df_else = POM_df_else.loc[POM_df_else['RankingDayNum'] == 133]
POM_df19 = massey_df19[massey_df19['SystemName'].str.contains("POM")]
POM_end_df_19 = POM_df19.loc[POM_df19['RankingDayNum'] == 128]
POM_end_df = POM_end_df_else.append(POM_end_df_19)
POM_end_df = POM_end_df[['Season','TeamID','OrdinalRank']]
POM_end_df.rename(columns={'OrdinalRank': 'RankPOM'}, inplace=True)

for hr in home_road:
    all_games_df = pd.merge(all_games_df,POM_end_df, left_on=[hr+'TeamID','Season'], right_on = ['TeamID','Season'], how='left')
    all_games_df.rename(columns={'RankPOM': hr+'RankPOM'}, inplace=True)
    all_games_df = all_games_df.drop(['TeamID'], axis=1)

# Calculate regular season avg stats

regseason_detail_df = pd.read_csv('Data/Kaggle NCAA/RegularSeasonDetailedResults.csv')
regseason_detail_df.head()

#Attach the POM data
teamslist = ['W','L']
for wl in teamslist:
    regseason_detail_df = pd.merge(regseason_detail_df, POM_end_df, left_on=[wl+'TeamID','Season'], right_on = ['TeamID','Season'], how='left')
    regseason_detail_df.rename(columns={'RankPOM': wl+'RankPOM'}, inplace=True)
    regseason_detail_df = regseason_detail_df.drop(['TeamID'], axis=1)

# Regular season (create one row for each team with avg stats and win info)

win_team_only = regseason_detail_df.drop(['LTeamID','WAst','LAst','LStl','WStl','LBlk','WBlk','LPF','WPF','NumOT','DayNum'], axis=1)
win_team_only.rename(columns={'WTeamName': 'TeamName'}, inplace=True)
win_team_only.rename(columns={'WTeamID': 'TeamID'}, inplace=True)

team_factors_list = ['Score','FGM','FGA','FGA3','FGM3','TO','FTM','FTA','RankPOM']
for factor in team_factors_list:
    win_team_only.rename(columns={'W'+factor: factor}, inplace=True)
    win_team_only.rename(columns={'L'+factor: 'Opponent'+factor}, inplace=True)

win_team_only['wins'] = 1 

win_team_only['wins_top25'] = 0
win_team_only.loc[win_team_only['OpponentRankPOM']<26, 'wins_top25'] = 1

win_team_only['wins_top5'] = 0
win_team_only.loc[win_team_only['OpponentRankPOM']<6, 'wins_top5'] = 1

win_team_only = win_team_only[['Season','TeamID','wins','wins_top25','wins_top5','Score','OpponentScore','FGM','FGA','OpponentFGM','OpponentFGA','FGM3','FGA3','OpponentFGM3','OpponentFGA3','FTM','FTA','OpponentFTM','OpponentFTA','TO','OpponentTO']]

loss_team_only = regseason_detail_df.drop(['WTeamID','WAst','LAst','LStl','WStl','LBlk','WBlk','LPF','WPF','NumOT','DayNum'], axis=1)
loss_team_only.rename(columns={'LTeamName': 'TeamName'}, inplace=True)
loss_team_only.rename(columns={'LTeamID': 'TeamID'}, inplace=True)

for factor in team_factors_list:
    loss_team_only.rename(columns={'L'+factor: factor}, inplace=True)
    loss_team_only.rename(columns={'W'+factor: 'Opponent'+factor}, inplace=True)

loss_team_only['wins'] = 0
loss_team_only['wins_top25'] = 0
loss_team_only['wins_top5'] = 0

loss_team_only = loss_team_only[['Season','TeamID','wins','wins_top25','wins_top5','Score','OpponentScore','FGM','FGA','OpponentFGM','OpponentFGA','FGM3','FGA3','OpponentFGM3','OpponentFGA3','FTM','FTA','OpponentFTM','OpponentFTA','TO','OpponentTO']]

reg_season_all = win_team_only.append(loss_team_only)

reg_season_all['TOmargin'] = reg_season_all['TO']-reg_season_all['OpponentTO']

reg_season_all['PointMargin'] = reg_season_all['Score']-reg_season_all['OpponentScore']

reg_season_all['FG'] = reg_season_all['FGM']/reg_season_all['FGA']
reg_season_all['FGopponent'] = reg_season_all['OpponentFGM']/reg_season_all['OpponentFGA']

reg_season_all['FG3'] = reg_season_all['FGM3']/reg_season_all['FGA3']
reg_season_all['FG3opponent'] = reg_season_all['OpponentFGM3']/reg_season_all['OpponentFGA3']

reg_season_all['FT'] = reg_season_all['FTM']/reg_season_all['FTA']

reg_season_all = reg_season_all.drop(['TO','OpponentTO','FGM','FGA','OpponentFGA','OpponentFGM','FGM3','FGA3','OpponentFGM3','OpponentFGA3','FTM','FTA','OpponentFTM','OpponentFTA'], axis=1)
list(reg_season_all)


# Collapse regular season sums/means

reg_season_means = reg_season_all.groupby(['TeamID','Season']).mean().reset_index()
reg_season_means = reg_season_means.drop(['wins','wins_top25','wins_top5'], axis=1)

reg_season_sum = reg_season_all.groupby(['TeamID','Season']).sum().reset_index()
reg_season_sum = reg_season_sum.drop(['Score','OpponentScore','TOmargin','PointMargin','FG','FG3','FGopponent','FG3opponent','FT'], axis=1)

regseason_df = pd.merge(reg_season_means, reg_season_sum, left_on=['TeamID','Season'], right_on = ['TeamID','Season'], how='left')

regseason_df.to_csv('Data/~Created data/regseason_df.csv', index=False)


# Combine regular season stats with game level data

#Merge the avg regular season performance with all games and have performance wide: 
performance_list = ['wins','wins_top5','wins_top25','Score','OpponentScore','TOmargin','PointMargin','FG','FGopponent','FG3','FG3opponent','FT','RBMargin']
for hr in home_road:
    all_games_df = pd.merge(all_games_df, regseason_df, left_on=[hr+'TeamID','Season'], right_on = ['TeamID','Season'], how='left')
    for var in performance_list:
        all_games_df.rename(columns={var: hr+var}, inplace=True)
    all_games_df = all_games_df.drop(['TeamID'], axis=1)

# Add binary experience dummy

temp_df = all_games_df[all_games_df['is_tourney'] == 1]
temp_df_home = temp_df[['home']]
temp_df_home.rename(columns={'home': 'team'}, inplace=True)
temp_df_road = temp_df[['road']]
temp_df_road.rename(columns={'road': 'team'}, inplace=True)
tourney_teams_df = temp_df_home.append(temp_df_road)
tourney_teams_df['team'].value_counts(dropna=False)

all_games_df['Htourny20plus'] = 0
all_games_df['Rtourny20plus'] = 0

experienced_teams = ['kansas','north carolina','kentucky','duke','michigan st.','wisconsin','florida','villanova','gonzaga','louisville','arizona','xavier','connecticut','syracuse','butler','ohio st.','ucla','west virginia','texas','michigan','pittsburgh','memphis','oregon']
for team in experienced_teams:
    all_games_df.loc[all_games_df['home']==team, 'Htourny20plus'] = 1
    all_games_df.loc[all_games_df['road']==team, 'Rtourny20plus'] = 1

temp_df = all_games_df[all_games_df['is_tourney'] == 1]
temp_df_home = temp_df[['Hconf']]
temp_df_home.rename(columns={'Hconf': 'conf'}, inplace=True)
temp_df_road = temp_df[['Rconf']]
temp_df_road.rename(columns={'Rconf': 'conf'}, inplace=True)
tourny_teams_df = temp_df_home.append(temp_df_road)
tourny_teams_df['conf'].value_counts(dropna=False)

all_games_df['HBig4Conf'] = 0
all_games_df['RBig4Conf'] = 0
conferences = ['ACC','B10','B12','SEC']
for conf in conferences:
    all_games_df.loc[all_games_df['Hconf']==conf, 'HBig4Conf'] = 1
    all_games_df.loc[all_games_df['Rconf']==conf, 'RBig4Conf'] = 1

#Note: only fills in seeds and some games missing line
#mean imputations of variables for model
all_games_df = all_games_df.fillna(all_games_df.mean())

# Output to csv

all_games_df.to_csv('Data/~Created data/all_games_df.csv', index=False)

