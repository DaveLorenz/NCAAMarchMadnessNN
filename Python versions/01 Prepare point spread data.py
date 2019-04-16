# Load packages

import numpy as np
import pandas as pd

# Load data

spread_df=[]
years_list = [18,17,16,15,14,13,12,11,10,'09','08','07','06','05','04','03']
year_int = 2019
for year in years_list:
    #Note: 18 refers to 2019 and so on
    temp_spread_df = pd.read_csv("Data/Point spreads/ncaabb" + str(year) + ".csv")
    temp_spread_df['Season'] = year_int
    year_last = year
    year_int = year_int-1    
    if year==18:
        spread_df = temp_spread_df
    else:
        spread_df = spread_df.append(temp_spread_df)

spread_df['home'] = spread_df['home'].str.lower()
spread_df['road'] = spread_df['road'].str.lower()

spread_df.head()

# Match Team IDs to data

teams_df = pd.read_csv('Data/Kaggle NCAA/TeamSpellings.csv', sep='\,', engine='python')

teams_df.head()

team_list = ['home','road']
for team in team_list:
    spread_df = pd.merge(spread_df, teams_df, left_on=[team], right_on = ['TeamNameSpelling'], how='left')
    if team=='home':
        spread_df.rename(columns={'TeamID': 'HTeamID'}, inplace=True)
    if team=='road':
        spread_df.rename(columns={'TeamID': 'RTeamID'}, inplace=True)
    spread_df = spread_df.drop(['TeamNameSpelling'], axis=1)


# Note: change to home and road team and score

spread_df.loc[spread_df['hscore']>spread_df['rscore'], 'WTeamID'] = spread_df['HTeamID']
spread_df.loc[spread_df['hscore']<spread_df['rscore'], 'LTeamID'] = spread_df['HTeamID']

spread_df.loc[spread_df['hscore']>spread_df['rscore'], 'WScore'] = spread_df['hscore']
spread_df.loc[spread_df['hscore']<spread_df['rscore'], 'LScore'] = spread_df['hscore']

spread_df.loc[spread_df['rscore']>spread_df['hscore'], 'WTeamID'] = spread_df['RTeamID']
spread_df.loc[spread_df['rscore']<spread_df['hscore'], 'LTeamID'] = spread_df['RTeamID']

spread_df.loc[spread_df['rscore']>spread_df['hscore'], 'WScore'] = spread_df['rscore']
spread_df.loc[spread_df['rscore']<spread_df['hscore'], 'LScore'] = spread_df['rscore']

spread_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['WTeamID','LTeamID','WScore','LScore','line'])
spread_df = spread_df[spread_df['WScore'] != "."]
spread_df = spread_df[spread_df['LScore'] != "."]
spread_df = spread_df[spread_df['line'] != "."]

spread_df['line'].value_counts(dropna=False)

#drop spreads with bad coverage
spread_df = spread_df.drop(['line7ot','lineargh','lineash','lineashby','linedd','linedunk','lineer','linegreen','linemarkov','linemass','linepib','linepig','linepir','linepiw','linepom','lineprophet','linerpi','lineround','linesauce','lineseven','neutral','lineteamrnks','linetalis','lineespn','linemassey','linedonc','linesaggm','std','linepugh','linefox','linedok','lineopen'], axis=1)

spread_df.tail()

spread_df.dropna(subset=['line'])

#Write the data to a csv
spread_df.to_csv('Data/~Created data/spreads_all.csv', index=False)

