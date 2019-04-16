
# Load packages

import time
import numpy as np
import pandas as pd

# Load data

kp_df=[]
years_list = [2019,2018,2017,2016,2015,2014,2013,2012,2011,2010,2009,2008,2007,2006,2005,2004,2003,2002]
for year in years_list:
    temp_kp_df = pd.read_csv("Data/Ken Pom/KP" + str(year) + "_final.csv")
    temp_kp_df['Season'] = year
    year_last = year   
    if year==2019:
        kp_df = temp_kp_df
    else:
        kp_df = kp_df.append(temp_kp_df)

kp_df.head()
kp_df.tail()

kp_df['Season'].value_counts(dropna=False)

kp_df['team'] = kp_df['team'].str.lower()

# Match Team IDs to data

teams_df = pd.read_csv('Data/Kaggle NCAA/TeamSpellings.csv', sep='\,', engine='python')
teams_df.head()

kp_df = pd.merge(kp_df, teams_df, left_on=['team'], right_on = ['TeamNameSpelling'], how='left')
kp_df = kp_df.drop(['TeamNameSpelling'], axis=1)

kp_df['Season'].value_counts(dropna=False)
kp_df.head()

# Write the data to a csv
kp_df.to_csv('Data/~Created data/kp_all.csv', index=False)

