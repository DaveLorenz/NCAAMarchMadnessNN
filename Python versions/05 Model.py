# Load packages

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from collections import Counter

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve, train_test_split
from sklearn.metrics import brier_score_loss

import xgboost as xgb
from xgboost import XGBClassifier

# Load data

all_games_df = pd.read_csv('Data/~Created data/all_games_df.csv')

test_combos_df = pd.read_csv('Data/~Created data/test_combos_df_19.csv')
test_combos_df = test_combos_df.sort_values(by=['ID']).reset_index(drop=True)

ind_var_selected = [
'is_tourney', 
'HRankPOM',
'RRankPOM',
'line',
'Hwins_top25',
'Rwins_top25',
'HPointMargin',
'RPointMargin',
'HFG',
'RFG',
'HFG3',
'RFG3',
'Hadjem',
'Hadjo',
'Hadjd',
'Hluck',
'Radjem',
'Radjo',
'Radjd',
'Rluck',
'Htourny20plus',
'Rtourny20plus',
'HBig4Conf',
'RBig4Conf', 
'HSeed',
'RSeed'
]

# Note: test is 2019 predictions but our "test" holdout set is referred to as "valid" 

#prediction set 2019
test_ids = test_combos_df['ID'].reset_index(drop=True)
X_test = test_combos_df[['is_tourney','HRankPOM','RRankPOM','Hwins_top25','Rwins_top25','HPointMargin','RPointMargin','HFG','RFG','HFG3','RFG3','Hadjem','Hadjo','Hadjd','Hluck','Radjem','Radjo','Radjd','Rluck','Htourny20plus','Rtourny20plus','HBig4Conf','RBig4Conf','HSeed','RSeed']].reset_index(drop=True)

#Predict the last two years as a test set (2017, 2018):
temp_df = all_games_df[all_games_df['Season']>2016]
temp_df = temp_df[temp_df['is_tourney']==1]
X_valid = temp_df[ind_var_selected].reset_index(drop=True)
y_valid = temp_df['Hwin'].reset_index(drop=True)

#Train on everything else:
temp_df1 = all_games_df[all_games_df['Season']>2016]
temp_df1 = temp_df1[temp_df1['is_tourney']==0]
temp_df2 = all_games_df[all_games_df['Season']<2017]
combined_temp_df = temp_df1.append(temp_df2)

X_train = combined_temp_df[ind_var_selected].reset_index(drop=True)
y_train = combined_temp_df['Hwin'].reset_index(drop=True)

#For final predictions:
X_train_orig = all_games_df[ind_var_selected].reset_index(drop=True)
y_train_orig = all_games_df['Hwin'].reset_index(drop=True)

#Create second holdout set to double-check not overfit and check model stability (season 2016)
temp_df16 = all_games_df[all_games_df['Season']==2016]
temp_df16 = temp_df16[temp_df16['is_tourney']==1]
X_valid16 = temp_df16[ind_var_selected].reset_index(drop=True)
y_valid16 = temp_df16['Hwin'].reset_index(drop=True)

temp_df1_16 = all_games_df[all_games_df['Season']==2016]
temp_df1_16 = temp_df1_16[temp_df1_16['is_tourney']==0]
temp_df2_16 = all_games_df[all_games_df['Season']!=2016]
combined_temp_df_16 = temp_df1_16.append(temp_df2_16)

X_train16 = combined_temp_df_16[ind_var_selected].reset_index(drop=True)
y_train16 = combined_temp_df_16['Hwin'].reset_index(drop=True)

X_test = X_test.astype("float64")

X_train_orig = X_train_orig.astype("float64")
y_train_orig = y_train_orig.astype("float64")

X_train = X_train.astype("float64")
X_valid = X_valid.astype("float64")
y_train = y_train.astype("float64")
y_valid = y_valid.astype("float64")

X_train16 = X_train16.astype("float64")
X_valid16 = X_valid16.astype("float64")
y_train16 = y_train16.astype("float64")
y_valid16 = y_valid16.astype("float64")

# Scoring rules and two benchmarks

def LogLoss(predictions, realizations):
    predictions_use = predictions.clip(0)
    realizations_use = realizations.clip(0)
    LogLoss = -np.mean( (realizations_use * np.log(predictions_use)) + 
                        (1 - realizations_use) * np.log(1 - predictions_use) )
    return LogLoss

# If the model doesn't beat assuming 50% it is poor

bench_5050 = np.repeat(0.5, len(y_valid))
LogLoss(bench_5050, y_valid)

# How does this compare to Lopez and Matthews (2014 winners)?

Z1 = LogisticRegression(C = 1e9, random_state=23)
Z1.fit(X_train[['line']], y_train)
Z1_pred = pd.DataFrame(Z1.predict_proba(X_valid[['line']]))[1]
LogLoss(Z1_pred, y_valid)

Z2 = LogisticRegression(C = 1e9, random_state=23)
Z2.fit(X_train[['Hadjo','Hadjd','Radjo','Radjd']], y_train)
Z2_pred = pd.DataFrame(Z2.predict_proba(X_valid[['Hadjo','Hadjd','Radjo','Radjd']]))[1]
LogLoss(Z2_pred, y_valid)

Z1 = LogisticRegression(C = 1e9, random_state=23)
Z1.fit(X_train16[['line']], y_train16)
Z1_pred = pd.DataFrame(Z1.predict_proba(X_valid16[['line']]))[1]
LogLoss(Z1_pred, y_valid16)

Z2 = LogisticRegression(C = 1e9, random_state=23)
Z2.fit(X_train16[['Hadjo','Hadjd','Radjo','Radjd']], y_train16)
Z2_pred = pd.DataFrame(Z2.predict_proba(X_valid16[['Hadjo','Hadjd','Radjo','Radjd']]))[1]
LogLoss(Z2_pred, y_valid16)

# Fit a neural network (with and without line)

# Normalize data (using z-scores) before neural network

scaler = StandardScaler()
scaler.fit(X_train)  # Fit only to the training data
scaled_X_train = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
scaled_X_valid = pd.DataFrame(scaler.transform(X_valid), index=X_valid.index, columns=X_valid.columns)

scaler = StandardScaler()
scaler.fit(X_train16)  # Fit only to the training data
scaled_X_train16 = pd.DataFrame(scaler.transform(X_train16), index=X_train16.index, columns=X_train16.columns)
scaled_X_valid16 = pd.DataFrame(scaler.transform(X_valid16), index=X_valid16.index, columns=X_valid16.columns)

#drop line from training since we won't use in predictions, need these to be same number of columns.
X_train_orig = X_train_orig.drop(['line'], axis=1)

scaler = StandardScaler()
scaler.fit(X_train_orig)  # Fit to all training data

scaled_X_train_orig = pd.DataFrame(scaler.transform(X_train_orig), index=X_train_orig.index, columns=X_train_orig.columns)
scaled_X_test = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)


# With line (note: we won't have line in the rounds after the first, but we could use this for the first round only like Lopez and Matthews did) 
#Note: I tried logistic activation and different combinations of hidden layers/nodes
#Hyperparameters below minimized the log loss in the holdout set
#I also submited a prediction with 10 nodes in the first layer, but this is the submission that placed 4th (w/ 8 in 1st)
nn = MLPClassifier(activation='relu', hidden_layer_sizes=(8,5,3),random_state=201, max_iter=1000)
nn.fit(scaled_X_train,y_train)
nn_pred = pd.DataFrame(nn.predict_proba(scaled_X_valid))[1]
LogLoss(nn_pred, y_valid)

#try second holdout (does worse, but still better than baseline of 54)
nn.fit(scaled_X_train16,y_train16)
nn_pred16 = pd.DataFrame(nn.predict_proba(scaled_X_valid16))[1]
LogLoss(nn_pred16, y_valid16)

# Without line
#Note: I tried logistic activation and different combinations of hidden layers/nodes
#Hyperparameters below minimized the log loss in the holdout set
ind_var_selected_no_line = ['is_tourney', 'Hwins_top25','Rwins_top25','HPointMargin','RPointMargin','HFG','RFG','HFG3','RFG3','HRankPOM','RRankPOM','Hadjem','Hadjo','Hadjd','Hluck','Radjem','Radjo','Radjd','Rluck','Htourny20plus','Rtourny20plus','HBig4Conf','RBig4Conf', 'HSeed','RSeed']
nn = MLPClassifier(activation='relu', hidden_layer_sizes=(7,5,3),random_state=201, max_iter=1000)
nn.fit(scaled_X_train[ind_var_selected_no_line],y_train)
nn_pred_no_line = pd.DataFrame(nn.predict_proba(scaled_X_valid[ind_var_selected_no_line]))[1]
LogLoss(nn_pred_no_line, y_valid)

#try second holdout (does better)
nn.fit(scaled_X_train16[ind_var_selected_no_line],y_train16)
nn_pred_no_line16 = pd.DataFrame(nn.predict_proba(scaled_X_valid16[ind_var_selected_no_line]))[1]
LogLoss(nn_pred_no_line16, y_valid16)

# Try avg of line and no line:

avg = (nn_pred_no_line+nn_pred)/2
LogLoss(avg, y_valid)

avg16 = (nn_pred_no_line16+nn_pred16)/2
LogLoss(avg16, y_valid16)

# Create test predictions

#different submissions: differ by first layer of nueral net
ind_var_selected_no_line = ['is_tourney', 'Hwins_top25','Rwins_top25','HPointMargin','RPointMargin','HFG','RFG','HFG3','RFG3','HRankPOM','RRankPOM','Hadjem','Hadjo','Hadjd','Hluck','Radjem','Radjo','Radjd','Rluck','Htourny20plus','Rtourny20plus','HBig4Conf','RBig4Conf', 'HSeed','RSeed']

#train model on all data (previously held out some tournaments for a test set)
nn = MLPClassifier(activation='relu', hidden_layer_sizes=(7,5,3),random_state=201, max_iter=1000)
nn.fit(scaled_X_train_orig[ind_var_selected_no_line],y_train_orig)
second_rd_submission_all = pd.DataFrame(nn.predict_proba(scaled_X_test[ind_var_selected_no_line]))

#Note: I'm predicting home (lower seed) win probability. Need to convert to be consistent with output file (lower team ID)

second_rd_submission = pd.merge(test_combos_df, second_rd_submission_all, left_index=True, right_index=True)

second_rd_submission.loc[second_rd_submission['HTeamID']<second_rd_submission['RTeamID'], 'pred'] = second_rd_submission[1]
second_rd_submission.loc[second_rd_submission['HTeamID']>second_rd_submission['RTeamID'], 'pred'] = second_rd_submission[0]

second_rd_submission.to_csv('Data/~Created data/second_rd_submission_all.csv', index=False)

second_rd_submission = second_rd_submission[['ID','pred']]

#Export to submit to Kaggle
second_rd_submission.to_csv('Data/~Created data/second_rd_submission.csv', index=False)


# Other models

# No model performed as well as the neural network

# XGBoost

X_train = X_train[['is_tourney', 'Hwins_top25','Rwins_top25','HPointMargin','RPointMargin','HFG','RFG','HFG3','RFG3','HRankPOM','RRankPOM','Hadjem','Hadjo','Hadjd','Hluck','Radjem','Radjo','Radjd','Rluck','Htourny20plus','Rtourny20plus','HBig4Conf','RBig4Conf', 'HSeed','RSeed']]
X_valid= X_valid[['is_tourney', 'Hwins_top25','Rwins_top25','HPointMargin','RPointMargin','HFG','RFG','HFG3','RFG3','HRankPOM','RRankPOM','Hadjem','Hadjo','Hadjd','Hluck','Radjem','Radjo','Radjd','Rluck','Htourny20plus','Rtourny20plus','HBig4Conf','RBig4Conf', 'HSeed','RSeed']]
X_train_xgb = xgb.DMatrix(X_train, label = y_train)
X_valid_xgb = xgb.DMatrix(X_valid)

num_round_for_cv = 1000
param = {'max_depth':3, 'eta':0.01, 'seed':201, 'objective':'binary:logistic', 'nthread':2}

p = xgb.cv(param,
       X_train_xgb,
       num_round_for_cv,
       nfold = 5,
       show_stdv = False,
       verbose_eval = False,
       as_pandas = False)

p = pd.DataFrame(p)
use_num = p['test-error-mean'].argmin()

num_round = use_num
xgb_train = xgb.train(param, X_train_xgb, num_round)
xgb_valid_prob = pd.Series(xgb_train.predict(X_valid_xgb))

xgb.plot_importance(xgb_train)
plt.rcParams['figure.figsize'] = [20, 20]
plt.show()

LogLoss(xgb_valid_prob, y_valid)

# Random forest

clf = RandomForestClassifier(n_estimators=200, max_depth=3, min_samples_leaf=3)
clf.fit(X_train, y_train)
rf_prob = pd.DataFrame(clf.predict_proba(X_valid))
LogLoss(rf_prob[1], y_valid)

avg = (rf_prob[1]+xgb_valid_prob+nn_pred)/3
LogLoss(avg, y_valid)

