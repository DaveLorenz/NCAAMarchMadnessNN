# TL;DR
I used historical NCAA men's basketball data to train a neural network that predicted the outcomes of each game in the 2019 NCAA tournament. The model achieved a 0.42788 log loss and placed 4th in the Kaggle Google Cloud & NCAA® ML Competition 2019-Men's.

# About me
I am master's candidate at the University of Virginia's Business Analytics program. An integral component of this part-time degree program is the application of machine learning to business problems. This was my first Kaggle competition, but our UVA cohort regularly has prediction challenges with a leaderboard for assignments. I also regularly wrangle data and build econometric models for my job as an Economic Consultant. I largely credit my success in the competition to my educational training at UVA, training on the job, and some hard work. I spent about 30 hours collection data and training/testing my model for the competition.

# Summary
I collected and combined historical regular season data, tournament data, and metrics from leading basketball statisticians from 2004-2019. Ken Pom's offensive and defensive efficiency metrics were important features. I trained a variety of models (boosted trees, random forest, neural network) and hyperparameters (learning rate, max depth of trees, number of hidden layers/nodes). The neural network that performed the best is depicted below. I wrote my code in Python and used scikit-learn and xgboost to train, test, and evaluate models. Training the model on the 60K observations takes less than 15 minutes. 



I trained my model using both tournament and regular season games and tested on two seperate holdout sets: (1) 2016 tournment only and (2) 2017/2018 tournments. A holdout set is some portion of historical data that you do not allow your model to “see.” This allows you to see how your model performs out of sample and evaluate where your model underfit or overfit.

