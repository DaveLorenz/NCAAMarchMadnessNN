# TL;DR
I used historical NCAA men's basketball data to train a neural network that predicted win probabilities for each game in the 2019 NCAA tournament. The neural network achieved a 0.42788 log loss and placed 4th in the Kaggle Google Cloud & NCAA® ML Competition 2019-Men's.

# About me
I am master's candidate at the University of Virginia's Business Analytics program. An integral component of this part-time degree program is the application of machine learning to business problems. This was my first Kaggle competition, but our UVA cohort regularly has prediction challenges with a leaderboard for assignments. I also wrangle data and build econometric models for my job as an Economic Consultant. 

# Summary
I collected and combined historical regular season data, tournament data, and metrics from leading basketball statisticians from 2004-2019. Ken Pom's efficiency margin was the important feature. I trained a variety of models (i.e., boosted trees, random forest, neural network) and tuned hyperparameters (e.g., learning rate, max depth of trees, number of hidden layers/nodes). The neural network performed the best. This neural network with the hidden layers and nodes I used to enter the competition is depicted below. I wrote my code in Python and used scikit-learn and xgboost to train, test, and evaluate models. Training the model on the 60K observations takes less than 15 minutes. My model picked UVA to win all of their games (go Hoos!).

![Image of NN](https://user-images.githubusercontent.com/49622342/56206156-b219e080-6019-11e9-9fbb-130a5ed24ce1.PNG)

# Feature selection
My final model included the following 12 features for each team (i.e., 24 total team-specific features):
1. Ken Pom end of season ranking
2. Ken Pom efficiency margin
3. Ken Pom offensive efficiency
4. Ken Pom defensive efficiency
5. Ken Pom "luck"
6. Regular season wins against top 25 ranked teams
7. Regular season average point margin
8. Regular season field goal percentage
9. Regular season 3-point field goal percentage
10. Binary indicator for whether team played 20+ tournament games during 2004-2018
11. Binary indicator for whether team is in the ACC, B10, B12, or SEC
12. Seed of team

I also included a binary indicator for whether the game was a tournament game. I collected Vegas line data and considered using line as a feature for the first round (similar to Matthews and Lopez approach in 2014 NCAA Kaggle win). Note that line is not available for predicting the tournament after the first round, because we don't yet know the matchups. However, the line can help predict the first round. Surprisingly, line did not improve model log loss. Therefore, I decided to exclude it. It is difficult to extract feature importance in a neural network. The figure below highlights that the efficiency margin (home and road) was very important in the boosted tree model.

![Image of xgb](https://user-images.githubusercontent.com/49622342/56207337-77fe0e00-601c-11e9-8c53-2a619915f538.PNG)

Note: Hadjem and Radjem are Ken Pom home and road team efficiency margins, respectively.

# Model
As mentioned above, I trained a variety of models (boosted trees, random forest, neural network) and tuned hyperparameters (learning rate, max depth of trees, number of hidden layers/nodes). The neural network with 3 hidden layers (nodes of 7, 5, 3) depicted above performed the best (i.e., had the lowest log loss). More specifically, the neural network I entered into the competition had a log loss of 0.42-0.45 in various test sets. The other two models were above 0.50 log loss. I did not consider ensembling the various models, because the neural network was far superior to the random forest and boosted trees. In future competitions, I plan to evaluate whether the application of a soft voting classifier improves model performance. A voting classifier would (1) apply greater weight to the neural network and (2) take advantage of model diversity.

I trained my model using both tournament and regular season games and tested on two seperate holdout sets: (1) 2016 tournment only and (2) 2017/2018 tournments. A holdout set is some portion of historical data that you do not allow your model to “see.” This allows you to see how your model performs out of sample and evaluate where your model underfit or overfit. I tuned a number of hyperparameters for the neural network. Unsurprisingly, the relu activation function performed better in the test set than sigmoid activation function (i.e., logistic). I also tried a number of hidden layers and nodes within those layers until I was able to find the combination that led to the lowest test log loss. Given that there is no risk involved with entering a Kaggle competition, I did not focus much on model stability. However, my model's log loss was 0.42 in one holdout set and 0.45 in another, which suggests the predictions are fairly stable. 



