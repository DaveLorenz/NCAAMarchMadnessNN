# NCAAMarchMadnessNN
My supporting code for Google Cloud &amp; NCAA® ML Competition 2019-Men's (4th place finish)

Below you can find a outline of how to reproduce my solution for the NCAA® ML Competition 2019-Men's competition.
If you run into any trouble with the setup/code or have any questions please contact me at dave.a.lorenz@gmail.com

#HARDWARE (this should have no problem running locally on an average laptop or deskop)
--Processor	Intel(R) Core(TM) i5-6200U CPU @ 2.30GHz, 2400 Mhz, 2 Core(s), 4 Logical Processor(s)
--8GB RAM
--Windows versoin 10

#SOFTWARE (python packages are detailed separately in `requirements.txt`):
--Python 3.7 64-Bit (https://www.anaconda.com/download/)

#DATA
1. Point spreads from: http://www.thepredictiontracker.com/ncaaresults.php
2. Ken Pom data from: https://kenpom.com/index.php (note: 2019 data collected prior to tournament)
3. Kaggle data from: https://www.kaggle.com/c/mens-machine-learning-competition-2019/data

#SCRIPTS
1. 

#How to reproduce competition results:

--Download input files from https://www.kaggle.com/c/mens-machine-learning-competition-2019/data: Stage2DataFiles.zip, MasseyOrdinals_thru_2019_day_128.zip
    
--Move following files above to 'Data/Kaggle NCAA': MasseyOrdinals_thru_2019_day_128.csv, NCAATourneyCompactResults.csv, NCAATourneySeeds.csv, RegularSeasonDetailedResults.csv, TeamSpellings.csv

 --Run scripts 1-6 (i.e., 
        
