# Anshuboom_Repository
 Anshu Personal Repository
This Readme file has the following main objective:
###################################################################################
if you want to run the model how do you load the assets and which notebooks you run
###################################################################################

1. Data folder contains all the necessary, latest (at commit time) datafiles
   1. Go through the myCode.py or myCode.ipynb and find the load section at the top and at the very bottom of the         file
   2. change the directory path according to where you download and save the data assets on your local pc
2. The main results notbooks are:
   1. ChartCompilation.ipynb found in notebooks
   2. mapchecker.ipynb found in notebooks
   
   As long as the datafile paths have been corrected in myCode.ipynb, each of the above loads the data for you.
3. Make sure you run each cell of the ChartCompilation and mapChecker
   
####################################################################################
ChartCompilation -- ass you run the individual cells describes the Shark Data found in dfAttacks.csv
It describes the sharks and their individual implications in the attacks
It describes the countries and their individual implications in the attacks
It also shows the risk of each shark and country as well as the oceans where attacks have occurred
The last integer argument in each of the bar charts is the number to subset the x axis to make the chart more presentable and readable since all charts are pre sorted going from highest to lowest, to chart the top 4 you only need to set that integer to 4
