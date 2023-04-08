# Anshuboom_Repository_SharkAttackFiles
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
ChartCompilation -- as you run the individual cells describes the Shark Data found in dfAttacks.csv
It describes the sharks and their individual implications in the attacks
It describes the countries and their individual implications in the attacks
It also shows the risk of each shark and country as well as the oceans where attacks have occurred
The last integer argument in each of the bar charts is the number to subset the x axis to make the chart more presentable and readable since all charts are pre sorted going from highest to lowest, to chart the top 4 you only need to set that integer to 4
#####################################################################################
mapChecker is the prediction notebook
1. It loads the assets
2. It presents a form that requires "Gender", "Location", "TimeSlot" and "Activity"
3. Location is geolocated using Googlemaps Geolocator
4. Activity is NLP'd so you can type full sentences "Snorkeling near th Barrier Reef"
5. When you click submit, it checks the inputs, creats an X row and feeds it into the trained model
6. It receives the predictions and probailities and displays a blurb with the results
7. The next cell uses the geolocation, finds the k nearest attacks around the location and plots them, each neigbor has a proximity/range circle with radius 20KM
8. The idea is to show you your location in proximity to know ATTACKS (not fatalities) in the area
9. The next cell then determines the possibility of the appropriate sharks that would be culprit if a FATAL attack were to occur based on their stats, habitat and history according to the ISAF
10. What you get is either the KNOWN sharks in the area identified by the records t have attacked/killed OR if the attacks were unidentified, the main sharks based on temperature conditions that could be resposible in that area as well as their fatality probability. It charts them on a bar chart.
