# Anshuboom_Repository_SharkAttackFiles
 Anshu Personal Repository
This Readme file has the following main objective:
###################################################################################

if you want to run the model how do you load the assets and which notebooks you run
###################################################################################

In the myCode.py file there are 2 places where you will need to insert a googeAPI key since this project uses geolocation
seach for: "googlemaps.Client(key=" to enter your personal key (I got a warning from Google when  made the repository public)

1. Data folder contains all the necessary, latest (at commit time) datafiles
   1. Go through the myCode.py or myCode.ipynb and find the load section at the top and at the very bottom of the file
   2. change the directory path according to where you download and save the data assets (of the Data folder) on your local pc
2. The main results notbooks are:
   1. ChartCompilation.ipynb found in notebooks
   2. mapchecker.ipynb found in notebooks
   
   As long as the datafile paths have been corrected in myCode.ipynb, each of the above loads the data for you.
3. Make sure you run each cell of the ChartCompilation and mapChecker (especially %run myCode.ipynb)
   
####################################################################################

ChartCompilation --The point of this file is to acquaint the user to the data as it describes the Shark Data found in dfAttacks.csv
It describes the sharks and their individual implications in the attacks
It describes the countries and their individual implications in the attacks
It also shows the risk of each shark and country as well as the oceans where attacks have occurred
The last integer argument in each of the bar charts is the number to subset the x axis to make the chart more presentable and readable since all charts are pre sorted going from highest to lowest y values;  
EXAMPLE:
  topProbabilityofFatalities(dfCountry, 'Country', 'FatalityProbability', 40)
  to chart the top 4, you only need to set that last integer to 4

#####################################################################################

mapChecker is the prediction notebook
1. It loads the assets
2. It presents a form that requires the input of "Gender", "Location", "TimeSlot" and "Activity" (which are used to compose a valid X record)
3. Location is geolocated using Googlemaps Geolocator
4. Activity is NLP'd so you can type full sentences "Snorkeling near th Barrier Reef"
5. When you click submit, it checks the inputs, creats an X row and feeds it into the trained model
6. It receives the predictions and probabilities and displays a blurb with the results
7. The next cell uses the geolocation, finds the k nearest attacks around the location and plots them, each neigbor has a proximity/range circle with radius 20KM
8. The idea is to show you your location in proximity to known ATTACKS (not fatalities) in the area
9. The next cell then determines the possibility of the appropriate sharks that would be culprit if a FATAL attack were to occur based on their stats, habitat and history according to the ISAF
10. What you get is either the KNOWN sharks in the area identified by the records t have attacked/killed OR if the attacks were unidentified, the main sharks based on temperature conditions that could be resposible in that area as well as their fatality probability. It charts them on a bar chart.

######################################################################################

My Sources:
Much of the data enrichment came from reading up about the sharks, their habitat preferences, etc from wikipaedia
example:https://en.wikipedia.org/wiki/Wobbegong#:~:text=They%20are%20found%20in%20shallow,as%20far%20north%20as%20Japan.

The actual DataSource downloaded using the API provided by:
https://public.opendatasoft.com/explore/dataset/global-shark-attack/api/?disjunctive.country&disjunctive.area&disjunctive.activity
This data eventually, after the operations of wrangler.ipynb then wrangler2.ipynb was converted to dfAttacksX.csv which was used in the modeling
I did not do ANY scraping in this project and I am QUITE happy about that

I tend to binge watch all shark attack videos on Youtube and started watching "Sharks Happen" which got me interesed and inspired to addopt this project topic
http://sharkshappen.com : what got me going was the "Sharks Happen Stats" xlsx file which the author generously provides. I looked at it and got the idea.
The file itself was too messy to clean up and only consists 474 records but individual records were used to enrich the dfAttacks.csv file above based on victim names, I am very grateful to the auhor for this because I appreciate his interest, motivation and efforts

In order to lookup water temperatures of the seas around given locations in order to compile my sharks dictionary, I also used 
https://www.seatemperature.org/.










