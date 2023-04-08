import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import seaborn as sns
import json
import pickle
import pickle as pkl
import googlemaps
import folium
from geopy.distance import distance
from IPython.display import display, HTML
import time
from tqdm import tqdm
import random
import csv
from sklearn.neighbors import BallTree, KDTree
from sklearn.metrics.pairwise import haversine_distances
import ast
import warnings 
warnings.filterwarnings('ignore')
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.eval_measures import rmse
import plotly.express as px
import spacy
gmaps = googlemaps.Client(key="AIzaSyBUN8KXWnNAoS36tpdZikfBWaNIEKfJ3-8")
import joblib
import ipywidgets as widgets
from unidecode import unidecode
import itertools

    #HELPER FUNCTIONS
nlp = spacy.load('en_core_web_sm')
def tokenize_text(text):
    return [tok.text for tok in nlp(text)]



###LOADING ALL ASSETS############################################
##Load and Test Dictionaries 
allSharks = loadMyDictionary("allSharkCounts.pkl")
sharksbyregion= loadMyDictionary("sharksbyRegion.pkl")
identifiedvsunidentified = loadMyDictionary("identifiedvsunidentified.pkl")
sharkSet={'lemon', 'grey reef', 'blacktip', 'unknown', 'tiger', 'porbeagle', 'bull', 'spinner', 'silky', 'sandbar', 'nurse', 'oceanic whitetip', 'mako', 'blue', 'sand tiger', 'white shark', 'bronze whaler', 'sevengill', 'zambezi', 'thresher', 'whitetip reef', 'hammerhead', 'wobbegong', 'grey nurse', 'goblin', 'dusky', 'galapagos', 'raggedtooth'}
#Load dfAttacks
dfAttacks=pd.read_csv("dfAttacksX.csv")
dfCountry = pd.read_csv("countryStatsX.csv")
dfSharks = pd.read_csv("SharkStatsX.csv")
fdf = pd.read_csv("fatalities.csv")
sharkRisk = pd.read_csv('sharkBoost.csv')
###LOADING MY PIPELINE###########################################

with open('trained_pipeline_LogisticRegression.pkl', 'rb') as f:
    pipeline = pickle.load(f)

###GLOBALS#####################################################
myInputs = []
theResults = []
rowDF = pd.DataFrame()




####THE FORM########################################################

def inputForm():
    # Create the input widgets
    gender_dropdown = widgets.Dropdown(options=['M', 'F'], layout=widgets.Layout(width='50px'))
    timeslot_dropdown = widgets.Dropdown(options=['Dawn', 'Morning', 'Midday', 'Afternoon', 'Dusk', 'Night'])
    location_textbox = widgets.Text()
    activity_textbox = widgets.Text()
    
    # Create the labels for the input widgets
    gender_label = widgets.Label('Gender:', layout=widgets.Layout(width='80px'), style={'font-weight': 'bold'})
    timeslot_label = widgets.Label('Timeslot:', layout=widgets.Layout(width='80px'), style={'font-weight': 'bold'})
    location_label = widgets.Label('Location:', layout=widgets.Layout(width='80px'), style={'font-weight': 'bold'})
    activity_label = widgets.Label('Activity:', layout=widgets.Layout(width='80px'), style={'font-weight': 'bold'})

    # Create the input form
    form_items = widgets.VBox([
        widgets.HBox([gender_label, gender_dropdown]),
        widgets.HBox([timeslot_label, timeslot_dropdown]),
        widgets.HBox([location_label, location_textbox]),
        widgets.HBox([activity_label, activity_textbox])
    ], layout=widgets.Layout(display='flex', flex_flow='column', align_items='flex-start'))


    
    # Define the callback function for the submit button
  
    def submit_form(button):
        myInputs.clear()
        theResults.clear()
        myInputs.extend([gender_dropdown.value, timeslot_dropdown.value, location_textbox.value, activity_textbox.value])
        
        #print("Making the row")
        pdf = makeXrow(myInputs) 
       
        position = pdf[0]
        #print("This is the position", pdf[0])
        
        rowDF = pdf[1]
        #print ("This is the row", rowDF)
        
        y_pred = pipeline.predict(rowDF)
        
        y_prob = pipeline.predict_proba(rowDF)
        
        y_predAnswer= "YES, if an attack occurs, it COULD well be Fatal" if y_pred =="Y" else "NO, Attack MAY not be Fatal"
        y_predAnswer += f", with a predicted probability of: {y_prob[0][1]}"
        
        print (y_predAnswer)
        
        
        theResults.extend([position, rowDF, y_pred, y_prob])
        
        
        
        
        
    # Create the submit button
    submit_button = widgets.Button(description='Submit')
    
    # Set the on_click attribute of the submit button to the callback function
    submit_button.on_click(submit_form)
    submit_button.button_style = 'success'
    # Add the form and submit button to a container
    form_container = widgets.VBox([form_items, submit_button])

    # Add a CSS class to the form container
    form_container.add_class('form-container')
    
    # Add the style to the page
    display(widgets.HTML("""
    <style>
    .form-container {
        background-color: lightblue;
        padding: 50px;
        color:white;
    }
    </style>
    """))


    # Display the form container
    #display(form_container)
    return form_container
    
##############MY TRUSTED HELPERS##########################################
    
    
    
def nanalysis(df):
    a=df.isnull().sum().sum()
    b=len(df)
    cost = round((a/b)*100,2)
    
    print("There are Rows with NaN: ", df.isnull().values.any())  
    print("How many rows with NaN? ", df.isnull().sum().sum())
    print("Doing a blanket dropNA will cost you:",cost,"% of your rows")
    print("The rows with NaN are:\n\n", df[df.isna().any(axis=1)])
    print("The columns with NaN are:\n\n", df.isna().any())

def get_latlong(location,gmaps):
    geocode_result = gmaps.geocode(str(location))
    location = geocode_result[0]['geometry']['location']
    latlong = [location['lat'],location['lng']]
    return latlong

def display_location(loc,gmaps):
    locationtup = get_latlong(loc,gmaps)
    lat=float(locationtup[0])
    long=float(locationtup[1])    
    map = folium.Map(location=[lat,long], zoom_start=15)    
    folium.Marker([lat,long]).add_to(map)
    #map.save("map.html")
    return map
    
def display_locations(coordinates):
    map = folium.Map(location=coordinates[0], zoom_start=1)
    marker_group = folium.FeatureGroup()
    for coordinate in coordinates:
        folium.Marker(coordinate).add_to(marker_group)
    
    marker_group.add_to(map)
    map.fit_bounds(marker_group.get_bounds())
   
    return map


def proximity_in_KM(myloc, attackloc):
    from geopy.distance import distance

    # Define the coordinates of the two points
    point1 = myloc 
    point2 = attackloc  
    
    # Compute the distance between the two points
    dist = distance(point1, point2).km
    return dist


def dummify(X,cat):
    dummyCopy = X.copy()
    for elem in cat:        
        dummyCopy = dummyCopy.join(pd.get_dummies( dummyCopy[elem], prefix = str(elem), drop_first=True))
        dummyCopy.drop([elem], axis=1, inplace=True)
    
    return dummyCopy

def trimFatX(df,model):
    modelParams = pd.DataFrame(model.params.index.values)
    modelParams.rename(columns={0:'col'}, inplace=True)
    modelParams['p']=np.array(model.pvalues)
    toKeep = modelParams[modelParams["p"] <=0.05]
    keeplist = toKeep['col'].values
    kl=[keeplist[i] for i in range(0,len(keeplist))]
    if kl[0] == 'const':
        const = kl.pop(0)
    trimmedX = df[kl]
    return trimmedX


def fillDates(df):
    pattern = r'^\d{4}\.\d{2}\.\d{2}$'
    
    for index, row in df.iterrows():
        if re.match(pattern, str(row['Case Number'])) and pd.isna(row['Date']):
            df.at[index,'Date'] = row['Case Number']
    return df

def dropNaN(col,df):
    df = df.dropna(subset=[col])
    return df

def fixWrongDateFormat(df):
    pattern = r'^\d{4}\.\d{2}\.\d{2}$'
    for index, row in df.iterrows():
        if re.match(pattern, str(row['Date'])):
            df.at[index, 'Date'] = row['Date'].replace('.', '-')
    return df
       
    
def sortedUniqueCounts(df, column_name):   
    value_counts = df[column_name].value_counts()
    value_counts_dict = dict(value_counts)
    value_counts_dict_sorted = {k: v for k, v in sorted(value_counts_dict.items(), key=lambda item: item[1], reverse=True)}

    return value_counts_dict_sorted


def cleanY(df):
    for index, row in df.iterrows():
        if row['Fatal'] in ["UNKNOWN", "F", "y"]:
            df.at[index,'Fatal'] = "Y"
        elif row['Fatal'] in ["n","2017.0", "Y x 2", "Nq"]:
            df.at[index, 'Fatal'] = "N"
    return df

def extract_sharks(x, sharks):
    if isinstance(x, str):
        x = x.lower()
        shark_list = []
        for shark in sharks:
            shark = shark.lower()
            if shark in x:
                shark_list.append(shark.lower().strip())

        if len(shark_list) > 0:
            return ', '.join(shark_list)
        else:
            return 'Unknown'
    else:
        return 'Unknown'

def classifyMyShark(df, col, targetcol, sharks):
    pattern = '|'.join(sharks)
    df[targetcol] = df[col].astype(str).str.findall(pattern, flags=re.IGNORECASE).apply(', '.join)
    df[targetcol].replace('', 'Unknown', inplace=True)
    #df[targetcol].replace(to_replace=r'(Juvenile )?(.*white.*)', value=r'White \2', regex=True, inplace=True)
    df.dropna(subset=[targetcol], inplace=True)
    df.sort_values(by=[targetcol], inplace=True)
    return df


def consolidate(myset, mydict):
    #first make a new dict
    conDict = {key: 0 for key in myset}
    #now transfer the values
    for key in mydict.keys():
        if "," in key:
            newkeys = key.split(',')
            for elem in newkeys:
                conDict[elem.strip()]+=mydict[key]
        else:
            conDict[key] += mydict[key]
            
    sortedDict = dict(sorted(conDict.items(), key=lambda item: item[1], reverse = True))
    conDict.clear()
    conDict.update(sortedDict)
    return conDict



def barFromDict(dictionary, excludekeys = [], t = "untitled"):
    
    new_dict = {k: v for k, v in dictionary.items() if k not in excludekeys}
    keys = list(new_dict.keys())
    values = list(new_dict.values())
    
    sns.set_style("whitegrid")
    
    ax = sns.barplot(x=keys, y=values, palette=sns.color_palette("muted"))
    
    # Add value annotations to the bars
    for i, v in enumerate(values):
        ax.annotate(str(v), xy=(i, v), ha='center', va='bottom')

    
    ax.set(xlabel='Shark Species', ylabel='Attack Counts so far ')
    ax.set_xticklabels(keys, rotation=90)
    ax.set_title(t)
    plt.show()


def probabilityDF(dict1,dict2):
    myKeys = list(dict2.keys())
    fatalist = []
    attacklist = []
    for key in myKeys:
        fatalist.append(dict2[key])
    
    for key in myKeys:
        attacklist.append(dict1[key])
    
    df = pd.DataFrame({'SharkSpecies': myKeys,
                       'Attacks': attacklist,
                       'Fatalities': fatalist})
    
    
    # do not want the unkown or unidentified attacks to skew my results so i remove them from the counts
    
    df= df.drop(df[df['SharkSpecies'] == 'unknown'].index)
    totalAttacks = df.Attacks.sum()
    df['AttackProbability']=df['Attacks']/totalAttacks
    df['FatalityRatio'] = df['Fatalities']/df['Attacks']
    df['FatalityProbability'] =  df['AttackProbability'] * df['FatalityRatio']
    
    
    return df

    
def fatalityRatioByCountry(dict1,dict2):
    myKeys = list(dict2.keys())
    fatalist = []
    attacklist = []
    
    for key in myKeys:
        fatalist.append(dict2[key])
    
    for key in myKeys:
        attacklist.append(dict1[key])
        
    df = pd.DataFrame({'Country': myKeys,
                       'Attacks': attacklist,
                       'Fatalities': fatalist})   
    totalAttacks = df.Attacks.sum()
    df['AttackProbability']=df['Attacks']/totalAttacks
    df['FatalityRatio'] = df['Fatalities']/df['Attacks']
    df['FatalityProbability'] =  df['AttackProbability'] * df['FatalityRatio']
    
    return df
    
def overlayBar(dict1, dict2, t= "untitled"):
    # Create a Pandas DataFrame from the two dictionaries
    
    ##Making sure Order is right!
    myKeys = list(dict2.keys())
    fatalist = []
    attacklist = []
    for key in myKeys:
        fatalist.append(dict2[key])
    
    for key in myKeys:
        attacklist.append(dict1[key])
    
    
    df = pd.DataFrame({'SharkSpecies': myKeys,
                       'Attacks': attacklist,
                       'Fatalities': fatalist})
    

    df1= df.drop(df[df['SharkSpecies'] == 'unknown'].index)
    
    # Melt the DataFrame to stack the values
    df_melted = pd.melt(df1, id_vars='SharkSpecies', var_name='AttackType', value_name='Value')

    # Create a custom color palette
    colors = sns.color_palette(['lightgreen', 'red'])
    
    # Create a bar chart using Seaborn
    sns.set_style('whitegrid')
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='SharkSpecies', y='Value', hue='AttackType', data=df_melted, palette=colors)
    plt.setp(ax.get_xticklabels(), rotation=90)
    plt.xlabel('Shark Species', fontsize=14)
    plt.ylabel('Number of Attacks', fontsize=14)
    plt.legend(title='Attack Type', title_fontsize=12, fontsize=10)
   
    # Add annotations to each bar with font properties
    for i in ax.containers:
        ax.bar_label(i, label_type='edge', labels=i.datavalues.astype(int), color='grey')
    ax.set_title(t)
    plt.show()
    
def makeLocation(df,col1,col2,col3):
    #concatenates my Location, Area, Country in that order so geolocate has the best chance of locating
    df[col1] = df[col1].fillna('')
    df[col2] = df[col2].fillna('')
    df[col3] = df[col3].fillna('')
    df['full_location'] = ''
    
    for index, row in df.iterrows():
        # concatenate values of the three columns
        location = row[col1]
        area = row[col2]
        country = row[col3]
        
        fullLocation = location+', '+area+', '+country
        fullLocation = fullLocation.strip(',')
        fullLocation = fullLocation.replace(', , ',', ')
        
        df.at[index, 'full_location'] = fullLocation
              
    return df
    
    
def getlatLong(df):
    location = df['full_location']
    df['latitude']=0.0
    df['longitude']=0.0
    df['pos'] = ""
    for index, row in tqdm(df.iterrows(), total=len(df)):
        location = row['full_location']
        pos=LocationSpot(location)
        df.at[index, 'latitude'] = pos.lat()
        df.at[index, 'longitude'] = pos.lng()
        df.at[index, 'pos'] = str(pos.coordinates())
        time.sleep(0.01)
    
    return df
    
def replace_invalid_age(age):
    try:
        age_int = int(age)
        return age_int
    except ValueError:
        age = random.randint(15, 40)
        return age

  
    
def identifiedvsunidentified(dictionary):
    mySet = {"Identified Sharks", "Unidentified Sharks"}
    pairDict = {key: 0 for key in mySet}
    for key in dictionary.keys():
        if key == "unknown":
            pairDict['Unidentified Sharks'] = dictionary['unknown']
        else:
            pairDict['Identified Sharks'] += dictionary[key]
    
    return pairDict
    
    

def pieFromDict(dictionary, excludekeys=[], t="Untitled"):
    
    new_dict = {k: v for k, v in dictionary.items() if k not in excludekeys}
    
    labels = list(new_dict.keys())
    sizes = list(new_dict.values())

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    ax1.set_title(t)
    
    plt.legend(labels, bbox_to_anchor=(1.6, 1.0))
    plt.show()
    
def pie2FromDict(dictionary, excludekeys=[], t="Untitled"):
    
    new_dict = {k: v for k, v in dictionary.items() if k not in excludekeys}
    
    labels = list(new_dict.keys())
    values = list(new_dict.values())

    fig = px.pie(values=values, names=labels, title=t)
    fig.show()    
    
def saveMyDictionary(dictionary, name):    
    filename = str(name) + ".pkl"
    with open(filename, 'wb') as fp:
        pickle.dump(dictionary, fp, protocol=pickle.HIGHEST_PROTOCOL)
        print("file:", filename, "succesfully created.")
    
    
def loadMyDictionary(filename):
    with open(filename, 'rb') as fp:
        loaded_dict = pickle.load(fp)
    return loaded_dict



def fixTimetext(text):
    
    text = str(text)
    text = text.lower()
    match = re.search(r"\d+(?=h)", text)
    dawn = ['06j00','dawn','daybreak']
    morning = ['a.m.','am.','morning','10jh45','10j30']
    midday = ['midday', 'lunchtime','noon']
    afternoon = ['afternoon', 'early afternoon', 'late afternoon', 'after noon','after lunch']
    dusk = ['sunset','dusk', 'after dusk', 'nightfall', 'evening']
    night = ['night', 'midnight', '500', 'p.m.','dark']
    
    text = text.lower()

    
    try:
        if match:
            digits = int(match.group())
            if digits <= 6:
                return 'Dawn'
            elif 6 < digits <= 11:
                return 'Morning'
            elif 11 < digits <= 13:
                return 'Midday'
            elif 13 < digits <= 17:
                return 'Afternoon'
            elif 17< digits <= 19:
                return 'Dusk'
            else:
                return 'Night'
            
            
        elif any(re.search(r'\b{}\b'.format(w), text) for w in dawn):
            return 'Dawn'
        elif any(re.search(r'\b{}\b'.format(w), text) for w in morning):
            return 'Morning'
        elif any(re.search(r'\b{}\b'.format(w), text) for w in midday):
            return 'Midday'
        elif any(re.search(r'\b{}\b'.format(w), text) for w in afternoon):
            return 'Afternoon'
        elif any(re.search(r'\b{}\b'.format(w), text) for w in dusk):
            return 'Dusk'
        elif any(re.search(r'\b{}\b'.format(w), text) for w in night):
            return 'Night'
        else:
            return None
            
    except:
        print("the value could not be fixed!")

        
def randomChoice():
    times = ['Afternoon', 'Dusk', 'Night', 'Dawn']
    return random.choice(times)
        
        
def topAttackedCountries(df,n):
   
    topX = df['Country'].value_counts().nlargest(n)

    colors = sns.color_palette('husl', len(topX))

    plt.figure(figsize=(12,6))
    ax = sns.countplot(data=df, x='Country', order=topX.index, palette=colors)
    plt.xticks(rotation=90)
    #plt.title('Top n Countries with Highest Shark Attacks Frequency')
    plt.xlabel('Country')
    plt.ylabel('Frequency')

    ax.set_title("Top n Countries with Highest Shark Attacks Frequency")
    # Add annotations to each bar with font properties
    for i in ax.containers:
        ax.bar_label(i, label_type='edge', labels=i.datavalues.astype(int), color='teal', size=7)
  
    plt.show()

def topFatalityCountries(df,n):
    
    fdf = df.loc[dfAttacks.Fatal == 'Y']
    topX = fdf['Country'].value_counts().nlargest(n)

    colors = sns.color_palette('husl', len(topX))

    plt.figure(figsize=(12,6))
    ax = sns.countplot(data=fdf, x='Country', order=topX.index, palette=colors)
    ax.set_title("Top n Countries with Highest Shark Attack Fatalities")
    plt.xticks(rotation=90, size=8)
    #plt.title("Top n Countries with Highest Shark Attack Fatalities")
    plt.xlabel('Country')
    plt.ylabel('Frequency')

    
    # Add annotations to each bar with font properties
    for i in ax.containers:
        ax.bar_label(i, label_type='edge', labels=i.datavalues.astype(int), color='teal', size=7)
  
    plt.show()

def topProbabilityofFatalities(df, x_col, y_col, n):
    sorted_df = df.sort_values(y_col, ascending=False)
    top_n = sorted_df[:n]
    ax =sns.barplot(x=x_col, y=y_col, data=top_n)
    plt.ylim(0, 0.07)
    ax.set_title('Probability of fatality if attacked by a shark or by shark per Country')
    plt.xticks(rotation=90, size = 8)
    plt.xlabel('Country', size = 10)
    plt.ylabel('Probability')

    plt.show()

def sharkReport(knowns,regional):
  
    if len(knowns) == 0:
        print("Could not find any IDENTIFIED sharks that have actually been IMPLICATED in attacks within 20 km of your location.")
        print("These are however risky sharks living in the waters of your location that could pose a threat if you were attacked.\n")
        print(list(set(regional)))
        givenSharkProbabilities(dfSharks, 'SharkSpecies', 'FatalityProbability', len(regional),regional)
      
    else:
        for i in range (0,len(knowns)):
            if knowns[i] == 'tiger': 
                knowns[i] = 'tiger shark' 

        print("Your location is within proximity (20km) of recorded attacks made by IDENTIFIED sharks according to ISAF")
        print("These are the identified sharks:\n")
        print(knowns)
        givenSharkProbabilities(dfSharks, 'SharkSpecies', 'FatalityProbability', len(knowns),knowns)     
    

    
def givenSharkProbabilities(df, x_col, y_col, n, sharks):
    # Filter the dataframe to only include rows where the shark is in the provided list
    filtered_df = df[df['SharkSpecies'].isin(sharks)]
    
    # Sort the filtered dataframe by the y_col column
    sorted_df = filtered_df.sort_values(y_col, ascending=False)
    
    # Select the top n rows from the sorted dataframe
    top_n = sorted_df[:n]

    # Create the bar plot using seaborn
    ax = sns.barplot(x=x_col, 
                     y=y_col, 
                     data=top_n, 
                     color='black', 
                     saturation=1, 
                     alpha=0.2, 
                     edgecolor='black',
                     linewidth=2)
    
    ax.bar(top_n[x_col], top_n[y_col], width=0.3)
    
    
    
    # Add annotations to the bars
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.4f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 10), 
                    textcoords='offset points',
                    fontsize=8, 
                    fontweight='normal',
                    color='teal'
                   )

    
    
    
    # Set the y-axis limit and plot title
    plt.ylim(0, 0.1)
    ax.set_title('likeliness of fatality if attacked by the following sharks')
    
    # Rotate the x-axis labels and set the x and y-axis labels sizes
    plt.xticks(rotation=90, size=8)
    plt.xlabel('Sharks', size=10)
    plt.ylabel('Probability')
    
    # Display the plot
    plt.show()    
    
    
    
    
    
    
    
    
def topSizedSharks(df, x_col, y_col, n):
    sorted_df = df.sort_values(y_col, ascending=False)
    top_n = sorted_df[:n]
    ax =sns.barplot(x=x_col, y=y_col, data=top_n)
    #plt.ylim(0, 0.07)
    ax.set_title('Average Shark Size in Feet')
    plt.xticks(rotation=90, size = 8)
    plt.xlabel('Shark', size = 10)
    plt.ylabel('Size')

    plt.show()    
    
def chartRisk(df, x_col, y_col, n):
    sorted_df = df.sort_values(y_col, ascending=False)
    top_n = sorted_df[:n]
    ax =sns.barplot(x=x_col, y=y_col, data=top_n)
    ax.set_title('Standardised Risk Scores')
    plt.xticks(rotation=90, size = 8)
    plt.xlabel(x_col, size = 10)
    plt.ylabel(y_col)

def classifyWaterConditions (df):
    df['Zone'] = ""
    myZone = ""
    for index, row in df.iterrows():
       
        latitude = row['latitude']
        longitude = row['longitude']
        
        if 0 < latitude <= 23:
            myZone = "TROPIC OF CANCER"
        elif -23.5 < latitude <= 0:
            myZone = "TROPIC OF CAPRICORN"
        elif 23.5 < latitude <= 55:
            myZone = "TEMPERATE"
        elif -23.5 <= latitude < -66:
            myZone = "TEMPERATE"
        else:
            myZone = "POLAR"
        
        df.at[index, 'Zone'] = myZone
              
    return df
        
def makePoint(index, df):
    row = df.iloc[index]
    latitude = row['latitude']
    longitude = row['longitude']
    name = row['full_location']
    country = row['Country']
    lp = LocationSpot.from_df_row(latitude,longitude,name,country,index)
    return lp
    
    

    
    
def loadSharksByRegion(filename):    
    data_dict = {}

    # Open the text file for reading
    with open(filename, 'r') as file:

        # Loop over each line in the file
        for line in file:

            # Split the line into a country and a list of sharks
            country, sharks = line.strip().split(',',1)
            country = country.upper()
            # Convert the list of sharks to a Python list
            shark_list = [s.strip()[1:-1] for s in sharks.split('[')[-1].split(']')[0].split(',')]
            # Add the country and the list of sharks to the dictionary
            data_dict[country] = shark_list

    return data_dict

    
##################################################MYLOCATIONCLASS###################################################    
        
class LocationSpot():
        
    gmaps = googlemaps.Client(key="AIzaSyBUN8KXWnNAoS36tpdZikfBWaNIEKfJ3-8") 
    
    myIndex = 0
    myName = ""
    myLatitude = 0
    myLongitude = 0
    myCoordinates = ()
    myMap = None
    myNeighbours = []
    myCountry = None
    myZone = ""
    myTemp = (0.0,0.0)
    nd = 0.0
    
    
    
    def __init__(self, loc): 
        self.myName = str(loc)
        geocode_result = gmaps.geocode(str(loc))
        try:
            location = geocode_result[0]['geometry']['location'] 
            self.myLatitude = location['lat']
            self.myLongitude = location['lng']
            self.myCoordinates = (self.myLatitude,self.myLongitude)
            
            #Get the Country
            
            for component in geocode_result[0]['address_components']:
                if "country" in component['types']:
                    self.myCountry = component['long_name']
                    break
            
            
            popuptxt = "Location:{}\nLongitude:{}\nLatitude:{}".format(self.myName, self.myLongitude, self.myLatitude)
            map = folium.Map(location=[self.myLatitude,self.myLongitude], zoom_start=15) 
            folium.Marker([self.myLatitude,self.myLongitude],popup=popuptxt).add_to(map)
            self.myMap = map
            latitude = self.myLatitude
            zone = ""
            if 0 < latitude <= 23:
                zone = "TROPIC OF CANCER"
                self.myTemp = (22,26)
            elif -23.5 < latitude <= 0:
                zone = "TROPIC OF CAPRICORN"
                self.myTemp = (18,30)
            elif 23.5 < latitude <= 55:
                zone = "TEMPERATE"
                self.myTemp = (10,20)
            elif -23.5 <= latitude < -66:
                zone = "TEMPERATE"
                self.myTemp = (5,18)
            else:
                zone = "POLAR"
                self.myTemp = (-2,5)
            self.myZone = zone
        
        except IndexError:
                print("Geocoding failed: no results found.")
                      
            
    @classmethod
    def from_df_row(cls, latitude, longitude, name, country,idx):
        #geocode_result = gmaps.reverse_geocode((latitude, longitude))
        obj = cls.__new__(cls)
        obj.myLatitude= latitude
        obj.myLongitude = longitude
        obj.myName = name
        obj.myCoordinates = (latitude, longitude)
        obj.myCountry = country
        obj.myNeighbours = []
        obj.myIndex = idx
        
        popuptxt = "Location:{}\nLongitude:{}\nLatitude:{}".format(name, longitude, latitude)
        map = folium.Map(location=[latitude,longitude], zoom_start=15) 
        folium.Marker([latitude,longitude],popup=popuptxt).add_to(map)
        obj.myMap = map
    
        zone = ""
        if 0 < latitude <= 23:
            zone = "TROPIC OF CANCER"
            obj.myTemp = (22,26)
        elif -23.5 < latitude <= 0:
            zone = "TROPIC OF CAPRICORN"
            obj.myTemp = (18,30)
        elif 23.5 < latitude <= 55:
            zone = "TEMPERATE"
            obj.myTemp = (10,20)
        elif -23.5 <= latitude < -66:
            zone = "TEMPERATE"
            self.myTemp = (5,18)
        else:
            zone = "POLAR"
            obj.myTemp = (-2,5)
        
        obj.myZone = zone
        
        return obj

    def setnd(self, num):
        self.nd=num
        
    def getnd(self):
        return self.nd
            
        
    def name(self):
        return self.myName
    
    def my_map(self):
        return self.myMap
    
    def lat(self):
        return self.myLatitude
    
    def lng(self):
        return self.myLongitude
    
    def neighbors(self):
        return self.myNeighbours
    
    
    def coordinates(self):
        return self.myCoordinates
    
    def proximityKM(self, other):
        # Define the coordinates of the two points
        point1 = self.myCoordinates
        point2 = other
        # Compute the distance between the two points
        dist = distance(point1, point2).km
        return dist
    
    def amClose(self,other):
        if self.proximityKM(other) < 30.00:
            #print("your within 20k")
            return True
        else:
            #print("not within 20k")
            return False
    
    def amInMediterranean(self):
        if (30 <= self.myLatitude <= 46) and (-6 <= self.myLongitude <= 36 ):
            return True
        else:
            return False
        
    def amInRedSea(self):
        if (12 <= self.myLatitude <= 28) and (33 <= self.myLongitude <= 43 ):
            return True
        else:
            return False
    
           
    def amInEnglishChannel(self):
        if (48 <= self.myLatitude <= 51.28) and (-5 <= self.myLongitude <= 1.55 ):
            return True
        else:
            return False 
    
    def amInIonianSea(self):
        if (36.5 <= self.myLatitude <= 39) and (16 <= self.myLongitude <= 20 ):
            return True
        else:
            return False    
    
    def amInPersianGulf(self):
        if (24.8 <= self.myLatitude <= 30.5) and (48.6 <= self.myLongitude <= 56.3 ):
            return True
        else:
            return False 
        
        
    def amInCarribeanSea(self):
        if (10 <= self.myLatitude <= 25) and (-85<= self.myLongitude <= -60 ):
            return True
        else:
            return False 
    
    
    def amInGulfOfMexico(self):
        if (20 <= self.myLatitude <= 30) and (-98<= self.myLongitude <= -88 ):
            return True
        else:
            return False 
    
    def amInArabianGulf(self):
        if (23.5 <= self.myLatitude <= 30.5) and (48.5 <= self.myLongitude <= 56.6 ):
            return True
        else:
            return False 
    
    def getZone(self):
        return self.myZone
    
    def getTempRange(self):
        return self.myTemp
        
        
    def country(self):
        if self.myCountry is not None:
            return self.myCountry
        else:
            return "unknown"
            
    def __str__(self):
        desc = ""
        desc+="Location:{}\nLongitude:{}\nLatitude:{}".format(self.myName, self.myLongitude, self.myLatitude)
        return desc
     
        ##Spacial DISTANCE based NearestNeightbour Analysis    
    
    def getNearestKDT(self, n, df):
        
        coordinates = df['pos'].tolist()

        #using my spatial distance function above
        tree = KDTree(coordinates, distance=proximity_in_KM)

        # Find the 5 nearest neighbors to me
        myPos = self.myCoordinates
        distances, indices = tree.query(myPos, k=n)

        # Get the original indices from the DataFrame
        original_indices = df.index.tolist()

        # Create a DataFrame containing the nearest neighbors and their distances
        nearest_neighbors = df.iloc[indices].copy()
        nearest_neighbors['distance'] = distances
        nearest_neighbors['original_index'] = [original_indices[i] for i in indices]
        
        myNeighborsIndices = nearest_neighbors['original_index'].tolist()
        
        for ind in myNeighborsIndices:
            p = makePoint(ind, nearest_neighbors)
            self.myNeighbours.append(p)
            
            
    def getNearestBallTree(self, n, df):
        # Convert the "pos" column to a list of tuples and convert to radians
        coordinates = [(np.radians(coord[0]), np.radians(coord[1])) for coord in df['pos'].apply(ast.literal_eval).tolist()]
        print (coordinates[0])


        myPos = (np.radians(self.myCoordinates[0]),np.radians(self.myCoordinates[1]))
        #print (m, myPos)
        # Find the 5 nearest neighbors to me
        distances, indices = BallTree(coordinates, metric=haversine_distances).query(myPos.reshape(1, -1), k=n)
        indices = indices[0]
        distances = distances[0]

        # Get the original indices from the DataFrame
        original_indices = df.index.tolist()

        # Create a DataFrame containing the nearest neighbors and their distances
        nearest_neighbors = df.iloc[indices].copy()
        nearest_neighbors['distance'] = distances
        nearest_neighbors['original_index'] = [original_indices[i] for i in indices]

        myNeighborsIndices = nearest_neighbors['original_index'].tolist()

        for ind in myNeighborsIndices:
            p = makePoint(ind, nearest_neighbors)
            self.myNeighbours.append(p)
            
    def myNearestNeighbours(self,n,df):
        
        
        original_indices = df.index.tolist()    
        myPos = np.array([np.deg2rad(self.myCoordinates[0]), np.deg2rad(self.myCoordinates[1])])

        myTree = BallTree(df[["latitude_rad", "longitude_rad"]].values, metric='haversine')
        
        distances, indices = myTree.query(myPos.reshape(1, -1), k = n)
        
        indices = indices[0]
        distances = (distances[0] * 6371)
        
        nearest_neighbors = df.iloc[indices].copy()
        nearest_neighbors['distance'] = distances
        nearest_neighbors['index'] = [original_indices[i] for i in indices]
        
        myNeighborsIndices = nearest_neighbors['index'].tolist()
        
        self.myNeighbours = []
        for index, row in nearest_neighbors.iterrows():
            p = makePoint(index, df)
            p.setnd(row['distance'])
            self.myNeighbours.append(p)
        
        return nearest_neighbors
    
    
    def mapMeAndMyNeighbours(self, n, df):

        mypopup = "Location:{}\nLongitude:{}\nLatitude:{}".format(self.name(), self.lng(), self.lat())
        me = self.myCoordinates
        map = folium.Map(location=me, popup= mypopup, zoom_start=2)

        
        marker_group = folium.FeatureGroup()

        dfn = self.myNearestNeighbours(n,df)

        for neighbour in self.neighbors():
            npopup = "Location:{}\nLongitude:{}\nLatitude:{}\nDistance-KM:{}".format(neighbour.name(), neighbour.lng(), neighbour.lat(),neighbour.getnd())
            ncoord = neighbour.coordinates()
            folium.Marker(ncoord,popup=npopup).add_to(marker_group)
            folium.Circle(location=ncoord, radius=50000, color='orange', fill=True, fill_opacity=0.1,stroke_opacity=0.5).add_to(marker_group)

        # Custom icon for center point
        icon = folium.Icon(icon='times-circle', color='red', prefix='fa')
        folium.Marker(location=me, popup=mypopup, icon=icon).add_to(marker_group)

        marker_group.add_to(map)
        map.fit_bounds(marker_group.get_bounds())


        return map


    def mapNeighbours(self,df):
        mypopup = "Location:{}\nLongitude:{}\nLatitude:{}".format(self.name(), self.lng(), self.lat())
        me = self.myCoordinates
        if not hasattr(self, 'map'):  # Create map if it does not exist
            self.map = folium.Map(location=me, popup= mypopup, zoom_start=10)
        else:  # Clear existing map
            self.map.save('map.html')
            self.map = folium.Map(location=me, popup= mypopup, zoom_start=10)
        marker_group = folium.FeatureGroup()

        dfn = self.myNearestNeighbours(5,df)

        for neighbour in self.neighbors():
            npopup = "Location:{}\nLongitude:{}\nLatitude:{}\nDistance-KM:{}".format(neighbour.name(), neighbour.lng(), neighbour.lat(),neighbour.getnd())
            ncoord = neighbour.coordinates()
            folium.Marker(ncoord,popup=npopup,icon=folium.Icon(color='black', icon='fish-fins', prefix='fa')).add_to(marker_group)
            folium.Circle(location=ncoord, radius=20000, color='orange', fill=True, fill_opacity=0.2,stroke_opacity=0.5).add_to(marker_group)

        # Add red circle marker for self
        folium.CircleMarker(
            location=me, radius=10, fill_color='red', color='red'
        ).add_to(marker_group)

        marker_group.add_to(self.map)
        self.map.fit_bounds(marker_group.get_bounds())

        return self.map
    
    
    
    def mapN(self, df):
        # Create a new map object
       
        mypopup = "Location:{}\nLongitude:{}\nLatitude:{}".format(self.name(), self.lng(), self.lat())
        me = self.myCoordinates
        map = folium.Map(location=me, popup= mypopup, zoom_start=10)


        
        # Add a red marker for yourself
        folium.Marker(location=self.myCoordinates,popup=mypopup, icon=folium.Icon(color='red', icon='fa-user', prefix='fa')).add_to(map)

        
        
        self.myNeighbours.clear()
        dfn = self.myNearestNeighbours(10,df)
        
        
        # Add circles for each neighbour
        for neighbour in self.neighbors():
            npopup = "Location:{}\nLongitude:{}\nLatitude:{}\nDistance-KM:{}".format(neighbour.name(), neighbour.lng(), neighbour.lat(), round(neighbour.getnd(),2))
            ncoord = neighbour.coordinates()
            folium.Circle(location=ncoord, radius=20000, color='orange', fill=True, fill_opacity=0.1,stroke_opacity=0.5).add_to(map)
            folium.Marker(location=ncoord, popup=npopup, icon=folium.Icon(color='black', icon='fish-fins', prefix='fa')).add_to(map)

        
        folium.Marker(location=self.myCoordinates,popup=mypopup, icon=folium.Icon(color='red', icon='fa-user', prefix='fa')).add_to(map)
        
        # Return the map
        return map
    
    
    def getMySharks(self,dfAttacks=dfAttacks,sharkRisk=sharkRisk):
        knownList = []
        regionList = []
        mySea = ""
        myZone = ""
        theCountry = ""
        
        
        def convert_to_list(input):
            if isinstance(input, list):
                return input
            elif isinstance(input, str):
                return ast.literal_eval(input)
            else:
                raise ValueError("Input must be a list or a string representation of a list.")
        
        
        for neighbour in self.neighbors():
            if self.amClose(neighbour.myCoordinates):
                sharks = dfAttacks.loc[neighbour.myIndex, 'SharkSpecies']
                #print(sharks)
                if sharks != 'unknown':
                    if "," in sharks:
                        sharktokens = set(sharks.split(','))
                        for token in sharktokens:
                            knownList.append(token.strip().lower())
                            
                    else: knownList.append(sharks.strip().lower())

        
        if len(knownList) == 0:
            theCountry = unidecode(self.country().upper())
            myZone= self.getZone()
            
            print(theCountry)
            if theCountry == "UNKNOWN":
                if self.amInArabianGulf():
                    mySea = "ARABIAN GULF"
                    regionList=getOceanSharks(mySea) 
                elif self.amInCarribeanSea():
                    mySea = "CARIBBEAN SEA"
                    regionList=getOceanSharks(mySea) 
                elif self.amInEnglishChannel():
                    mySea = "ENGLISH CHANNEL"
                    regionList=getOceanSharks(mySea) 
                elif self.amInGulfOfMexico():
                    mySea = "GULF OF MEXICO"
                    regionList=getOceanSharks(mySea) 
                elif self.amInIonianSea():
                    mySea = "IONIAN SEA"
                    regionList=getOceanSharks(mySea) 
                elif self.amInMediterranean():
                    mySea = "MEDITERRANEAN SEA"
                    regionList=getOceanSharks(mySea) 
                elif self.amInPersianGulf():
                    mySea = "PERSIAN GULF"
                    regionList=getOceanSharks(mySea) 
                elif self.amInRedSea():
                    mySea = "RED SEA"
                    regionList=getOceanSharks(mySea) 
            elif  theCountry in sharkRisk['Country'].values: 
                #print("Country Found in SharkList")
                regionList = sharkRisk.loc[sharkRisk.Country == theCountry, 'Sharks'].values[0]
            else:
                #print("Country not found in sharklist and sea unknown")
                regionList = getOceanSharks(myZone)
                #print("I finally have", regionList)
        
    
        
        regionList = convert_to_list(regionList)
        
        return [list(set(knownList)), regionList]
    
  
    def display(self):
        dfAttacks=pd.read_csv("dfAttacksX.csv")
        self.myNearestNeighbours(10,dfAttacks)
        self.mapN(dfAttacks)
 #sharkList = list(itertools.chain.from_iterable(sharkList))          
###########################################################################################################################
#All the reads of csvs and dicts and Compilation of Stats as assets

##Load and Test Dictionaries 
allSharks = loadMyDictionary("allSharkCounts.pkl")
sharksbyregion= loadMyDictionary("sharksbyRegion.pkl")
identifiedvsunidentified = loadMyDictionary("identifiedvsunidentified.pkl")
sharkSet={'lemon', 'grey reef', 'blacktip', 'unknown', 'tiger', 'porbeagle', 'bull', 'spinner', 'silky', 'sandbar', 'nurse', 'oceanic whitetip', 'mako', 'blue', 'sand tiger', 'white shark', 'bronze whaler', 'sevengill', 'zambezi', 'thresher', 'whitetip reef', 'hammerhead', 'wobbegong', 'grey nurse', 'goblin', 'dusky', 'galapagos', 'raggedtooth'}
#Load dfAttacks
dfAttacks=pd.read_csv("dfAttacksX.csv")
dfCountry = pd.read_csv("countryStatsX.csv")
dfSharks = pd.read_csv("SharkStatsX.csv")
fdf = pd.read_csv("fatalities.csv")
sharkRisk = pd.read_csv('sharkBoost.csv')
dfFatality = dfAttacks.loc[dfAttacks.Fatal == "Y"]
dfNONFatality = dfAttacks.loc[dfAttacks.Fatal == "N"]
fatalities = sortedUniqueCounts(dfFatality, 'SharkSpecies')
nonfatalities = sortedUniqueCounts(dfNONFatality, 'SharkSpecies')
fatalitiesDict = consolidate(sharkSet,fatalities)
fatalDict = sortedUniqueCounts(dfAttacks, 'Fatal')
CountryAttacksDict = sortedUniqueCounts(dfAttacks, 'Country')
CountryFatalitiesDict = sortedUniqueCounts(fdf, 'Country')
#dfCountry = fatalityRatioByCountry(CountryAttacksDict,CountryFatalitiesDict)
#dfCountry.to_csv("countryStats2.csv", index=False)



def getOceanRisk(sea, df=sharkRisk):
    #If in the middle of the ocean and you get attacked.. fatality risk HAS to be boosted    
    myValue = df.loc[df.Country == sea, "risk"].values[0]
    myValue += 5
    return myValue

def getOceanSharks(sea, df=sharkRisk):
    #If in the middle of the ocean and you get attacked.. fatality risk HAS to be boosted    
    print(sea)
    myValue = df.loc[df.Country == sea, "Sharks"].values[0]
    return  ast.literal_eval(myValue)


def getCountryRisk(country, df=dfCountry):
    country = unidecode(country)
    myValue = df.loc[df.Country == country, "risk"].values   
    return myValue[0]


def makeXrow(myInputs):
   
    rowDF= pd.DataFrame(columns=['CaseNumber', 'Date', 'Year', 'Type', 'Country', 'Area', 'Location', 'Activity', 'Name', 'Gender', 'Injury', 'Time', 'Species', 'SharkSpecies', 'full_location', 'latitude', 'longitude', 'pos', 'Timeslot', 'Zone', 'latitude_rad', 'longitude_rad', 'risk'])
    drop_features = ['CaseNumber', 'Date', 'Year', 'Type', 'Area', 'Location','Name', 'Injury','Time','Species', 'SharkSpecies', 'full_location', 'pos', 'latitude_rad','longitude_rad']
    mySea=""   
    for col in drop_features:
        rowDF.loc[0,[col]]=""

    rowDF.loc[0, ['Gender', 'Timeslot', 'Location', 'Activity']] = myInputs[:4]        
    rowDF.loc[0,['full_location']] = myInputs[2]
    myPosition = LocationSpot(str(myInputs[2]))
    rowDF.loc[0,['Zone']]=myPosition.getZone()
    myCountry = myPosition.country()
    if myCountry == "United Kingdom":
        myCountry == "England"
    elif myCountry == "United States":
        myCountry = "USA"
    rowDF.loc[0,['Country']]=unidecode(myCountry.upper())

    myRisk = 0.0
    if myPosition.amInArabianGulf():mySea = "ARABIAN GULF"
    elif myPosition.amInCarribeanSea():mySea = "CARIBBEAN SEA"
    elif myPosition.amInEnglishChannel():mySea = "ENGLISH CHANNEL"
    elif myPosition.amInGulfOfMexico():mySea = "GULF OF MEXICO"
    elif myPosition.amInIonianSea():mySea = "IONIAN SEA"
    elif myPosition.amInMediterranean():mySea = "MEDITERRANEAN SEA"
    elif myPosition.amInPersianGulf():mySea = "PERSIAN GULF"
    elif myPosition.amInRedSea():mySea = "RED SEA"
        
    if mySea != "":
        print("sea was:", mySea)
        myRisk = getOceanRisk(mySea)
        #print("my risk in the ocean is:", myRisk)
    else:
        print("Location country: ", myCountry)
        myRisk = getCountryRisk(myCountry.upper())
        print("Risk associated to this Country:", myRisk)
        
    rowDF.loc[0,['risk']] = myRisk
    rowDF.loc[0,['latitude']] = myPosition.lat()
    rowDF.loc[0,['longitude']] = myPosition.lng()
    posndf = (myPosition, rowDF)
    return posndf