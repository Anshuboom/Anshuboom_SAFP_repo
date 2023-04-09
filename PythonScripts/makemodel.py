from cleantext import clean
from pandas.core.common import SettingWithCopyWarning
import warnings
import scipy as sc
import sklearn
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer,TfidfTransformer
from sklearn.preprocessing import (
    OneHotEncoder, 
    OrdinalEncoder, 
    StandardScaler
)

import gensim
import gensim.downloader as model_api
word_vectors = model_api.load("glove-wiki-gigaword-50")
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import lightgbm as lgb
import xgboost as xgb


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    GridSearchCV,
    cross_val_score,
    cross_validate,
    train_test_split,
)
from sklearn.preprocessing import (
    StandardScaler,
    PowerTransformer,
)

#from sklearn.preprocessing import PowerTransformer

from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import FunctionTransformer

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
stop_words = set(ENGLISH_STOP_WORDS)
vocab = word_vectors.key_to_index

import spacy




##JUST_THE_LOGISTIC_REGRESSION_MODEL

##LOAD THE DATA####################

dfAttacks=pd.read_csv("dfAttacksX.csv")

#need to convert fatal into 1,0
dfM = dfAttacks.copy()
dfM['Fatal'] = dfM['Fatal'].apply(lambda x: 1 if x =='Y' else 0)

##PREPROCESSOR AND COLUMN SELECTION AND TRANSFORMATION

nlp = spacy.load('en_core_web_sm')
categorical_features = ['Country', 'Gender','Timeslot','Zone']
numeric_features = ['latitude','longitude','risk']
#numeric_features = ['risk']
drop_features = ['CaseNumber', 'Date', 'Year', 'Type', 'Area', 'Location','Name', 'Injury','Time','Species', 'SharkSpecies', 'full_location', 'pos', 'latitude_rad','longitude_rad']
word_features = ['Activity']


train_df, test_df = train_test_split(dfM, test_size=0.2, random_state=5)

train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)

X_train = train_df.drop(columns='Fatal', axis=1)
y_train = train_df['Fatal']

X_test = test_df.drop(columns='Fatal', axis=1)
y_test = test_df['Fatal']



# Define the column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('Categorical', OneHotEncoder(handle_unknown='ignore'), categorical_features),
#        ('numerical', StandardScaler(), numeric_features),
        ('numerical', PowerTransformer(), numeric_features),
#        ('text', CountVectorizer(tokenizer=lambda text: [tok.text for tok in nlp(text)]), 'Activity'),
        ('text', TfidfVectorizer(tokenizer=tokenize_text), 'Activity'),
        ('droplist', 'drop', drop_features)
        ],
        remainder='passthrough'
    )


# Define the pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])



# Fit the pipeline
pipeline.fit(X_train, y_train)

###SAVE_THE_PIPELINE##################
with open('trained_pipeline_LogisticRegression.pkl', 'wb') as f:
    pickle.dump(pipeline, f)






