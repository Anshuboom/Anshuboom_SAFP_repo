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
import pickle
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

##############Load_the_trained_pipeline######################################

with open('./Model/trained_pipeline_LogisticRegression.pkl', 'rb') as f:
    pipeline = pickle.load(f)

dfAttacks=pd.read_csv("./Data/dfAttacksX.csv")

#need to convert fatal into 1,0
dfM = dfAttacks.copy()
dfM['Fatal'] = dfM['Fatal'].apply(lambda x: 1 if x =='Y' else 0)
y_full = dfM['Fatal']
dfM = dfM.drop(columns='Fatal', axis=1)

##############UseitToPredict##################################################

y_predfull = pipeline.predict(dfM)

dfM['PredictedY'] = y_predfull


#############SaveResults######################################################

dfM.to_csv('./Data/resultsTable.csv', index=False)


