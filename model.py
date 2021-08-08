# Importing Required Libraries 

import re
import sklearn
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report
from sklearn import model_selection
from sklearn import metrics

# xgboost library for model building
import xgboost as xgboost

# Visualisatiopn libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Stats library
import statsmodels as sm

# pickle for pickling the models built useful for deployment
import pickle

# set options for displaying max columns and rows 
pd.set_option('max_columns',500)
pd.set_option('max_rows',10000000)


#Nlp libraries
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split


ps=PorterStemmer()
lm=WordNetLemmatizer()
nltk.download('wordnet')
nltk.download('stopwords')

# Initialise Countvectoriser and Tf-IDF(TermFrequency -Inverse Document Frequency)
cv=CountVectorizer()
tf=TfidfVectorizer(max_features=5000,ngram_range=(1,3))

# Linear regression model
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()


pickle.dump(tf,open('transform.pkl','wb'))

# Read the sample30 csv file
data=pd.read_csv('sample30.csv')
data.head(3)

# Drop the columns where null values are more than 40 percent in a column
data.drop(['reviews_didPurchase','reviews_userCity','reviews_userProvince'],axis=1,inplace=True)

#Apply Null value Treatment by replacing null values with highest values in that column
data['reviews_username'].replace(np.NaN,data['reviews_username'].value_counts().index[0],inplace=True)
data['manufacturer'].replace(np.NaN,data['manufacturer'].value_counts().index[0],inplace=True)
data['reviews_date'].replace(np.NaN,data['reviews_date'].value_counts().index[0],inplace=True)
data['reviews_title'].replace(np.NaN,data['reviews_title'].value_counts().index[0],inplace=True)
data['reviews_doRecommend'].replace(np.NaN,data['reviews_doRecommend'].value_counts().index[0],inplace=True)

# Drop the columns where null values are still present we have eleiminated one row here provided detailed explanaation in ipynb file
data.dropna(how='any',axis=0,inplace=True)
data.reset_index(drop=True,inplace=True)

#create a new dataframe called data2 for Text Preprocessing
data2=data[:]
data2.head()
# Append Reviews_title and reviews_text and create a new column called reviews
data2['reviews']=data2['reviews_title']+' '+data2['reviews_text']
data2.head()

# Convert categorical values into numerical for model building
data2['usersentiment']=data2['user_sentiment'].replace({'Positive':1,'Negative':0})
data2.head(3)

#Create a new dataframe called data3 for making text data ready for model building
data3=data2[:]
data3.head()

corpus=[] # empty corpus

# Clean the data as shown below
for sentence in data3['reviews']:    
    #1.Lowering the sentences: To eliminate the casesensitive we are lowering all the sentences
    sentence=sentence.lower()    
    
    #2.Use regex to eliminate characters other than alphabets and numbers 
    sentence=re.sub('[^a-zA-Z0-9]',' ',sentence)
    
    #3.Unnecessary Space elimination from the data
    sentence=sentence.split()
    
    #4.Join the sentence with spaces in between and split the data suitable for prerpocessing
    sentence=' '.join(sentence)
    sentence=sentence.split()
    
    #5.Remove stopwords from the data and Apply Lemmatisation on the data
    sentence=[lm.lemmatize(word) for word in sentence if word not in set(stopwords.words('english'))]
    
    #6.Append the preprocessed data into corpus
    corpus.append(' '.join(sentence))
    
# Replace corpus data into the dataframe : data3     
data3['reviews_cleaned']=corpus
data3.head()


# Consider reviews only where word count is less than 150 words
counter=len(data3['reviews_cleaned'])
counter
indexes=[]

# Remove rows where the reviews length is greater than 150 words
for i,j in zip(range(counter+1),data3['reviews_cleaned']):
    if(len(j.split())>150):  #comments greater than 150 words
        if(j == (data3['reviews_cleaned'][i])):
            indexes.append(i)             
              
# Lets drop indexes where reviews containing word count of more than 150
data3.drop(indexes,inplace=True)
data3.reset_index(inplace=True)

# Input/Independent variable
x=data3['reviews_cleaned']
x.head()

# Dependent Variable
y=data3['usersentiment']
y.head()

# TF-IDF: term frequency inverse document frequency
# Convert text data into vector representation
x=tf.fit_transform(x).toarray()
x

#perform train test split of data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42,stratify=y)

# Perform SMOTE Analysis to treat class imbalance since we observed postive percent as 89 and Negative sentiment as 11 percent only
from imblearn.over_sampling import SMOTE
smt = SMOTE(0.5,random_state=42)
x_train_SMOTE, y_train_SMOTE = smt.fit_resample(x_train, y_train)

# Train the linear regression model
lr.fit(x_train_SMOTE,y_train_SMOTE)

# Make Predictions on test data
y_pred=lr.predict(x_test)
y_pred


## We have considered TF-IDF with Logistic Regression model since it provided better results with our data.