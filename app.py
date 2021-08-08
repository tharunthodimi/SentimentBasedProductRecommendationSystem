# -*- coding: utf-8 -*-
"""
Created on Tue Auguest 08 22:38:27 2021

@author: Tharun Tej reddy Thodimi
"""
# import Libraries 
import re
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import sklearn
from nltk.stem import WordNetLemmatizer
lm=WordNetLemmatizer()
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
tf=TfidfVectorizer(max_features=5000,ngram_range=(1,3))
#Import NLP Libraries
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# Initialise the Flask and load pickle files.
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
tf=pickle.load(open('transform.pkl','rb'))
FinalRating=pickle.load(open('FinalRating.pkl','rb'))
recommend_train=pickle.load(open('traindata.pkl','rb'))

# Initialise empty string for userid
Inputfromuser = None  

@app.route('/')
def home():
    return render_template('index.html') #Render html page for UI


@app.route('/predict',methods=['POST'])  #perform Post call with user provided details
def predict():
    Inputfromuser=request.form['Username']  #Fetch the username from Frontend
    Inputfromuser=Inputfromuser.lower()     # Lower the userid
    Inputfromuser=Inputfromuser  #assign value to global varibale 
    prediction=pickle.loads(pickle.dumps(recommend))(Inputfromuser)
    if(type(prediction)==str): #compare if output is a string or a dataframe
        return render_template('index.html', prediction_text='{}'.format(prediction))
    else: #Output obtained is a dataframe
        return (render_template('index.html', prediction_text='Products Recommended are: \n \n \n 1.{}  2.{} 3.{} 4.{} 5.{}'.format(prediction[0],prediction[1],prediction[2],prediction[3],prediction[4])))

# Recommendation system
def recommend(user_input):
    try:
        userRecommendations = FinalRating.loc[user_input].sort_values(ascending=False)[0:20] # Filter top 20  recommendations
        userRecommendationsResult = {'product': userRecommendations.index, 'recomvalue': userRecommendations.values}
        newdf=pd.DataFrame(userRecommendationsResult,index=range(0,20))
        positiverating=[]
        for i in range(20):
            positiverating.append(sum(recommend_train[recommend_train['name'] == newdf['product'][i]]['user_sentiment'].values)/len(recommend_train[recommend_train['name'] == newdf['product'][i]]))
        newdf['positiverating']=positiverating
        newdf.sort_values(['positiverating'],ascending=False)
        ## Top 5 Recommendations
       # sort values based on positive rating
        result=newdf.sort_values(['positiverating'],ascending=False)[:5]
        result.reset_index(inplace=True)
        recommended = result['product'].values
        return recommended
    except:
        return "No User Available /Zero Recommendations for valid user."

if __name__ == "__main__":
    app.run(debug=True)
