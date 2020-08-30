import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk
nltk.download()
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
import re

dataset=pd.read_csv('train.csv')

dataset['tweet'][0]

processed_tweet=[]

for i in range(31962):
    tweet=re.sub('@',' ',dataset['tweet'][i])#sub function of re library is used to subsitute the characters given first with the characters given second
    tweet=re.sub('[^a-zA-Z#]',' ',dataset['tweet'][i])#[^a-zA-Z#] means all characters except a-z A-Z and # are replaced by space
    tweet=tweet.lower()
    tweet=tweet.split()#used to get the words of the string tweet
    tweet=[ps.stem(token) for token in tweet if not token in set(stopwords.words('english'))]#means all the words that are in the string tweet and are not in the stopwords are kept and then stemming is applied on the remaining words
    tweet=' '.join(tweet)#joins the words of the list  tweet with a space in between the words
    processed_tweet.append(tweet)
    
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=3000)#like we did in the copy example that after all the editing the number of words in all the final strings added up to make 5000 and then we made the sparse matrix .if we increase the count of max_feature the analysis will be better
X=cv.fit_transform(processed_tweet)
X=X.toarray()
y=dataset['label'].values
print(cv.get_feature_names())

from sklearn.linear_model import LogisticRegression#working of LR algo
log_reg=LogisticRegression()
log_reg.fit(X,y)

log_reg.score(X,y)

y_pred=log_reg.predict(X)#making prediction s using prediction method of class Logistics Regression

from sklearn.metrics import confusion_matrix#creation of confusion matrix
cm=confusion_matrix(y,y_pred)






    
    


