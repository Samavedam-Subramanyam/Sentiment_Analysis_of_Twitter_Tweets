import pandas as pd
import tweepy
import datetime
import numpy as np 
import pandas as pd 
import re  
import nltk
from aylienapiclient import textapi
#nltk.download('stopwords')  
from nltk.corpus import stopwords
#import seaborn as sns
#import matplotlib as plt
application_id = "dd9fde46"
application_key = "730b3c3cb9400b7f53a341b705c1d9bc"
#client = textapi.Client(application_id, application_key)
## AYLIEN credentials

def printtweetdata(n, ith_tweet):
    print()
    print(f"Tweet {n}:")
    print(f"Username:{ith_tweet[0]}")
    print(f"Description:{ith_tweet[1]}")
    print(f"Location:{ith_tweet[2]}")
    print(f"Following Count:{ith_tweet[3]}")
    print(f"Follower Count:{ith_tweet[4]}")
    print(f"Total Tweets:{ith_tweet[5]}")
    print(f"Retweet Count:{ith_tweet[6]}")
    print(f"Tweet Text:{ith_tweet[7]}")
    print(f"Hashtags Used:{ith_tweet[8]}")
  
  

def scrape(words, date_since, numtweet):
    end_date = '2022-04-30'
    db = pd.DataFrame(columns=['username', 'description', 'location', 'following','followers', 'totaltweets', 'retweetcount', 'text', 'hashtags','sentiment'])
    print(date_since)
    print(end_date)
    tweets = tweepy.Cursor(api.search, q=words, lang="en",since=date_since, tweet_mode='extended').items(numtweet)
    list_tweets = [tweet for tweet in tweets]
    i = 1  
    for tweet in list_tweets:
        username = tweet.user.screen_name
        description = tweet.user.description
        location = tweet.user.location
        following = tweet.user.friends_count
        followers = tweet.user.followers_count
        totaltweets = tweet.user.statuses_count
        retweetcount = tweet.retweet_count
        #Jon Snow,Slack ,Danny Manning,Wake Forest,#SmallBizSaturday,World War 3,US State department,#PlannedParenthood,#ModiInSingapore,#1in5Muslims,#blackFridayIn3Words
        #hashtags = tweet.entities['#Jon Snow','#Slack' ,'#Danny Manning','#Wake Forest','#SmallBizSaturday','#World War 3','#US State department','#PlannedParenthood','#ModiInSingapore','#1in5Muslims','#blackFridayIn3Words']
        hashtags = tweet.entities['hashtags']
        #tweet['created_at']
        sentiment = 0
        try:
            text = tweet.retweeted_status.full_text
            response = client.Sentiment({'text': text})
            if response['polarity'] == 'neutral':
                score = 0
            if response['polarity'] == 'Positive':
                score = 1
            if response['polarity'] == 'negative':
                score = -1
            sentiment = score
            print('Sentiment')
            print(score)
        except AttributeError:
            text = tweet.full_text
        hashtext = list()
        for j in range(0, len(hashtags)):
            hashtext.append(hashtags[j]['text'])
        ith_tweet = [username, description, location, following,followers, totaltweets, retweetcount, text, hashtext,sentiment]
        db.loc[len(db)] = ith_tweet
        printtweetdata(i, ith_tweet)
        i = i+1
    filename = 'dataset.csv'
    db.to_csv(filename)


import re
def pre_process(text):
    
    # lowercase
    text=text.lower()
    
    #remove tags
    text=re.sub("","",text)
    
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text)
    
    return text
from sklearn.feature_extraction.text import CountVectorizer
import re

def get_stop_words(stop_file_path):
    """load stop words """
    
    with open(stop_file_path, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return frozenset(stop_set)


###################################
def calculate_accuracy(labels_predicted, labels,eval_counter):
    label_nozero=[]
    #print("labels:",labels)
    labels=list(labels)
    for index,label in enumerate(labels):
        if label>0:
            label_nozero.append(index)
    if eval_counter<2:
        print("labels_predicted:",labels_predicted," ;labels_nozero:",label_nozero)
    count = 0
    label_dict = {x: x for x in label_nozero}
    for label_predict in labels_predicted:
        flag = label_dict.get(label_predict, None)
    if flag is not None:
        count = count + 1
    return count / len(labels)

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)

def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]

    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])

    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results
  

    
################################
import matplotlib.pyplot as plt
from nltk import word_tokenize, NaiveBayesClassifier
import pandas as pd
import numpy as np
from tweet_miner import TweetMiner
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import time
import sys

from tkinter import *
from tkinter.simpledialog import askstring
from tkinter.messagebox import showinfo



def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2-time1)*1000))
        return ret
    return wrap


  

@timing
def train_dec_trees_model():
    df_train = pd.read_csv('training.csv')
    df_train.columns = ['sentiment', 'id' ,'date', 'query', 'user', 'tweets']
    word_tokens = [word_tokenize(tweet) for tweet in df_train.tweets]
    len_tokens = []
    for i in range(len(word_tokens)):
        len_tokens.append(len(word_tokens[i]))
    vect = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS, ngram_range=(1, 2), max_features=1000, token_pattern=r'\b[^\d\W][^\d\W]+\b').fit(df_train.tweets)
    X = vect.transform(df_train.tweets)
    tweets_transformed = pd.DataFrame(X.toarray(), columns=vect.get_feature_names())
    tweets_transformed['sentiment'] = df_train['sentiment']
    tweets_transformed['n_words'] = len_tokens
    y = tweets_transformed.sentiment
    X = tweets_transformed.drop('sentiment', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=853)
    dec_trees = DecisionTreeClassifier().fit(X_train, y_train)
    y_predicted = dec_trees.predict(X_test)
    print('Accuracy on the decision trees classifier: ', accuracy_score(y_test, y_predicted))
    print(confusion_matrix(y_test, y_predicted)/len(y_test))
    filename = 'dec_trees_model.sav'
    pickle.dump(dec_trees, open(filename, 'wb'))
    #############################################
    print('Random Forest...');
    from sklearn.ensemble import RandomForestClassifier
    clf=RandomForestClassifier(n_estimators=100)
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    from sklearn import metrics
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    #########################################################
    

def predict_sentiment(word, method, model, data, x):
    print('Predicting sentiment from tweets using ' + method + "...")
    y_predicted = model.predict(x)
    print(y_predicted)
    data['sentiment'] = y_predicted
    print("Average sentiment: " + str(pd.to_numeric(data["sentiment"]).mean()))
    data.to_csv(word +'/' + word+'_analyzed.csv')
    print('Analyzed tweets were stored in '+ word +'/' + word+'_analyzed.csv.')


@timing
def main(k):
    word = k
    tweet_miner = TweetMiner(word)
    tweet_miner.main_function(test=False)
    df = tweet_miner.get_df()
    word_tokens = [word_tokenize(tweet) for tweet in df.text]
    len_tokens = []
    for i in range(len(word_tokens)):
        len_tokens.append(len(word_tokens[i]))
    vect = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS, ngram_range=(1, 2), max_features=1000, token_pattern=r'\b[^\d\W][^\d\W]+\b').fit(df.text)

    X = vect.transform(df.text)
    tweets_transformed = pd.DataFrame(X.toarray(), columns=vect.get_feature_names())
    tweets_transformed['n_words'] = len_tokens
    print('Training decision trees model...')
    #train_dec_trees_model()
    print('Decision trees model trained.')
    model = pickle.load(open('dec_trees_model.sav', 'rb'))
    predict_sentiment(word, 'Decision Trees ', model, df, tweets_transformed)
    ####################################################
    
##consumer_key = "ZZCQIC9BpywndHAM0AiefpgLf"
##consumer_secret = "zGy5D2Z2AN2xZUblym7SjYRW37fKlEV2euYbT86eIUqOVZJ96W"
##access_key = "845406513338429442-VMJDy4xMuEmEL8p4Ung1aTWCa4WyGF3"
##access_secret = "gll1RwCMthjM5sLVLLdtaFYxR2wSyAnc1LJ71FyzXSRWg"
##consumer_key = "1774465869130735616-ijugZmuRHcduFuf6ApMg5LEXfo7tqO"
##consumer_secret = "wg3keU3dONV3PBUpDztTakRB3Y4qkoeqgdwWTmJRExbQ9"
##access_key = "G63yfDnPoK9rUC6rpZmvlnKQM"
##access_secret = "7mxG7dITfszt8JnNWFXsrVrggiTzuHrtKLW3rxDlH4UjcMGqhk"
##auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
##auth.set_access_token(access_key, access_secret)
##api = tweepy.API(auth)
api_key = 'BUyqqAnRvZKCvCLsKLqX2gx5K'
api_key_secret = 'd6lXHz4bW0iOxs0c52DLyM6DMYr65npUkDOdKpdyxGA46dBmqj'
access_token = '1782425097485144064-qorqLMg5p3GTs1se2bx9mQTuOLrAIg'
access_token_secret = 'mzZpXKrnoGyEDxWSZByboPs7rzirujOibWzOkHFxzGkbW'
clinet_id = 'V2xVeWk1QjBoUnFrUFp5QkJZZ0M6MTpjaQ'
clinet_secret = '4EW-W9J5H1Khms_HYgaBk2qbBm1fo3T5hDx9D8TPE6pImtXQeF'

auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
print(api)
api = tweepy.API(auth, wait_on_rate_limit=True)
username = "bharath mv"
no_of_tweets =100
print(api)
# Enter Hashtag and initial date
#print("Enter Twitter HashTag to search for")
win=Tk()
win.geometry("700x300")

name = askstring('Name', 'Enter Key Word for Tweet extraction?')
showinfo('Entered Key Word!', 'is, {}'.format(name))

win.mainloop()
words = name
main(words)
date_since = '2023-01-01'
numtweet = 100  
scrape(words, date_since, numtweet)
print('Scraping has completed!')
import pandas as pd
df = pd.read_csv('dataset.csv', encoding = 'unicode_escape')
print(df.head(5))
print(len(df.index))
serlis=df.duplicated().tolist()
print(serlis.count(True))
serlis=df.duplicated(['text']).tolist()
print(serlis.count(True))
df=df.drop_duplicates(['text'])
print(df)
#df['text'] = df_idf['title'] + df_idf['body']
df['text'] = df['text'].apply(lambda x:pre_process(x))
print(df['text'][2])
stopwords=get_stop_words("./stopwords.txt")
docs=df['text'].tolist()
cv=CountVectorizer(max_df=0.85,stop_words=stopwords)
word_count_vector=cv.fit_transform(docs)
print(word_count_vector)
cv=CountVectorizer(max_df=0.85,stop_words=stopwords,max_features=10000)
word_count_vector=cv.fit_transform(docs)
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(word_count_vector)
df['text'] =df['text'].apply(lambda x:pre_process(x))
# get test docs into a list
docs_test=df['text'].tolist()
feature_names=cv.get_feature_names()
doc=docs_test[0]
tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))
sorted_items=sort_coo(tf_idf_vector.tocoo())
keywords=extract_topn_from_vector(feature_names,sorted_items,10)
print("\n=====Doc=====")
print(doc)
print("\n===Keywords===")
for k in keywords:
    print(k,keywords[k])
################################################
import matplotlib.pyplot as plt
counts = df.groupby(['text']).size()\
           .reset_index(name='counts')\
           .counts
my_bins = np.arange(0,counts.max()+2, 1)-0.5
plt.figure()
plt.hist(counts, bins = my_bins)
plt.xlabels = np.arange(1,counts.max()+1, 1)
plt.xlabel('copies of each tweet')
plt.ylabel('frequency')
plt.yscale('log', nonposy='clip')
plt.show()
popular_hashtags = df.groupby('hashtags').size()\
                                        .reset_index(name='counts')\
                                        .sort_values('counts', ascending=False)\
                                        .reset_index(drop=True)
counts = df.groupby(['hashtags']).size()\
                              .reset_index(name='counts')\
                              .counts
my_bins = np.arange(0,counts.max()+2, 5)-0.5
plt.figure()
plt.hist(counts, bins = my_bins)
plt.xlabels = np.arange(1,counts.max()+1, 1)
plt.xlabel('hashtag number of appearances')
plt.ylabel('frequency')
plt.yscale('log', nonposy='clip')
plt.show()


    

