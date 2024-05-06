from textblob import TextBlob
import csv
import re
import operator
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import codecs
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

tweets = []

def strip_non_ascii(string):
    stripped = (c for c in string if 0 < ord(c) < 127)
    return ''.join(stripped)


#ifile  = open('newtwittersmall.csv', "rb")
ifile  = open('output.csv', "rb")
reader = csv.reader(codecs.iterdecode(ifile, 'unicode_escape'))
print(reader.line_num)
for row in reader:
    print(row[0])
    tweet= dict()
    tweet['orig'] = row[0]
    tweet['id'] = (row[1])
    tweet['pubdate'] = (row[2])
   # Ignore retweets
    if re.match(r'^RT.*', tweet['orig']):
        continue
    tweet['clean'] = tweet['orig']
    # Remove all non-ascii characters
    tweet['clean'] = strip_non_ascii(tweet['clean'])
    # Normalize case
    tweet['clean'] = tweet['clean'].lower()
    # Remove URLS. (I stole this regex from the internet.)
    tweet['clean'] = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', tweet['clean'])
    # Fix classic tweet lingo
    tweet['clean'] = re.sub(r'\bthats\b', 'that is', tweet['clean'])
    tweet['clean'] = re.sub(r'\bive\b', 'i have', tweet['clean'])
    tweet['clean'] = re.sub(r'\bim\b', 'i am', tweet['clean'])
    tweet['clean'] = re.sub(r'\bya\b', 'yeah', tweet['clean'])
    tweet['clean'] = re.sub(r'\bcant\b', 'can not', tweet['clean'])
    tweet['clean'] = re.sub(r'\bwont\b', 'will not', tweet['clean'])
    tweet['clean'] = re.sub(r'\bid\b', 'i would', tweet['clean'])
    tweet['clean'] = re.sub(r'wtf', 'what the fuck', tweet['clean'])
    tweet['clean'] = re.sub(r'\bwth\b', 'what the hell', tweet['clean'])
    tweet['clean'] = re.sub(r'\br\b', 'are', tweet['clean'])
    tweet['clean'] = re.sub(r'\bu\b', 'you', tweet['clean'])
    tweet['clean'] = re.sub(r'\bk\b', 'OK', tweet['clean'])
    tweet['clean'] = re.sub(r'\bsux\b', 'sucks', tweet['clean'])
    tweet['clean'] = re.sub(r'\bno+\b', 'no', tweet['clean'])
    tweet['clean'] = re.sub(r'\bcoo+\b', 'cool', tweet['clean'])
    tweet['TextBlob'] = TextBlob(tweet['clean'])
    tweets.append(tweet)
for tweet in tweets:
    tweet['polarity'] = float(tweet['TextBlob'].sentiment.polarity)
    tweet['subjectivity'] = float(tweet['TextBlob'].sentiment.subjectivity)

    if tweet['polarity'] >= 0.1:
        tweet['sentiment'] = 'positive'
    elif tweet['polarity'] <= -0.1:
        tweet['sentiment'] = 'negative'
    else:
        tweet['sentiment'] = 'neutral'

tweets_sorted = sorted(tweets, key=lambda k: k['polarity'])

print("\n\nTOP NEGATIVE TWEETS")
negative_tweets = [d for d in tweets_sorted if d['sentiment'] == 'negative']
for tweet in negative_tweets[0:5]:
    print((tweet['id'], tweet['polarity'], tweet['clean']))

print("\n\nTOP POSITIVE TWEETS")
positive_tweets = [d for d in tweets_sorted if d['sentiment'] == 'positive']
for tweet in positive_tweets[-5:]:
    print((tweet['id'], tweet['polarity'], tweet['clean']))

print("\n\nTOP NEUTRAL TWEETS")
neutral_tweets = [d for d in tweets_sorted if d['sentiment'] == 'neutral']
for tweet in neutral_tweets[0:5]:
    print((tweet['id'], tweet['polarity'], tweet['clean']))

x = [d['polarity'] for d in tweets_sorted]
##num_bins = 21
##n, bins, patches = plt.hist(x, num_bins, normed=1, facecolor='green', alpha=0.5)
##plt.xlabel('Polarity')
##plt.ylabel('Probability')
##plt.title(r'Histogram of polarity')
##plt.subplots_adjust(left=0.15)
##plt.show()

pos = len(positive_tweets)
neu = len(negative_tweets)
neg = len(neutral_tweets)
labels = 'Positive', 'Neutral', 'Negative'
sizes = [pos, neu, neg]
colors = ['yellowgreen', 'gold', 'lightcoral']
plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
plt.axis('equal')
plt.show()
######################################
df = pd.DataFrame(tweets_sorted)
df.to_csv('outputres.csv',index = False)
#####################################
dataset = df
Columns = ["orig", "id", "pubdate",  "TextBlob", "sentiment"]
dataset.drop(columns = Columns, axis = 1, inplace = True)
Encoder = LabelEncoder()
dataset["sentiment"] = Encoder.fit_transform(dataset["polarity"])
dataset["sentiment"].value_counts()

# Defining our vectorizer with total words of 5000 and with bigram model
TF_IDF = TfidfVectorizer(max_features = 5000, ngram_range = (2, 2))

# Fitting and transforming our reviews into a matrix of weighed words
# This will be our independent features
X = TF_IDF.fit_transform(dataset["clean"])

# Check our matrix shape
X.shape

# Declaring our target variable
y = dataset["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

DTree = DecisionTreeClassifier()
LogReg = LogisticRegression()
SVC = SVC()
RForest = RandomForestClassifier()
Bayes = BernoulliNB()
KNN = KNeighborsClassifier()

Models = [DTree, LogReg, SVC, RForest, Bayes, KNN]
Models_Dict = {0: "Decision Tree", 1: "Logistic Regression", 2: "SVC", 3: "Random Forest", 4: "Naive Bayes", 5: "K-Neighbors"}

for i, model in enumerate(Models):
  print("{} Test Accuracy: {}".format(Models_Dict[i], cross_val_score(model, X, y, cv = 10, scoring = "accuracy").mean()))

Param = {"C": np.logspace(-4, 4, 50), "penalty": ['l1', 'l2']}
grid_search = GridSearchCV(estimator = LogisticRegression(random_state = 42), param_grid = Param, scoring = "accuracy", cv = 10, verbose = 0, n_jobs = -1)

grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)


Classifier = LogisticRegression(random_state = 42, C = 6866.488450042998, penalty = 'l2')
Classifier.fit(X_train, y_train)

Prediction = Classifier.predict(X_test)

accuracy_score(y_test, Prediction)


ConfusionMatrix = confusion_matrix(y_test, Prediction)


def plot_cm(cm, classes, title, normalized = False, cmap = plt.cm.Blues):

  plt.imshow(cm, interpolation = "nearest", cmap = cmap)
  plt.title(title, pad = 20)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes)
  plt.yticks(tick_marks, classes)

  if normalized:
    cm = cm.astype('float') / cm.sum(axis = 1)[: np.newaxis]
    print("Normalized Confusion Matrix")
  else:
    print("Unnormalized Confusion Matrix")
  
  threshold = cm.max() / 2
  for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
      plt.text(j, i, cm[i, j], horizontalalignment = "center", color = "white" if cm[i, j] > threshold else "black")

  plt.tight_layout()
  plt.xlabel("Predicted Label", labelpad = 20)
  plt.ylabel("Real Label", labelpad = 20)
  plt.show()  
plot_cm(ConfusionMatrix, classes = ["Positive", "Neutral", "Negative"], title = "Confusion Matrix of Sentiment Analysis")

print(classification_report(y_test, Prediction))

