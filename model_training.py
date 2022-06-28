import pandas as pd
import text_preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import recall_score
import joblib


features = pd.read_csv('training-data/train.csv')
features = text_preprocessing.drop_null(features)
features.reset_index(inplace=True)
target = features['label']
features = features.drop('label', axis=1)
features = text_preprocessing.text_processing(features)
X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.33, random_state=42)
classifier = MultinomialNB()
classifier.fit(X_train, Y_train)
joblib.dump(classifier,'joblib/model')
prediction = classifier.predict(X_test)
score = recall_score(Y_test, prediction)



