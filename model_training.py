import pandas as pd
import text_preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import recall_score, confusion_matrix, precision_score, f1_score, accuracy_score
import joblib


features = pd.read_csv('training-data/train.csv')
features = text_preprocessing.drop_null(features)
features.reset_index(inplace=True)
target = features['label']
features = features.drop(['id','label'], axis=1)
features = text_preprocessing.text_processing(features)
X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.33, random_state=42)
classifier = PassiveAggressiveClassifier(max_iter=50)
classifier.fit(X_train, Y_train)
joblib.dump(classifier,'joblib/model',protocol=5)
prediction = classifier.predict(X_test)
score = recall_score(Y_test, prediction)
print(score)
print(accuracy_score(Y_test,prediction))
print(f1_score(Y_test,prediction))
print(precision_score(Y_test,prediction))
print(confusion_matrix(Y_test,prediction))



