import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
import joblib

# Read in data
data = pd.read_csv('training-data/clean_data.csv')
texts = data['text'].astype(str)
y = data['is_offensive']

# Vectorize the text
vectorizer = CountVectorizer(stop_words='english', min_df=0.0001)
X = vectorizer.fit_transform(texts)

# Train the model
model = LinearSVC(random_state=0, tol=1e-5)
model.fit(X,y)

# Save the model
joblib.dump(vectorizer, 'joblib/vectorizer', protocol=5)
joblib.dump(model, 'joblib/profanity_model', protocol=5)
