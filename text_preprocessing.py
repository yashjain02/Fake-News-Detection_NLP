import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from nltk.corpus import stopwords
import re
from nltk.stem import PorterStemmer


def drop_null(features):
    features = features.dropna()
    return features


def stemming(features):
    ps = PorterStemmer()
    corpus = []
    for i in range(0, len(features)):
        review = re.sub('[^a-zA-Z]', ' ', features['title'][i])
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        corpus.append(review)
    return corpus


def bag_of_words(features):
    bag_of_words = CountVectorizer(max_features=5000, ngram_range=(1, 5))
    bag_of_words.fit(features)
    joblib.dump(bag_of_words,'joblib/bagofwords')
    cv = joblib.load('joblib/bagofwords')
    features = cv.transform(features)
    return features


def text_processing(features):
    features = stemming(features)
    features = bag_of_words(features)
    return features


