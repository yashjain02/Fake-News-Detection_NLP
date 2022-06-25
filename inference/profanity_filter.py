import numpy as np
import joblib

vectorizer = joblib.load('../models/vectorizer.joblib', mmap_mode=None)
model = joblib.load('../models/model.joblib', mmap_mode=None)


def predict(texts):
    return model.predict(vectorizer.transform(texts))
