import numpy as np
import joblib

vectorizer = joblib.load('../models/vectorizer.joblib', mmap_mode=None)
model = joblib.load('../models/model.joblib', mmap_mode=None)


def predict(texts):
    data = vectorizer.transform(texts)
    model(data)
    return model(data)