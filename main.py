import joblib
import pandas as pd
import streamlit as st
import text_preprocessing
st.title('Fake news and profinity detection')

input_data = [st.text_input('title'), st.text_input('author'), st.text_input('text')]

input_data = pd.DataFrame([input_data], columns=['title', 'author', 'text'])

test = pd.read_csv('training-data/test.csv')

if st.button('predict'):
    features = text_preprocessing.stemming(input_data)
    bagofwords = joblib.load('joblib/bagofwords')
    features = bagofwords.transform(features)
    prediction = joblib.load('joblib/model')
    features = prediction.predict(features)
    profaniy_test = joblib.load('joblib/vectorizer')
    profanity = profaniy_test.transform(input_data['text'])
    profanity_model = joblib.load('joblib/profanity_model')
    profanity_prediction = profanity_model.predict(profanity)
    if features[0] == 1:
        st.success('Its a fake news')
    else:
        st.success('Its is a true news')
    print(profanity_prediction)
    if profanity_prediction[0] == 1:
        st.success('Profanity detected')
    else:
        st.success('No Profanity found')
st.text('Some of the sample inputs for testing our model')
st.dataframe(test)