import joblib
import pandas as pd
import streamlit as st
import text_preprocessing
st.title('Fake news and profinity detection')

input_data = [st.text_input('ID'), st.text_input('title'), st.text_input('author'), st.text_input('text')]

input_data = pd.DataFrame([input_data], columns=['id', 'title', 'author', 'text'])

if st.button('predict'):
    x = text_preprocessing.stemming(input_data)
    cv = joblib.load('joblib/bagofwords')
    x = cv.transform(x)
    pred = joblib.load('joblib/model')
    x = pred.predict(x)
    if x[0] == 1:
      st.success('Its a fake news')
    else:
        st.success('true news')