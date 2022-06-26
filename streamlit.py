import streamlit as st
import pyperclip


def copyToClipboard():
    pyperclip.copy(result)

#TODO: return analyse result instead of txt
def run_sentiment_analysis(txt):
    return txt

st.title('Fake News Detection')
col1, col2 = st.columns(2)
with col1:
    st.header("detection text")
    txt = st.text_area('Text to analyze')
    analyseButton = st.button('analyse')

with col2:
    st.header("result text")
    if analyseButton:
        result = run_sentiment_analysis(txt)
        st.text_area('Analyse text', value = result)
    else:
        st.text_area('Analyse text')
    st.button('copy', on_click=copyToClipboard)

