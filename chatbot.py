import streamlit as st


st.header("Fine Tuned Chatbot")

with st.sidebar:
    st.title("Document Store")
    file = st.file_uploader(" Upload a PDF file(s) to fine tune your chatbot", type="pdf")