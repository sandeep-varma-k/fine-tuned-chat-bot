import streamlit as st
from PyPDF2 import PdfReader


st.header("Fine Tuned Chatbot")

# Document Upload Section
with st.sidebar:
    st.title("Document Store")
    file = st.file_uploader(" Upload a PDF file(s) to fine tune your chatbot", type="pdf")


# Extract the text from pdf file
if file is not None:
    pdf_reader = PdfReader(file)
    extracted_text = ""

    for page in pdf_reader.pages:
        extracted_text += page.extract_text()

    # Print all text to screen
    # st.write(extracted_text)


