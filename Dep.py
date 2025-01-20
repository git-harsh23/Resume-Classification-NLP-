import streamlit as st
import pickle
import re
from docx import Document

# Function to load model and vectorizer with enhanced error handling
def load_model_and_vectorizer():
    try:
        with open('C:/Users/user/Documents/EXCELR/Project-2_NLP/Dataset/artifacts/rf_clf.sav', 'rb') as model_name:
            model = pickle.load(model_name)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

    try:
        with open('C:/Users/user/Documents/EXCELR/Project-2_NLP/Dataset/artifacts/vectorizer_filename.sav', 'rb') as vectorizer_filename:
            vectorizer = pickle.load(vectorizer_filename)
    except Exception as e:
        st.error(f"Error loading vectorizer: {e}")
        return model, None
    
    return model, vectorizer

# Load the trained model and TF-IDF vectorizer
model, vectorizer = load_model_and_vectorizer()

if model is None or vectorizer is None:
    st.stop()  # Stop execution if model or vectorizer is not loaded

# Function to preprocess the text (optional but can improve consistency)
def preprocess_text(text):
    text = text.lower()  # Lowercase text
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'\W', ' ', text)  # Remove non-alphanumeric characters
    return text

# Function to predict the job role from the resume text
def predict_job_role(resume_text):
    try:
        if resume_text.strip():  # Ensure resume_text is not empty
            resume_text = preprocess_text(resume_text)  # Preprocess the text
            resume_tfidf = vectorizer.transform([resume_text])  # Transform the text into TF-IDF features
            prediction = model.predict(resume_tfidf)  # Predict the job role
            return prediction[0]
        else:
            st.error("Uploaded resume is empty.")
            return None
    except Exception as e:
        st.error(f"Error during job role prediction: {e}")
        return None

# Function to read text from a .docx file
def extract_text_from_docx(docx_file):
    try:
        doc = Document(docx_file)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
    except Exception as e:
        st.error(f"Error extracting text from docx file: {e}")
        return ""

# Streamlit app layout
st.title("Resume Classification")
st.header("Upload your Resume")

# File upload functionality (supports only .docx files)
uploaded_file = st.file_uploader("Choose a resume file (Word Document)", type=["docx"])

if uploaded_file is not None:
    # Extract the resume content from the .docx file
    resume_text = extract_text_from_docx(uploaded_file)

    # Check if resume content is empty
    if not resume_text.strip():
        st.error("The uploaded resume is empty. Please upload a valid .docx file.")
    else:
        # Predict job role only when the button is pressed
        if st.button("Classify Job Role"):
            predicted_role = predict_job_role(resume_text)
            if predicted_role:
                # Show only the classified job role
                st.success(f"The resume is classified for the job role: {predicted_role}")