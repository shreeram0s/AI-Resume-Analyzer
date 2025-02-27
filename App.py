import streamlit as st
import pdfplumber
import docx
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    return text

# Function to extract text from DOCX
def extract_text_from_docx(uploaded_file):
    doc = docx.Document(uploaded_file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

# Function to compare resume and job description
def compare_texts(resume_text, job_desc_text):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_desc_text])
    similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return round(similarity_score * 100, 2)  # Convert to percentage

# Streamlit UI
st.title("📄 AI Resume Analyzer")

st.sidebar.header("Upload Files")
resume_file = st.sidebar.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx"])
job_desc_file = st.sidebar.file_uploader("Upload Job Description (TXT/DOCX)", type=["txt", "docx"])

if resume_file and job_desc_file:
    # Extract text
    resume_text = extract_text_from_pdf(resume_file) if resume_file.name.endswith(".pdf") else extract_text_from_docx(resume_file)
    job_desc_text = extract_text_from_docx(job_desc_file) if job_desc_file.name.endswith(".docx") else job_desc_file.read().decode("utf-8")

    # Compare and display results
    match_score = compare_texts(resume_text, job_desc_text)
    st.subheader("🔍 Match Analysis")
    st.write(f"**Matching Score:** {match_score}%")

    # Suggestions based on score
    if match_score >= 80:
        st.success("✅ Your resume is a great match for this job!")
    elif 50 <= match_score < 80:
        st.warning("⚠️ Your resume is a moderate match. Consider improving alignment with job keywords.")
    else:
        st.error("❌ Your resume does not match well. Try updating it with relevant skills and experience.")

    st.subheader("📌 Recommendations")
    st.write("🔹 Ensure the resume includes key skills mentioned in the job description.")
    st.write("🔹 Optimize bullet points and align them with the job requirements.")
    st.write("🔹 Use industry-specific terminology to improve alignment.")
