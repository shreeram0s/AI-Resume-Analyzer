import streamlit as st
import pdfplumber
import docx
import re
import spacy
from io import BytesIO

# Load NLP model for named entity recognition
nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def extract_email(text):
    match = re.search(r"[\w.-]+@[\w.-]+\.\w+", text)
    return match.group(0) if match else "Not Found"

def extract_phone(text):
    match = re.search(r"\+?\d{10,15}", text)
    return match.group(0) if match else "Not Found"

def extract_name(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return "Not Found"

def extract_skills(text):
    skills = ["Python", "Java", "Machine Learning", "Data Science", "Deep Learning", "SQL", "AI", "NLP"]
    found_skills = [skill for skill in skills if skill.lower() in text.lower()]
    return found_skills if found_skills else ["Not Found"]

# Streamlit UI
st.title("📄 AI Resume Analyzer")
uploaded_file = st.file_uploader("Upload your Resume (PDF/DOCX)", type=["pdf", "docx"])

if uploaded_file:
    file_extension = uploaded_file.name.split(".")[-1].lower()
    if file_extension == "pdf":
        resume_text = extract_text_from_pdf(uploaded_file)
    elif file_extension == "docx":
        resume_text = extract_text_from_docx(uploaded_file)
    else:
        st.error("Unsupported file format. Please upload a PDF or DOCX file.")
        st.stop()
    
    # Extract details
    name = extract_name(resume_text)
    email = extract_email(resume_text)
    phone = extract_phone(resume_text)
    skills = extract_skills(resume_text)
    
    # Display results
    st.subheader("Extracted Resume Details")
    st.write(f"**Name:** {name}")
    st.write(f"**Email:** {email}")
    st.write(f"**Phone:** {phone}")
    st.write(f"**Skills:** {', '.join(skills)}")
    
    # Download results
    result_text = f"Name: {name}\nEmail: {email}\nPhone: {phone}\nSkills: {', '.join(skills)}"
    result_bytes = BytesIO(result_text.encode())
    st.download_button("Download Extracted Details", result_bytes, "resume_analysis.txt", "text/plain")
