import streamlit as st
import requests
import pandas as pd
import pdfplumber
import docx2txt
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
import googleapiclient.discovery

# Load AI Model
st_model = SentenceTransformer('all-MiniLM-L6-v2')

# YouTube API Key (Replace with a new secured key)
YOUTUBE_API_KEY = "AIzaSyBoRgw0WE_KzTVNUvH8d4MiTo1zZ2SqKPI"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

# Set Page Configuration
st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄", layout="wide")

# Initialize session state
if "skills_analyzed" not in st.session_state:
    st.session_state.skills_analyzed = False
    st.session_state.missing_skills = []
    st.session_state.matching_score = 0.0
    st.session_state.resume_skills = []
    st.session_state.job_skills = []

# Streamlit UI
st.title("📄 AI Resume Analyzer & Skill Enhancer")
st.markdown("Analyze your resume, compare it with job requirements, and get AI-driven course recommendations! 🎯")

# Layout for file upload
col1, col2 = st.columns(2)
with col1:
    resume_file = st.file_uploader("📄 Upload Resume (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], key="resume")
with col2:
    job_file = st.file_uploader("📄 Upload Job Description (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"], key="job")

st.markdown("---")  # Divider

# Function to extract text from files
def extract_text(uploaded_file):
    if uploaded_file is not None:
        ext = uploaded_file.name.split(".")[-1].lower()
        if ext == "pdf":
            with pdfplumber.open(uploaded_file) as pdf:
                return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        elif ext in ["docx", "doc"]:
            return docx2txt.process(uploaded_file)
        elif ext == "txt":
            return uploaded_file.read().decode("utf-8")
        else:
            st.error("Unsupported file format! Please upload PDF, DOCX, or TXT.")
    return ""

if resume_file and job_file:
    resume_text = extract_text(resume_file)
    job_text = extract_text(job_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📌 Resume Summary")
        st.write(resume_text[:300] + "...")
    with col2:
        st.subheader("📌 Job Description Summary")
        st.write(job_text[:300] + "...")
    
    st.markdown("---")  # Divider
    
    if st.button("🚀 Analyze Skills & Matching Score"):
        resume_skills = ["Python", "Machine Learning", "Data Science", "AI", "Deep Learning", "NLP", "SQL", "Power BI", "Tableau", "TensorFlow", "Pandas", "Numpy"]
        job_skills = [skill for skill in resume_skills if skill.lower() in job_text.lower()]
        extracted_resume_skills = [skill for skill in resume_skills if skill.lower() in resume_text.lower()]
        missing_skills = list(set(job_skills) - set(extracted_resume_skills))
        
        matching_score = round(float(util.pytorch_cos_sim(st_model.encode(resume_text, convert_to_tensor=True),
                                                           st_model.encode(job_text, convert_to_tensor=True))[0]), 2) * 100
        
        # Store results in session state
        st.session_state.skills_analyzed = True
        st.session_state.missing_skills = missing_skills
        st.session_state.matching_score = matching_score
        st.session_state.resume_skills = extracted_resume_skills
        st.session_state.job_skills = job_skills
        
        st.success("Skills analyzed successfully! Scroll down to see the results.")

# Display results if analysis has been done
if st.session_state.skills_analyzed:
    st.subheader("📊 Resume Matching Score")
    st.success(f"Your resume matches **{st.session_state.matching_score}%** of the job requirements.")
    
    st.subheader("⚠️ Missing Skills")
    if st.session_state.missing_skills:
        st.warning(f"You are missing: {', '.join(st.session_state.missing_skills)}")
    else:
        st.success("Great! You have all the required skills.")
    
    # Skill comparison chart
    all_skills = list(set(st.session_state.resume_skills + st.session_state.job_skills))
    resume_counts = [1 if skill in st.session_state.resume_skills else 0 for skill in all_skills]
    job_counts = [1 if skill in st.session_state.job_skills else 0 for skill in all_skills]
    
    df = pd.DataFrame({"Skills": all_skills, "Resume": resume_counts, "Job Requirements": job_counts})
    df.set_index("Skills").plot(kind="bar", figsize=(8, 4), color=["#0073E6", "#E63946"], alpha=0.7)
    plt.xticks(rotation=45)
    plt.ylabel("Presence (1 = Present, 0 = Missing)")
    plt.title("Resume vs. Job Skills Comparison")
    st.pyplot(plt)
    
    st.markdown("---")  # Divider
    
    if st.button("📚 Get Recommended Courses"):
        youtube = googleapiclient.discovery.build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=YOUTUBE_API_KEY)
        all_courses = []
        for skill in st.session_state.missing_skills:
            request = youtube.search().list(q=f"{skill} course", part="snippet", maxResults=5, type="video")
            response = request.execute()
            all_courses.extend([
                {"Title": item["snippet"]["title"], "Channel": item["snippet"]["channelTitle"], "Video Link": f'https://www.youtube.com/watch?v={item["id"]["videoId"]}'}
                for item in response["items"]
            ])
        
        if all_courses:
            st.subheader("🎥 Recommended YouTube Courses")
            st.table(pd.DataFrame(all_courses))
        else:
            st.error("No courses found. Try again later!")
