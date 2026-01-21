import pdfplumber
import spacy
import re
import os
from sklearn.metrics import classification_report, accuracy_score


#load spacy model
nlp = spacy.load("en_core_web_sm")

# upload pdf and read
pdf_path = "Prema_Dongare.pdf"

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text


resume_text =extract_text_from_pdf(pdf_path)
print(resume_text)

# extract basic info (rule-base)

# email
def extract_email(text):
    match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
    return match.group()if match else None

# phone number
def extract_phone_number(text):
    match= re.search(r'\b\d{10}\b', text)
    return match.group() if match else None

# name using spacy person entity
def extract_name(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return None
        
#extract skills(keyword matching)

skills_list=["Python", "Java", "C++", "Machine Learning", "Deep Learning", "SQL", "NoSQL", "TensorFlow", "Keras", "PyTorch"]

def extract_skills(text):
    found_skills = []
    for skill in skills_list:
        if skill.lower() in text.lower():
            found_skills.append(skill)
    return found_skills

# Extract information
resume_data = {
    "email": extract_email(resume_text),
    "phone_number": extract_phone_number(resume_text),
    "name": extract_name(resume_text),
    "skills": extract_skills(resume_text)
}
# display final op
for key, value in resume_data.items():
    print(f"{key}: {value}")





