from fastapi import FastAPI, UploadFile, File, Form
import os
import fitz
from dotenv import load_dotenv
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = FastAPI()

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_text_from_pdf(uploaded_file):
    document = fitz.open(stream=uploaded_file.file.read(), filetype="pdf")
    text_parts = [page.get_text() for page in document]
    return " ".join(text_parts)

def get_gemini_response(input_text, pdf_content, prompt):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content([input_text, pdf_content, prompt])
    return response.text

# API Endpoints
@app.post("/analyze-resume/")
async def analyze_resume(
    input_text: str = Form(...),
    uploaded_file: UploadFile = File(...),
    prompt_type: str = Form(...)
):
    pdf_content = extract_text_from_pdf(uploaded_file)
    
    prompts = {
        "review": """
        You are an experienced Technical Human Resource Manager. Your task is to review the provided resume against the job description. 
        Please share your professional evaluation on whether the candidate's profile aligns with the role. 
        Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements.
        """,
        "improve": """
        You are a Technical HR Manager with expertise in data science. 
        Your role is to scrutinize the resume in light of the job description provided. 
        Share insights on the candidate's suitability for the role and offer advice on enhancing skills.
        """,
        "keywords": """
        You are an ATS scanner with a deep understanding of data science and ATS functionality. 
        Your task is to evaluate the resume against the job description and identify missing keywords.
        """,
        "match": """
        You are an ATS scanner with expertise in resume analysis. 
        Evaluate the resume and provide a match percentage, missing keywords, and final thoughts.
        """
    }
    
    selected_prompt = prompts.get(prompt_type, "Invalid prompt type")
    
    if selected_prompt == "Invalid prompt type":
        return {"error": "Invalid prompt type selected"}

    response = get_gemini_response(input_text, pdf_content, selected_prompt)
    return {"response": response}
