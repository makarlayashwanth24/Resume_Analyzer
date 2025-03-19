from fastapi import FastAPI, Body
import os
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
    allow_origins=["*"],  # You can restrict this to your frontend domain for better security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_gemini_response(job_description, resume_content, prompt):
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    # Combine the job description and resume content for the model input
    combined_input = f"""
    Job Description:
    {job_description}

    Resume Content:
    {resume_content}
    """
    
    response = model.generate_content([combined_input, prompt])
    return response.text

# API Endpoint
@app.post("/analyze-resume/")
async def analyze_resume(
    jobDescription: str = Body(..., embed=True),
    promptType: str = Body(..., embed=True),
    resumeContent: str = Body(..., embed=True)
):
    # Define your prompt templates
    prompts = {
        "review": """
        You are an experienced Technical Human Resource Manager. Your task is to review the provided resume text against the job description. 
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

    # Get the selected prompt based on promptType
    selected_prompt = prompts.get(promptType)

    if not selected_prompt:
        return {"error": "Invalid prompt type selected"}

    # Get the response from Gemini
    try:
        response_text = get_gemini_response(jobDescription, resumeContent, selected_prompt)
        return {"response": response_text}
    except Exception as e:
        return {"error": f"Resume analysis failed: {str(e)}"}
