from fastapi import FastAPI, File, UploadFile, Query
from .ocr_engine import extract_text, extract_text_advanced
from .preprocess import preprocess_image
from .llm.llm import IDExtractor 
import shutil
import os
from PIL import Image
import io
import json

app = FastAPI(title="National ID OCR Service")

# Environment variables with defaults
token = os.getenv("GITHUB_TOKEN")
model = os.getenv("MODEL") 
print(f"GITHUB_TOKEN loaded: {'Yes' if token else 'No'}")
print(f"MODEL: {model}")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/extract")
async def extract_id_text(
    file: UploadFile = File(...),
    preprocess: bool = Query(default=True, description="Apply image preprocessing"),
    advanced: bool = Query(default=False, description="Use advanced OCR with multiple strategies"),
):
    """
    Upload an image and get the raw extracted text.
    Optionally apply preprocessing (grayscale, contrast, sharpen) if preprocess=true.
    Use advanced mode for better accuracy with multiple OCR strategies.
    """
    # Validate environment variables
    if not token:
        return {"error": "GITHUB_TOKEN environment variable is required"}
    
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        
        if preprocess:
            img = preprocess_image(img)
        
        if advanced:
            text = extract_text_advanced(img)
        else:
            text = extract_text(img)  
        print("test 1")
        # Initialize LLM and extract structured data
        print(token)
        llm = IDExtractor(token) 
        print("test 2")
        structured_data = llm.extract_to_json(text, model)
        print("test 3")
        # Parse JSON response if it's a string
        if isinstance(structured_data, str):
            print("test 3")
            try:
                structured_data = json.loads(structured_data)
                print("test 4")
            except json.JSONDecodeError:
                # If it's not valid JSON, wrap it
                print("test 5")
                structured_data = {"raw_response": structured_data}
        
        return structured_data
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "filename": file.filename
        }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "National ID OCR Service"}

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "message": "National ID OCR Service",
        "endpoints": {
            "extract_text": "/extract-text",
            "docs": "/docs", 
            "health": "/health"
        }
    }