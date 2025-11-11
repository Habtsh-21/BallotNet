from fastapi import FastAPI, File, UploadFile, Query
from .ocr_engine import extract_text, extract_text_advanced
from .preprocess import preprocess_image
import shutil
import os
from PIL import Image
import io

app = FastAPI(title="National ID OCR Service")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/extract-text")
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

    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    
    if preprocess:
        img = preprocess_image(img)
    
    if advanced:
        text = extract_text_advanced(img)
    else:
        text = extract_text(img)  

    return {
        "filename": file.filename, 
        "raw_text": text, 
        "preprocessed": preprocess,
        "advanced_mode": advanced,
     
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