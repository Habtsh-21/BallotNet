from fastapi import FastAPI, File, UploadFile, Query
from .ocr_engine import extract_text
from .preprocess import preprocess_image
import shutil
import os
from PIL import Image

app = FastAPI(title="National ID OCR Service")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/extract-text")
async def extract_id_text(
    file: UploadFile = File(...),
    preprocess: bool = Query(default=True, description="Apply image preprocessing")
):
    """
    Upload an image and get the raw extracted text.
    Optionally apply preprocessing (grayscale, contrast, sharpen) if preprocess=true.
    """
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    # Save uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Load image
    img = Image.open(file_path)

    # Apply preprocessing if requested
    # if preprocess:
    #     img = preprocess_image(img)
    
    # OCR extraction
    text = extract_text(img)

    return {"filename": file.filename, "raw_text": text, "preprocessed": preprocess}
