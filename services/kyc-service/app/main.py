from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import Response
from .services.ocr_engine import extract_text, extract_text_advanced
from .services.preprocess import preprocess_image
from .services.face_service import get_face_service
from .models.llm import IDExtractor 
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

@app.post("/kyc")
async def kyc_service(
    id_card: UploadFile = File(...),
    selfie: UploadFile = File(...),
    preprocess: bool = Query(default=True, description="Apply image preprocessing"),
    advanced: bool = Query(default=False, description="Use advanced OCR with multiple strategies"),
):

    if not token:
        return {"error": "GITHUB_TOKEN environment variable is required"}
    
    try:
        contents = await id_card.read()
        img = Image.open(io.BytesIO(contents))
        
        if preprocess:
            img = preprocess_image(img)
        
        if advanced:
            text = extract_text_advanced(img)
        else:
            text = extract_text(img)  
    
        llm = IDExtractor(token) 
      
        structured_data = llm.extract_to_json(text, model)
        
        if isinstance(structured_data, str):
           
            try:
                structured_data = json.loads(structured_data)
               
            except json.JSONDecodeError:
                
                structured_data = {"raw_response": structured_data}
        
        return structured_data
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "filename": id_card.filename
        }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "National ID OCR Service"}




@app.post("/face/detect")
async def detect_faces(
    file: UploadFile = File(...),
):
    """
    Detect faces and extract 5-point landmarks using RetinaFace detector.
    Returns bounding boxes, landmarks, and detection scores.
    """
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        
        face_service = get_face_service()
        faces = face_service.detect_faces(img)
        
        return {
            "status": "success",
            "face_count": len(faces),
            "faces": faces
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/face/align")
async def align_face(
    file: UploadFile = File(...),
    output_size: str = Query(default="112,112", description="Output size as width,height"),
):
    """
    Auto-align face crop for embedding using 5-point landmarks.
    Requires face to be detected first. Returns aligned face image.
    """
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        
        # Parse output size
        try:
            width, height = map(int, output_size.split(','))
            output_size_tuple = (width, height)
        except:
            output_size_tuple = (112, 112)
        
        face_service = get_face_service()
        
        # First detect faces to get landmarks
        faces = face_service.detect_faces(img)
        if not faces:
            raise HTTPException(status_code=400, detail="No faces detected in image")
        
        # Use first face's landmarks for alignment
        landmarks = faces[0]["landmarks"]
        if len(landmarks) < 5:
            raise HTTPException(status_code=400, detail="Insufficient landmarks for alignment")
        
        aligned_face = face_service.align_face(img, landmarks, output_size_tuple)
        
        # Convert aligned image to bytes for response
        img_byte_arr = io.BytesIO()
        aligned_face.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return Response(content=img_byte_arr.read(), media_type="image/png")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "message": "National ID OCR Service",
        "endpoints": {
            "ocr": {
                "extract": "/extract",
            },
            "face": {
                "detect": "/face/detect",
                "align": "/face/align",
                "embed": "/face/embed",
                "detect_and_embed": "/face/detect-and-embed"
            },
            "docs": "/docs", 
            "health": "/health"
        }
    }