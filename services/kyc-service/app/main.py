from pickle import DICT
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.responses import Response
from .services.ocr_engine import extract_text, extract_text_advanced
from .services.preprocess import preprocess_image
from .services.face_service import get_face_service
from .models.llm import IDExtractor
import os
from PIL import Image
import io
import json
import tempfile

app = FastAPI(
    title="KYC Service",
    description="Know Your Customer service with OCR and Face Recognition"
)

token = os.getenv("GITHUB_TOKEN")
model = os.getenv("MODEL", "openai/gpt-4.1")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/kyc")
async def kyc_service(
    front_id_card: UploadFile = File(..., description="Front ID card image"),
    back_id_card: UploadFile = File(..., description="Back ID card image"),
    selfie: UploadFile = File(..., description="Selfie photo"),
    user_data: str = Query(..., description="User data as JSON string"),
    preprocess: bool = Query(default=True, description="Apply image preprocessing"),
    advanced: bool = Query(default=False, description="Use advanced OCR with multiple strategies"),
    compare_faces: bool = Query(default=True, description="Compare selfie with ID photo")
):
    """
    Complete KYC processing endpoint with face comparison
    """
    if not token:
        raise HTTPException(
            status_code=500,
            detail="GITHUB_TOKEN environment variable is required"
        )
    
    temp_selfie_path = None
    temp_front_id_card_path = None
    temp_back_id_card_path  = None
    
    try:
        try:
            user_input = json.loads(user_data)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON data: {str(e)}")
        

        selfie_contents = await selfie.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_selfie:
            temp_selfie.write(selfie_contents)
            temp_selfie_path = temp_selfie.name
        
        front_id_card_contents = await front_id_card.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_front_id_card_path:
            temp_front_id_card_path.write(front_id_card_contents)
            temp_front_id_card_path = temp_front_id_card_path.name

        back_id_card_contents = await back_id_card.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_back_id_card_path:
            temp_back_id_card_path.write(back_id_card_contents)
            temp_back_id_card_path = temp_back_id_card_path.name
        
        front_id_card_image = Image.open(io.BytesIO(front_id_card_contents))
        back_id_card_image = Image.open(io.BytesIO(back_id_card_contents))

        
        if preprocess:
            front_id_card_image = preprocess_image(front_id_card_image)
            back_id_card_image  = preprocess_image(back_id_card_image)

        if advanced:
            front_text = extract_text_advanced(front_id_card_image)
            back_text  =  extract_text_advanced(back_id_card_image)
        else:
            front_text = extract_text(front_id_card_image)
            back_text  =  extract_text(back_id_card_image)
        
        llm = IDExtractor(token)
        structured_data = llm.extract_to_json(front_text + back_text , user_data,model)
        
        if isinstance(structured_data, str):
            try:
                structured_data = json.loads(structured_data)
            except json.JSONDecodeError:
                structured_data = {"raw_response": structured_data}
        
        face_service = get_face_service()
        
        selfie_result = face_service.process_kyc_document(temp_selfie_path)
        
        face_comparison_result = None
        if compare_faces and selfie_result["status"] == "success":
            id_photo_result = face_service.extract_id_photo_embedding(
                Image.open(io.BytesIO(front_id_card_contents))
            )
            
            if id_photo_result["status"] == "success":
                face_comparison_result = face_service.compare_faces(
                    selfie_result["embeddings"],
                    id_photo_result["embedding"]
                )
        
        if temp_selfie_path and os.path.exists(temp_selfie_path):
            os.unlink(temp_selfie_path)
        if temp_front_id_card_path and os.path.exists(temp_front_id_card_path):
            os.unlink(temp_front_id_card_path)
        if temp_back_id_card_path and os.path.exists(temp_back_id_card_path):
            os.unlink(temp_back_id_card_path)
        
        response = {
            "status": "success",
            "id_card_data": structured_data,
            "selfie_processing": {
                "embeddings": selfie_result.get("embeddings"),
                "embedding_dim": selfie_result.get("embedding_dim"),
                "aligned": selfie_result.get("aligned", True),
                "face_detected": selfie_result.get("face_detected", False),
                "detection_confidence": selfie_result.get("detection_confidence"),
                "quality_checks": selfie_result.get("quality_checks")
            }
        }
        
        if face_comparison_result:
            response["face_comparison"] = face_comparison_result
            response["kyc_verified"] = (
                selfie_result["status"] == "success" and 
                face_comparison_result.get("verified", False)
            )


        
        return response
        
    except Exception as e:
        if temp_selfie_path and os.path.exists(temp_selfie_path):
            os.unlink(temp_selfie_path)
        if temp_front_id_card_path and os.path.exists(temp_front_id_card_path):
            os.unlink(temp_front_id_card_path)
        if temp_back_id_card_path and os.path.exists(temp_back_id_card_path):
            os.unlink(temp_back_id_card_path)
               
        raise HTTPException(
            status_code=500,
            detail=f"Error processing KYC: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "KYC Service"
    }


@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "message": "KYC Service",
        "description": "Know Your Customer service with OCR and Face Recognition",
        "endpoints": {
            "kyc": "/kyc",
            "compare_faces": "/compare-faces",
            "health": "/health"
        }
    }