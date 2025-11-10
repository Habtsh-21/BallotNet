from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from app.ocr_engine import OCR
from app.parser import parse_id
from app.preprocess import preprocess_image
from app.schemas import OCRResponse
from PIL import Image
import io

app = FastAPI(title="ID OCR Service", version="1.0")

ocr_engine = OCR()

@app.post("/extract", response_model=OCRResponse)
async def extract(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = preprocess_image(image)

        # OCR using both engines
        raw_text = ocr_engine.extract_text(image)

        # parse key fields
        parsed = parse_id(raw_text)

        return JSONResponse(content={
            "ok": True,
            "raw_ocr": raw_text,
            "parsed": parsed
        })

    except Exception as e:
        return JSONResponse(content={"ok": False, "error": str(e)}, status_code=500)
