from pydantic import BaseModel
from typing import Dict, Optional

class OCRResponse(BaseModel):
    ok: bool
    raw_ocr: Dict[str, str]
    parsed: Optional[Dict[str, Optional[str]]] = None