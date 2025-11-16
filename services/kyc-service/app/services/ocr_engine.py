import pytesseract
from PIL import Image
import logging


logger = logging.getLogger(__name__)


def extract_text(img):
    """Extract text using pytesseract with optimized configuration"""
    
    # Configure pytesseract for ID documents
    custom_config = r'--oem 3 --psm 6'
    
    try:
        text = pytesseract.image_to_string(img, config=custom_config)
        return text.strip()
    except Exception as e:
        logger.error(f"Pytesseract OCR error: {e}")
        return ""


def extract_text_advanced(img):
    """Advanced pytesseract with multiple PSM strategies"""
    results = []
    psm_modes = [6, 11, 7, 12, 8]  
    
    for psm in psm_modes:
        try:
            config = f'--oem 3 --psm {psm}'
            text = pytesseract.image_to_string(img, config=config)
            results.append((psm, text))
        except Exception as e:
            logger.warning(f"PSM {psm} failed: {e}")
            continue
    
    # Return the result with most non-empty lines
    if results:
        best_result = max(results, key=lambda x: len([line for line in x[1].split('\n') if line.strip()]))
        return best_result[1].strip()
    
    return ""