import pytesseract
import easyocr
from PIL import Image

class OCR:
    def __init__(self):
        self.easyocr_reader = easyocr.Reader(['en'], gpu=False)

    def extract_text(self, image: Image.Image) -> dict:
        """
        Returns raw OCR text from both engines.
        """
        # Tesseract
        tesseract_result = pytesseract.image_to_string(image)

        # EasyOCR
        easyocr_result = self.easyocr_reader.readtext(
            image, detail=0, paragraph=True
        )

        return {
            "tesseract_text": tesseract_result,
            "easyocr_text": "\n".join(easyocr_result),
            "combined_text": tesseract_result + "\n" + "\n".join(easyocr_result)
        }
