from PIL import Image, ImageFilter, ImageOps
import pytesseract


def extract_text(img):
   
    text = pytesseract.image_to_string(img)
    return text

if __name__ == "__main__":
    print(extract_text("id.png"))
