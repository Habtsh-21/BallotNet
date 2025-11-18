from PIL import Image, ImageFilter, ImageOps, ImageEnhance

def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Enhanced preprocessing for better OCR results
    """

    image = image.convert("L")
    
    image = ImageOps.autocontrast(image, cutoff=2)
    
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(2.0)
    
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)
    

    image = image.filter(ImageFilter.MedianFilter(size=3))
    
    return image
