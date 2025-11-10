from PIL import Image, ImageFilter, ImageOps

def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Convert to grayscale, enhance contrast, and apply sharpening.
    """
    image = image.convert("L")                     # grayscale
    image = ImageOps.autocontrast(image)           # auto contrast
    image = image.filter(ImageFilter.SHARPEN)      # sharpen edges
    return image
