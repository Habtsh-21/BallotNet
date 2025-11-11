from PIL import Image, ImageFilter, ImageOps, ImageEnhance
# import cv2
# import numpy as np

def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Enhanced preprocessing for better OCR results
    """
    # Convert to grayscale
    image = image.convert("L")
    
    # Increase contrast
    image = ImageOps.autocontrast(image, cutoff=2)
    
    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(2.0)
    
    # Enhance contrast again
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.5)
    
    # Apply mild smoothing to reduce noise
    image = image.filter(ImageFilter.MedianFilter(size=3))
    
    return image

# def preprocess_with_opencv(image: Image.Image) -> Image.Image:
#     """
#     Use OpenCV for more advanced image preprocessing
#     """
#     # Convert PIL to OpenCV format
#     opencv_image = np.array(image)
    
#     # Convert to grayscale if needed
#     if len(opencv_image.shape) == 3:
#         opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2GRAY)
    
#     # Apply Gaussian blur to reduce noise
#     opencv_image = cv2.GaussianBlur(opencv_image, (3, 3), 0)
    
#     # Apply adaptive threshold
#     opencv_image = cv2.adaptiveThreshold(
#         opencv_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
#     )
    
#     # Convert back to PIL
#     return Image.fromarray(opencv_image)