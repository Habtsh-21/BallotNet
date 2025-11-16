"""
Face Embedding Service
Extracts 512-D face embeddings using ArcFace
"""
import cv2
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def get_embeddings(
    face_service,
    image: Image.Image,
    aligned: bool = True
) -> Dict:
    """
    Extract 512-D face embeddings using ArcFace
    
    Args:
        face_service: FaceService instance
        image: PIL Image (face crop or full image)
        aligned: If True, expects aligned face. If False, will auto-detect and align.
        
    Returns:
        Dictionary with:
        - embedding: 512-D embedding vector
        - embedding_dim: Dimension of embedding
        - aligned: Whether the input was pre-aligned
        - face_info: Face detection info (if not pre-aligned)
    """
    try:
        cv2_image = face_service._pil_to_cv2(image)
        
        if aligned:
            # Direct embedding extraction for pre-aligned faces
            faces = face_service.app.get(cv2_image)
            if not faces:
                raise ValueError("No face detected in image")
            
            # Get the first (and typically only) face
            face = faces[0]
            embedding = face.normed_embedding.tolist()
            
            return {
                "embedding": embedding,
                "embedding_dim": len(embedding),
                "aligned": True
            }
        else:
            # Auto-detect, align, and extract embedding
            faces = face_service.app.get(cv2_image)
            if not faces:
                raise ValueError("No face detected in image")
            
            # Use the first detected face
            face = faces[0]
            embedding = face.normed_embedding.tolist()
            
            # Get face info
            bbox = face.bbox.astype(int).tolist()
            landmarks = face.landmark_2d_106.astype(int).tolist() if hasattr(face, 'landmark_2d_106') else []
            
            return {
                "embedding": embedding,
                "embedding_dim": len(embedding),
                "aligned": False,
                "face_info": {
                    "bbox": bbox,
                    "det_score": float(face.det_score),
                    "landmark_count": len(landmarks)
                }
            }
            
    except Exception as e:
        logger.error(f"Embedding extraction error: {e}")
        raise
