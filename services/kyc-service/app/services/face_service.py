"""
Face Detection, Landmarks, Alignment, and Embeddings Service
Uses insightface with RetinaFace detector and ArcFace embeddings
"""
import cv2
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FaceService:
    """Service for face detection, landmarks, alignment, and embeddings using insightface"""
    
    def __init__(self):
        """Initialize insightface with RetinaFace detector and ArcFace embeddings"""
        try:
            self.app = FaceAnalysis(
                name='buffalo_l', 
                providers=['CPUExecutionProvider']  
            )
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("FaceService initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize FaceService: {e}")
            raise
    
    def _pil_to_cv2(self, pil_image: Image.Image) -> np.ndarray:
        """Convert PIL Image to OpenCV format (BGR)"""
        if pil_image.mode == 'RGB':
            return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        elif pil_image.mode == 'L':
           
            return cv2.cvtColor(np.array(pil_image), cv2.COLOR_GRAY2BGR)
        else:
           
            rgb_image = pil_image.convert('RGB')
            return cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR)
    
    def _cv2_to_pil(self, cv2_image: np.ndarray) -> Image.Image:
        """Convert OpenCV image (BGR) to PIL Image (RGB)"""
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_image)
    
    def detect_faces(
        self, 
        image: Image.Image
    ) -> List[Dict]:
        """
        Detect faces and extract 5-point landmarks using RetinaFace detector
        
        Args:
            image: PIL Image
            
        Returns:
            List of face dictionaries with:
            - bbox: [x1, y1, x2, y2] bounding box coordinates
            - landmarks: 5-point landmarks [[x1,y1], [x2,y2], ...]
            - det_score: Detection confidence score
        """
        try:
            cv2_image = self._pil_to_cv2(image)
            faces = self.app.get(cv2_image)
            
            results = []
            for face in faces:
               
                bbox = face.bbox.astype(int).tolist()  
                
                landmarks_5pt = None
                landmarks_full = []
                

                if hasattr(face, 'landmark_5') and face.landmark_5 is not None:
                    landmarks_5pt = face.landmark_5.astype(int).tolist()
                elif hasattr(face, 'kps') and face.kps is not None:
                   
                     landmarks_5pt = face.kps.astype(int).tolist()
                
                if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
                    landmarks_full = face.landmark_2d_106.astype(int).tolist()
                
                if landmarks_5pt is None and len(landmarks_full) >= 106:
                    # Extract 5 key points from 106-point landmarks
                    # Standard indices for buffalo_l model (106 points)
                    key_landmarks = [
                        landmarks_full[38],   # left eye center
                        landmarks_full[88],   # right eye center
                        landmarks_full[86],   # nose tip
                        landmarks_full[61],   # left mouth corner
                        landmarks_full[84],   # right mouth corner
                    ]
                    landmarks_5pt = key_landmarks
                elif landmarks_5pt is None and len(landmarks_full) > 0:
                    # Fallback: use first 5 points if available
                    landmarks_5pt = landmarks_full[:5] if len(landmarks_full) >= 5 else landmarks_full
                elif landmarks_5pt is None:
                    landmarks_5pt = []
                
                results.append({
                    "bbox": bbox,
                    "landmarks": landmarks_5pt,
                    "landmarks_full": landmarks_full if landmarks_full else None,
                    "det_score": float(face.det_score),
                    "landmark_count": len(landmarks_full) if landmarks_full else len(landmarks_5pt) if landmarks_5pt else 0
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Face detection error: {e}")
            raise
    



    def align_face(
        self, 
        image: Image.Image,
        landmarks: List[List[int]],
        output_size: Tuple[int, int] = (112, 112)
    ) -> Image.Image:
        """
        Auto-align face crop for embedding using 5-point landmarks
        
        Args:
            image: PIL Image
            landmarks: 5-point landmarks [[x1,y1], [x2,y2], ...]
            output_size: Output image size (width, height)
            
        Returns:
            Aligned face image as PIL Image
        """
        try:
            if len(landmarks) < 5:
                raise ValueError("Need at least 5 landmarks for alignment")
            
            cv2_image = self._pil_to_cv2(image)
            landmarks_np = np.array(landmarks, dtype=np.float32)
            
            reference_points = np.array([
                [30.2946, 51.6963],  # left eye
                [65.5318, 51.5014],  # right eye
                [48.0252, 71.7366],  # nose tip
                [33.5493, 92.3655],  # left mouth corner
                [62.7299, 92.2041],  # right mouth corner
            ], dtype=np.float32)
            
            reference_points[:, 0] *= (output_size[0] / 112.0)
            reference_points[:, 1] *= (output_size[1] / 112.0)
            
            transform_matrix = cv2.getAffineTransform(
                landmarks_np[:3],  
                reference_points[:3]
            )
            
            aligned_face = cv2.warpAffine(
                cv2_image,
                transform_matrix,
                output_size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            
            return self._cv2_to_pil(aligned_face)
            
        except Exception as e:
            logger.error(f"Face alignment error: {e}")
            raise
    

    def fine_tune_face(
        self,
        image: Image.Image,
        output_size: Tuple[int, int] = (112, 112)
    ) -> Image.Image:
        """
        Combined method: Detect faces and align the first detected face.
        Calls both detect_faces and align_face methods.
        
        Args:
            image: PIL Image
            output_size: Output size for aligned face (width, height)
            
        Returns:
            Aligned face image as PIL Image
        """
        try:
            # Detect faces
            detected_faces = self.detect_faces(image)
            
            if not detected_faces:
                raise ValueError("No faces detected in image")
            
            # Get landmarks from first detected face
            landmarks = detected_faces[0]["landmarks"]
            
            if len(landmarks) < 5:
                raise ValueError("Insufficient landmarks for alignment")
            
            # Align the face
            fine_tuned_face = self.align_face(image, landmarks, output_size)
            
            return fine_tuned_face

        except Exception as e:
            logger.error(f"Face fine-tuning error: {e}")
            raise



_face_service: Optional[FaceService] = None


def get_face_service() -> FaceService:
    """Get or create FaceService instance (singleton pattern)"""
    global _face_service
    if _face_service is None:
        _face_service = FaceService()
    return _face_service

