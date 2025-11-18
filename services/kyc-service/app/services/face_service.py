"""
Enhanced Face Service for KYC Verification
Combining detection, alignment, embedding, and quality checks
"""

import cv2
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis
from typing import List, Dict, Tuple, Optional, Any
import logging
import os

logger = logging.getLogger(__name__)


class FaceService:
    """Complete face processing service for KYC verification"""

    def __init__(self):
        """Initialize insightface with enhanced configuration"""
        try:
            self.app = FaceAnalysis(
                name="buffalo_l", providers=["CPUExecutionProvider"]
            )
            # Use smaller detection size for better performance
            self.app.prepare(ctx_id=-1, det_size=(320, 320), det_thresh=0.5)
            logger.info("FaceService initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize FaceService: {e}")
            raise

    def _pil_to_cv2(self, pil_image: Image.Image) -> np.ndarray:
        """Convert PIL Image to OpenCV format (BGR)"""
        try:
            if pil_image.mode == "RGB":
                return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            elif pil_image.mode == "L":
                return cv2.cvtColor(np.array(pil_image), cv2.COLOR_GRAY2BGR)
            else:
                rgb_image = pil_image.convert("RGB")
                return cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.error(f"Image conversion error: {e}")
            raise

    def _cv2_to_pil(self, cv2_image: np.ndarray) -> Image.Image:
        """Convert OpenCV image (BGR) to PIL Image (RGB)"""
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_image)

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image to improve detection"""
        try:
            # Convert to float32 for processing
            processed = image.astype(np.float32)

            # Normalize to 0-1
            processed /= 255.0

            # Apply contrast enhancement
            processed = np.clip(processed * 1.2, 0, 1)

            # Convert back to uint8
            processed = (processed * 255).astype(np.uint8)

            return processed
        except Exception as e:
            logger.warning(f"Preprocessing failed: {e}")
            return image

    def _remove_duplicate_faces(self, faces: List, iou_threshold: float = 0.5) -> List:
        """Remove duplicate face detections using IoU"""
        if not faces:
            return []

        # Sort by detection score (highest first)
        faces.sort(key=lambda x: getattr(x, "det_score", 0), reverse=True)

        keep_faces = []
        for current_face in faces:
            is_duplicate = False
            current_bbox = current_face.bbox

            for kept_face in keep_faces:
                kept_bbox = kept_face.bbox

                # Calculate IoU
                x1 = max(current_bbox[0], kept_bbox[0])
                y1 = max(current_bbox[1], kept_bbox[1])
                x2 = min(current_bbox[2], kept_bbox[2])
                y2 = min(current_bbox[3], kept_bbox[3])

                intersection = max(0, x2 - x1) * max(0, y2 - y1)
                area_current = (current_bbox[2] - current_bbox[0]) * (
                    current_bbox[3] - current_bbox[1]
                )
                area_kept = (kept_bbox[2] - kept_bbox[0]) * (
                    kept_bbox[3] - kept_bbox[1]
                )
                union = area_current + area_kept - intersection

                iou = intersection / union if union > 0 else 0

                if iou > iou_threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                keep_faces.append(current_face)

        return keep_faces

    def _is_face_centered(self, bbox: List[int], img_w: int, img_h: int) -> bool:
        """Check if face is reasonably centered"""
        face_center_x = (bbox[0] + bbox[2]) / 2
        face_center_y = (bbox[1] + bbox[3]) / 2

        img_center_x = img_w / 2
        img_center_y = img_h / 2

        # Allow 40% deviation from center
        x_deviation = abs(face_center_x - img_center_x) / img_w
        y_deviation = abs(face_center_y - img_center_y) / img_h

        return x_deviation < 0.4 and y_deviation < 0.4

    def _check_brightness(self, image: np.ndarray, bbox: List[int]) -> bool:
        """Check if face region has adequate brightness"""
        x1, y1, x2, y2 = bbox
        face_region = image[y1:y2, x1:x2]

        if face_region.size == 0:
            return False

        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray_face)

        # Acceptable brightness range (0-255)
        return 30 < mean_brightness < 220

    def _check_blur(self, image: np.ndarray, bbox: List[int]) -> bool:
        """Check if face region is not too blurry"""
        x1, y1, x2, y2 = bbox
        face_region = image[y1:y2, x1:x2]

        if face_region.size == 0:
            return False

        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        blur_value = cv2.Laplacian(gray_face, cv2.CV_64F).var()

        # Higher values indicate less blur
        return blur_value > 50

    def validate_face_quality(self, face_data: Dict, image: Image.Image) -> Dict:
        """Validate face quality for KYC verification"""
        bbox = face_data["bbox"]
        landmarks = face_data["landmarks"]

        cv2_image = self._pil_to_cv2(image)
        img_h, img_w = cv2_image.shape[:2]

        # Calculate face size ratio
        face_width = bbox[2] - bbox[0]
        face_height = bbox[3] - bbox[1]
        face_area = face_width * face_height
        image_area = img_w * img_h
        size_ratio = face_area / image_area

        # Quality checks - CONVERT NUMPY BOOLS TO PYTHON BOOLS
        quality_checks = {
            "face_detected": True,
            "face_size_adequate": bool(size_ratio > 0.05),  # Convert to Python bool
            "face_centered": bool(
                self._is_face_centered(bbox, img_w, img_h)
            ),  # Convert to Python bool
            "landmarks_available": bool(len(landmarks) >= 5),  # Convert to Python bool
            "detection_confidence": bool(
                face_data["det_score"] > 0.6
            ),  # Convert to Python bool
            "brightness_ok": bool(
                self._check_brightness(cv2_image, bbox)
            ),  # Convert to Python bool
            "blur_ok": bool(
                self._check_blur(cv2_image, bbox)
            ),  # Convert to Python bool
            "size_ratio": float(round(size_ratio, 3)),  # Convert to Python float
        }

        quality_checks["overall_quality"] = bool(
            all(
                [
                    quality_checks["face_size_adequate"],
                    quality_checks["landmarks_available"],
                    quality_checks["detection_confidence"],
                    quality_checks["brightness_ok"],
                ]
            )
        )

        return quality_checks

    def detect_faces(self, image: Image.Image) -> List[Dict]:
        """
        Enhanced face detection with multiple fallback strategies
        """
        try:
            cv2_image = self._pil_to_cv2(image)
            original_h, original_w = cv2_image.shape[:2]
            logger.info(f"Processing image of size: {original_w}x{original_h}")

            def _run_detection(img_np, description="original"):
                try:
                    faces = self.app.get(img_np)
                    logger.info(f"Detection {description}: found {len(faces)} faces")
                    return faces
                except Exception as e:
                    logger.warning(f"Detection failed for {description}: {e}")
                    return []

            detection_strategies = []

            # Strategy 1: Original image
            detection_strategies.append(("original", cv2_image))

            # Strategy 2: Preprocessed image
            processed_img = self._preprocess_image(cv2_image)
            detection_strategies.append(("processed", processed_img))

            # Strategy 3: Different scales
            scales = [1.5, 2.0, 0.7, 0.5]
            for scale in scales:
                new_w = max(160, int(original_w * scale))
                new_h = max(160, int(original_h * scale))
                resized = cv2.resize(
                    cv2_image, (new_w, new_h), interpolation=cv2.INTER_AREA
                )
                detection_strategies.append((f"scale_{scale}", resized))

            # Strategy 4: Grayscale conversion
            gray_img = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
            gray_3channel = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
            detection_strategies.append(("grayscale", gray_3channel))

            # Strategy 5: Histogram equalization
            lab_img = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2LAB)
            lab_img[:, :, 0] = cv2.equalizeHist(lab_img[:, :, 0])
            equalized_img = cv2.cvtColor(lab_img, cv2.COLOR_LAB2BGR)
            detection_strategies.append(("equalized", equalized_img))

            # Try all strategies
            all_faces = []
            for strategy_name, strategy_img in detection_strategies:
                faces = _run_detection(strategy_img, strategy_name)
                if faces:
                    # Convert faces back to original coordinates if needed
                    if "scale" in strategy_name:
                        scale_factor = 1.0 / float(strategy_name.split("_")[1])
                        for face in faces:
                            face.bbox = face.bbox * scale_factor
                            if hasattr(face, "kps") and face.kps is not None:
                                face.kps = face.kps * scale_factor

                    all_faces.extend(faces)

                    # If we found faces, we can break early
                    if len(all_faces) > 0:
                        break

            # Remove duplicate faces
            if len(all_faces) > 1:
                all_faces = self._remove_duplicate_faces(all_faces)

            if not all_faces:
                logger.warning("No faces detected after all strategies")
                return []

            # Convert to result format
            results = []
            for face in all_faces:
                bbox = face.bbox.astype(int).tolist()
                landmarks_5pt = None

                # Get landmarks
                if hasattr(face, "kps") and face.kps is not None:
                    landmarks_5pt = face.kps.astype(int).tolist()
                elif hasattr(face, "landmark_5") and face.landmark_5 is not None:
                    landmarks_5pt = face.landmark_5.astype(int).tolist()

                # Get 106-point landmarks if available
                landmarks_full = []
                if (
                    hasattr(face, "landmark_2d_106")
                    and face.landmark_2d_106 is not None
                ):
                    landmarks_full = face.landmark_2d_106.astype(int).tolist()

                results.append(
                    {
                        "bbox": bbox,
                        "landmarks": landmarks_5pt if landmarks_5pt else [],
                        "landmarks_full": landmarks_full if landmarks_full else None,
                        "det_score": float(getattr(face, "det_score", 0.0)),
                        "embedding": (
                            face.embedding if hasattr(face, "embedding") else None
                        ),
                    }
                )

            return results

        except Exception as e:
            logger.error(f"Face detection error: {e}")
            return []

    def align_face(
        self,
        image: Image.Image,
        landmarks: List[List[int]],
        output_size: Tuple[int, int] = (112, 112),
    ) -> Image.Image:
        """
        Auto-align face crop for embedding using 5-point landmarks
        """
        try:
            if len(landmarks) < 5:
                raise ValueError("Need at least 5 landmarks for alignment")

            cv2_image = self._pil_to_cv2(image)
            landmarks_np = np.array(landmarks, dtype=np.float32)

            reference_points = np.array(
                [
                    [30.2946, 51.6963],
                    [65.5318, 51.5014],
                    [48.0252, 71.7366],
                    [33.5493, 92.3655],
                    [62.7299, 92.2041],
                ],
                dtype=np.float32,
            )

            reference_points[:, 0] *= output_size[0] / 112.0
            reference_points[:, 1] *= output_size[1] / 112.0

            transform_matrix = cv2.getAffineTransform(
                landmarks_np[:3], reference_points[:3]
            )

            aligned_face = cv2.warpAffine(
                cv2_image,
                transform_matrix,
                output_size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

            return self._cv2_to_pil(aligned_face)

        except Exception as e:
            logger.error(f"Face alignment error: {e}")
            raise

    def get_embeddings(self, aligned_face: Image.Image) -> Dict[str, Any]:
        """
        Extract face embeddings from aligned face image
        """
        try:
            # Convert PIL to cv2 for insightface
            cv2_face = self._pil_to_cv2(aligned_face)

            # Get embedding using insightface
            faces = self.app.get(cv2_face)

            if not faces:
                raise ValueError("No face found in aligned image")

            # Get the first face (should be only one in aligned image)
            face = faces[0]

            if hasattr(face, "embedding") and face.embedding is not None:
                embedding = face.embedding.tolist()
                return {
                    "embedding": embedding,
                    "embedding_dim": len(embedding),
                    "aligned": True,
                    "norm": np.linalg.norm(embedding),
                }
            else:
                raise ValueError("No embedding found for face")

        except Exception as e:
            logger.error(f"Embedding extraction error: {e}")
            raise

    def process_kyc_document(self, image_input: Any) -> Dict[str, Any]:
        """
        Main method to process KYC document - called from main.py
        """
        try:
            # Load image
            if not os.path.exists(image_input):
                return {
                    "status": "error",
                    "error": f"Image file not found: {image_input}",
                    "face_detected": False,
                }

            image = Image.open(image_input)
            logger.info(f"Processing KYC document: {image_input}")

            # Detect faces with enhanced method
            faces = self.detect_faces(image)

            if not faces:
                return {
                    "status": "error",
                    "error": "No face detected in the image",
                    "face_detected": False,
                    "suggestions": [
                        "Ensure face is clearly visible and not obscured",
                        "Improve lighting conditions - avoid shadows on face",
                        "Remove glasses or hat if possible",
                        "Face the camera directly with neutral expression",
                        "Ensure face covers at least 10% of the image",
                    ],
                }

            # Get the best face (highest detection score)
            best_face = max(faces, key=lambda x: x["det_score"])

            # Validate face quality
            quality_checks = self.validate_face_quality(best_face, image)

            if not quality_checks["overall_quality"]:
                return {
                    "status": "error",
                    "error": "Face quality does not meet KYC requirements",
                    "face_detected": True,
                    "quality_checks": quality_checks,
                    "suggestions": self._get_quality_suggestions(quality_checks),
                }

            # Align face and get embeddings
            if len(best_face["landmarks"]) >= 5:
                aligned_face = self.align_face(image, best_face["landmarks"])
                embeddings = self.get_embeddings(aligned_face)

                # Ensure all numeric values are Python native types
                return {
                    "status": "success",
                    "face_detected": True,
                    "face_count": int(len(faces)),  # Convert to Python int
                    "detection_confidence": float(
                        round(best_face["det_score"], 3)
                    ),  # Convert to Python float
                    "quality_checks": quality_checks,
                    "embeddings": embeddings.get("embedding"),
                    "embedding_dim": int(
                        embeddings.get("embedding_dim", 0)
                    ),  # Convert to Python int
                    "aligned": bool(
                        embeddings.get("aligned", True)
                    ),  # Convert to Python bool
                    "embedding_norm": float(
                        embeddings.get("norm", 0.0)
                    ),  # Convert to Python float
                    "bbox": [
                        int(x) for x in best_face["bbox"]
                    ],  # Convert to Python int list
                    "landmarks_count": int(
                        len(best_face["landmarks"])
                    ),  # Convert to Python int
                }
            else:
                return {
                    "status": "error",
                    "error": "Insufficient landmarks for face alignment",
                    "face_detected": True,
                    "landmarks_available": int(
                        len(best_face["landmarks"])
                    ),  # Convert to Python int
                    "suggestions": [
                        "Ensure all facial features are clearly visible",
                        "Try a different image with better lighting",
                        "Make sure face is not at an extreme angle",
                    ],
                }

        except Exception as e:
            logger.error(f"KYC document processing error: {e}")
            return {
                "status": "error",
                "error": f"Processing error: {str(e)}",
                "face_detected": False,
            }

    def compare_faces(
        self, selfie_embedding: List[float], id_photo_embedding: List[float]
    ) -> Dict[str, Any]:
        """
        Compare face embeddings from selfie and ID photo

        Args:
            selfie_embedding: Embedding from selfie photo
            id_photo_embedding: Embedding from ID card photo

        Returns:
            Comparison results with similarity score and verification status
        """
        try:
            if not selfie_embedding or not id_photo_embedding:
                return {
                    "status": "error",
                    "error": "Missing embeddings for comparison",
                    "similarity_score": 0.0,
                    "verified": False,
                }

            selfie_emb = np.array(selfie_embedding)
            id_emb = np.array(id_photo_embedding)

            selfie_emb_norm = selfie_emb / np.linalg.norm(selfie_emb)
            id_emb_norm = id_emb / np.linalg.norm(id_emb)

            similarity_score = np.dot(selfie_emb_norm, id_emb_norm)

            similarity_score = (similarity_score + 1) / 2

            verification_threshold = 0.6  
            verified = similarity_score >= verification_threshold

            if similarity_score >= 0.8:
                confidence = "high"
            elif similarity_score >= 0.6:
                confidence = "medium"
            else:
                confidence = "low"

            return {
                "status": "success",
                "similarity_score": float(similarity_score),
                "verified": bool(verified),
                "confidence": confidence,
                "threshold_used": float(verification_threshold),
                "difference": float(1.0 - similarity_score),
            }

        except Exception as e:
            logger.error(f"Face comparison error: {e}")
            return {
                "status": "error",
                "error": f"Comparison failed: {str(e)}",
                "similarity_score": 0.0,
                "verified": False,
            }

    def extract_id_photo_embedding(self, id_card_image: Image.Image) -> Dict[str, Any]:
        """
        Extract face embedding from ID card photo

        Args:
            id_card_image: PIL Image of ID card

        Returns:
            Embedding and processing results
        """
        try:
            faces = self.detect_faces(id_card_image)

            if not faces:
                return {
                    "status": "error",
                    "error": "No face detected in ID card photo",
                    "face_detected": False,
                }

            best_face = max(faces, key=lambda x: x["det_score"])

            quality_checks = self.validate_face_quality(best_face, id_card_image)

            if not quality_checks["overall_quality"]:
                return {
                    "status": "error",
                    "error": "ID card photo quality does not meet requirements",
                    "face_detected": True,
                    "quality_checks": quality_checks,
                }

            if len(best_face["landmarks"]) >= 5:
                aligned_face = self.align_face(id_card_image, best_face["landmarks"])
                embeddings = self.get_embeddings(aligned_face)

                return {
                    "status": "success",
                    "face_detected": True,
                    "embedding": embeddings.get("embedding"),
                    "embedding_dim": embeddings.get("embedding_dim"),
                    "detection_confidence": float(round(best_face["det_score"], 3)),
                    "quality_checks": quality_checks,
                    "aligned_face": aligned_face,  # Optional: return aligned face for debugging
                }
            else:
                return {
                    "status": "error",
                    "error": "Insufficient landmarks for ID card face alignment",
                    "face_detected": True,
                }

        except Exception as e:
            logger.error(f"ID photo extraction error: {e}")
            return {
                "status": "error",
                "error": f"ID photo processing failed: {str(e)}",
                "face_detected": False,
            }

    def _get_quality_suggestions(self, quality_checks: Dict) -> List[str]:
        """Get specific suggestions based on quality check failures"""
        suggestions = []

        if not quality_checks["face_size_adequate"]:
            suggestions.append(
                "Move closer to camera - face should cover 10-30% of image"
            )

        if not quality_checks["face_centered"]:
            suggestions.append("Position face in the center of the frame")

        if not quality_checks["detection_confidence"]:
            suggestions.append(
                "Improve image quality and ensure face is clearly visible"
            )

        if not quality_checks["brightness_ok"]:
            suggestions.append(
                "Adjust lighting - avoid too dark or too bright conditions"
            )

        if not quality_checks["blur_ok"]:
            suggestions.append("Hold steady while capturing - image is too blurry")

        if not quality_checks["landmarks_available"]:
            suggestions.append("Ensure all facial features are clearly visible")

        return suggestions if suggestions else ["Try again with better image quality"]


# Singleton Pattern
_face_service: Optional[FaceService] = None


def get_face_service() -> FaceService:
    """Get or create FaceService instance"""
    global _face_service
    if _face_service is None:
        _face_service = FaceService()
    return _face_service