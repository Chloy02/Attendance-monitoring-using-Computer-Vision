import cv2
import numpy as np
from deepface import DeepFace

def get_dominant_emotion(face_image):
    """
    Enhances the input face image and uses DeepFace's emotion analysis to determine
    the dominant emotion. If multiple faces are detected within the crop, it returns
    a comma-separated list of dominant emotions.

    :param face_image: A NumPy array (BGR format) containing the cropped face.
    :return: A string representing the dominant emotion (or comma-separated emotions),
             or "Undetected" if an error occurs.
    """
    try:
        # Validate input
        if face_image is None or face_image.size == 0:
            return "Undetected"

        # Convert from BGR to RGB (DeepFace expects RGB)
        rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        
        # Enhanced preprocessing: convert to YUV to equalize the luminance channel
        yuv_face = cv2.cvtColor(rgb_face, cv2.COLOR_RGB2YUV)
        # Equalize histogram on the Y channel to improve contrast without losing color info
        yuv_face[:,:,0] = cv2.equalizeHist(yuv_face[:,:,0])
        # Convert back to RGB
        enhanced_face = cv2.cvtColor(yuv_face, cv2.COLOR_YUV2RGB)
        
        # Resize to a fixed size (224x224) for consistency
        enhanced_face = cv2.resize(enhanced_face, (224, 224))

        # Use DeepFace to analyze emotion.
        # Note: For emotion analysis, DeepFace uses its dedicated emotion model.
        # Although the "Facenet512" model is available for face recognition, it isn’t applicable for emotion analysis.
        result = DeepFace.analyze(
            img_path=enhanced_face,
            actions=['emotion'],
            enforce_detection=False,
            detector_backend='retinaface'
        )

        # If multiple faces are detected, result may be a list.
        if isinstance(result, list):
            emotions = [r.get('dominant_emotion', 'Undetected') for r in result]
            return ', '.join(emotions)
        else:
            return result.get('dominant_emotion', 'Undetected')
    except Exception as e:
        print(f"Emotion detection error: {e}")
        return "Undetected"


















# import cv2
# import numpy as np
# from deepface import DeepFace

# def get_dominant_emotion(face_image):
#     """
#     Enhances the input face image and uses DeepFace's emotion analysis to determine
#     the dominant emotion. If multiple faces are detected within the crop, it returns
#     a comma-separated list of dominant emotions.

#     :param face_image: A NumPy array (BGR format) containing the cropped face.
#     :return: A string representing the dominant emotion (or comma-separated emotions),
#              or "Undetected" if an error occurs.
#     """
#     try:
#         # Check for empty input
#         if face_image is None or face_image.size == 0:
#             return "Undetected"

#         # Preprocessing steps:
#         # Convert from BGR to RGB (DeepFace expects RGB)
#         rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
#         # Optional: convert to grayscale and equalize histogram to improve contrast
#         gray_face = cv2.cvtColor(rgb_face, cv2.COLOR_RGB2GRAY)
#         equalized_face = cv2.equalizeHist(gray_face)
#         # Convert back to RGB (3 channels)
#         enhanced_face = cv2.cvtColor(equalized_face, cv2.COLOR_GRAY2RGB)
#         # Resize to a fixed size for consistency (e.g., 224x224)
#         enhanced_face = cv2.resize(enhanced_face, (224, 224))

#         # Run DeepFace emotion analysis using a valid detector backend.
#         # Note: "facenet512" is not a valid detector_backend. We'll use "retinaface".
#         result = DeepFace.analyze(
#             img_path=enhanced_face,
#             actions=['emotion'],
#             enforce_detection=False,
#             detector_backend='retinaface'
#         )

#         # If multiple faces are detected, result may be a list.
#         if isinstance(result, list):
#             emotions = [r.get('dominant_emotion', 'Undetected') for r in result]
#             return ', '.join(emotions)
#         else:
#             return result.get('dominant_emotion', 'Undetected')
#     except Exception as e:
#         print(f"Emotion detection error: {e}")
#         return "Undetected"
