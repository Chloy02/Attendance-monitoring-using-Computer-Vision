# import cv2
# import numpy as np
# import streamlit as st
# from PIL import Image, ImageDraw, ImageFont
# from ultralytics import YOLO
# from emotion_detector import get_dominant_emotion
# from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# # Load your YOLO model once and cache it.
# @st.cache_resource
# def load_yolo_model():
#     return YOLO("yolov8n-face.pt")  # Ensure the weight file is in your working directory.

# model = load_yolo_model()

# # Define a VideoProcessor that uses YOLO and emotion detection.
# class EmotionVideoProcessor(VideoProcessorBase):
#     def __init__(self):
#         # Try loading a professional font. Update font_path as needed.
#         try:
#             self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
#         except Exception as e:
#             st.warning("Custom font not found, using default font.")
#             self.font = ImageFont.load_default()
    
#     def recv(self, frame):
#         # Get the frame as a numpy array in BGR format.
#         img = frame.to_ndarray(format="bgr24")
        
#         # Run YOLO detection.
#         results = model(img, conf=0.25, verbose=False)
        
#         # Convert frame to PIL image for annotation.
#         pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#         draw = ImageDraw.Draw(pil_img)
        
#         # Process each detection result.
#         for result in results:
#             # Depending on your ultralytics version, adapt how you extract bounding boxes.
#             try:
#                 boxes = result.boxes.data.cpu().numpy() if hasattr(result.boxes.data, "cpu") else result.boxes.data.numpy()
#             except Exception as e:
#                 continue
            
#             for box in boxes:
#                 x1, y1, x2, y2, conf, class_id = box
#                 x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
#                 # Draw bounding box.
#                 draw.rectangle([(x1, y1), (x2, y2)], outline="green", width=2)
                
#                 # Crop face from the original BGR image.
#                 face = img[y1:y2, x1:x2]
#                 if face.size != 0:
#                     emotion = get_dominant_emotion(face)
#                 else:
#                     emotion = "Undetected"
                
#                 # Create text to overlay.
#                 text = f"{emotion}"
#                 text_size = draw.textsize(text, font=self.font)
#                 # Draw a filled rectangle as background for the text.
#                 text_background = [(x1, y1 - text_size[1] - 4), (x1 + text_size[0] + 4, y1)]
#                 draw.rectangle(text_background, fill="green")
#                 # Draw the text with clear, professional font.
#                 draw.text((x1 + 2, y1 - text_size[1] - 2), text, fill="white", font=self.font)
        
#         # Convert annotated PIL image back to a numpy array (RGB to BGR).
#         annotated_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
#         return annotated_img

# st.title("Live Face Emotion Detection")
# st.write("This app detects faces and emotions in live video with clear, professional annotations.")

# # Choose input source: live camera stream or video file.
# input_option = st.radio("Select input source:", ("Live Camera", "Upload Video File"))

# if input_option == "Upload Video File":
#     uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])
#     if uploaded_video is not None:
#         # Save uploaded video to disk.
#         video_bytes = uploaded_video.read()
#         with open("temp_video.mp4", "wb") as f:
#             f.write(video_bytes)
#         st.video("temp_video.mp4")
#         webrtc_streamer(key="video", video_url="temp_video.mp4", desired_playing_state=True, video_processor_factory=EmotionVideoProcessor, rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}))
# else:
#     # Live camera feed.
#     st.write("Starting live camera stream...")
#     webrtc_streamer(key="live", video_processor_factory=EmotionVideoProcessor, rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}))
