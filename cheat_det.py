import cv2
import sys
from ultralytics import YOLO
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def main(source):
    # Adjust the model_path to where you saved your weights locally.
    # For example: "model_runs/detect/train/weights/best.pt"
    model_path = "/Users/christina/Downloads/best-2.pt"

    # Set device to 'mps' to use Apple Silicon GPU if supported (or 'cpu' if not)
    device = 'mps'

    # Load the YOLO model with your custom weights
    model = YOLO(model_path)
    
    # Open video capture: 0 for webcam, or a video file path.
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source: {source}")
        sys.exit(1)
    
    print("Press 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No more frames or error encountered.")
            break
        
        # Run inference on the current frame.
        # Adjust conf (confidence threshold) as needed.
        results = model.predict(frame, conf=0.25, device=device, verbose=False)
        
        # Get the annotated frame (the first result if multiple inputs)
        annotated_frame = results[0].plot()
        
        # Display the annotated frame
        cv2.imshow("YOLO Inference", annotated_frame)
        
        # Exit if 'q' is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # By default, use the webcam (source = 0). 
    # To use a video file, pass its path as a command-line argument.
    # Example usage:
    #   python your_script.py "path/to/video.mp4"
    if len(sys.argv) > 1:
        source_input = sys.argv[1]
        # If the input is a digit, convert it to an integer
        source = int(source_input) if source_input.isdigit() else source_input
    else:
        source = 0  # default webcam

    main(source)













# import cv2
# import mediapipe as mp
# import numpy as np
# import time
# from ultralytics import YOLO
# from scipy.optimize import linear_sum_assignment

# # =====================================
# # 1. Parameters & Initialization
# # =====================================

# # Sliding window size (frames)
# WINDOW_SIZE = 30  

# # Thresholds for the simple heuristics
# MAR_THRESHOLD = 0.3             # If mouth aspect ratio > 0.3 => "talking/suspicious"
# HEAD_RATIO_DIFF_THRESHOLD = 0.2 # If abs(head_ratio - 1) > 0.2 => "looking around"

# # Suspicious frames needed within the last WINDOW_SIZE frames to flag "cheating"
# SUSPICIOUS_FRAME_THRESHOLD = 15

# # Load YOLOv8n for person detection
# yolo_model = YOLO("yolov8n.pt")  # Make sure you have yolov8n.pt in the same folder or specify full path

# # Initialize MediaPipe FaceMesh for face landmarks
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(
#     static_image_mode=False,
#     max_num_faces=5,           # Increase if you expect multiple faces in one bounding box
#     refine_landmarks=True,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# # =====================================
# # 2. Centroid Tracker (Multi-Person)
# # =====================================
# class CentroidTracker:
#     """
#     Keeps track of objects using their centroid + bounding boxes
#     and assigns unique IDs across frames.
#     """
#     def __init__(self, max_disappeared=15):
#         self.nextObjectID = 0
#         self.objects = {}       # objectID -> (centroid, bbox)
#         self.disappeared = {}   # objectID -> frames disappeared
#         self.max_disappeared = max_disappeared

#     def register(self, centroid, bbox):
#         self.objects[self.nextObjectID] = (centroid, bbox)
#         self.disappeared[self.nextObjectID] = 0
#         self.nextObjectID += 1

#     def deregister(self, objectID):
#         del self.objects[objectID]
#         del self.disappeared[objectID]

#     def update(self, rects):
#         """
#         rects: list of bounding boxes [ (x1, y1, x2, y2), ... ]
#         """
#         if len(rects) == 0:
#             # Increase 'disappeared' for all existing objects
#             for objectID in list(self.disappeared.keys()):
#                 self.disappeared[objectID] += 1
#                 if self.disappeared[objectID] > self.max_disappeared:
#                     self.deregister(objectID)
#             return self.objects

#         # Compute centroids of new detections
#         inputCentroids = []
#         for (x1, y1, x2, y2) in rects:
#             cX = int((x1 + x2) / 2.0)
#             cY = int((y1 + y2) / 2.0)
#             inputCentroids.append((cX, cY))

#         # If no existing objects, register all
#         if len(self.objects) == 0:
#             for i in range(len(inputCentroids)):
#                 self.register(inputCentroids[i], rects[i])
#         else:
#             # Get current object IDs and their centroids
#             objectIDs = list(self.objects.keys())
#             objectCentroids = [self.objects[oid][0] for oid in objectIDs]

#             # Compute distances between each pair (old, new)
#             D = np.linalg.norm(
#                 np.array(objectCentroids)[:, np.newaxis] - np.array(inputCentroids),
#                 axis=2
#             )

#             # Hungarian assignment
#             rows, cols = linear_sum_assignment(D)

#             assignedCols = set()
#             usedRows = set()

#             # Update matched objects
#             for (row, col) in zip(rows, cols):
#                 objectID = objectIDs[row]
#                 self.objects[objectID] = (inputCentroids[col], rects[col])
#                 self.disappeared[objectID] = 0
#                 assignedCols.add(col)
#                 usedRows.add(row)

#             # Unmatched existing objects
#             unmatchedObjectIDs = set(range(len(objectIDs))) - usedRows
#             for row in unmatchedObjectIDs:
#                 objectID = objectIDs[row]
#                 self.disappeared[objectID] += 1
#                 if self.disappeared[objectID] > self.max_disappeared:
#                     self.deregister(objectID)

#             # Unmatched new detections
#             unmatchedCols = set(range(len(inputCentroids))) - assignedCols
#             for col in unmatchedCols:
#                 self.register(inputCentroids[col], rects[col])

#         return self.objects

# # =====================================
# # 3. Feature Extraction (Face Landmarks)
# # =====================================
# def compute_face_features(face_landmarks):
#     """
#     Returns (head_ratio, mouth_aspect_ratio).
#     head_ratio ~ nose->left_eye / nose->right_eye
#     mouth_aspect_ratio (MAR) ~ upper_lip->lower_lip / left_eye->right_eye
#     """
#     # MediaPipe indices
#     # Nose tip=1, Left eye outer corner=33, Right eye outer corner=263,
#     # Upper lip=13, Lower lip=14
#     try:
#         nose = face_landmarks.landmark[1]
#         left_eye = face_landmarks.landmark[33]
#         right_eye = face_landmarks.landmark[263]
#         upper_lip = face_landmarks.landmark[13]
#         lower_lip = face_landmarks.landmark[14]
#     except IndexError:
#         return None

#     # Compute distances in 2D (x,y)
#     d_left = np.linalg.norm([nose.x - left_eye.x, nose.y - left_eye.y])
#     d_right = np.linalg.norm([nose.x - right_eye.x, nose.y - right_eye.y])
#     head_ratio = d_left / (d_right + 1e-6)

#     mouth_dist = np.linalg.norm([upper_lip.x - lower_lip.x, upper_lip.y - lower_lip.y])
#     eye_dist = np.linalg.norm([left_eye.x - right_eye.x, left_eye.y - right_eye.y])
#     mar = mouth_dist / (eye_dist + 1e-6)

#     return (head_ratio, mar)

# # =====================================
# # 4. Heuristic Classification
# # =====================================

# # We keep a buffer of the last WINDOW_SIZE 'suspicious' flags for each person.
# # "Suspicious" if either talking or looking_around.
# person_suspicious_buffers = {}  # objectID -> deque/list of booleans (size up to WINDOW_SIZE)

# def check_suspicious(head_ratio, mar):
#     """
#     Return True if either:
#       - mar > MAR_THRESHOLD (talking)
#       - abs(head_ratio - 1.0) > HEAD_RATIO_DIFF_THRESHOLD (looking around)
#     """
#     talking = (mar > MAR_THRESHOLD)
#     looking_around = (abs(head_ratio - 1.0) > HEAD_RATIO_DIFF_THRESHOLD)
#     return talking or looking_around

# def get_cheating_label(objectID):
#     """
#     Check how many 'True' in the last WINDOW_SIZE frames.
#     If above SUSPICIOUS_FRAME_THRESHOLD => "cheating"
#     else "not_cheating"
#     """
#     # Count how many 'True'
#     susp_count = sum(person_suspicious_buffers[objectID])
#     if susp_count > SUSPICIOUS_FRAME_THRESHOLD:
#         return "cheating"
#     else:
#         return "not_cheating"

# # Optional: store cheat log
# cheating_log = []  # (objectID, timestamp)

# # =====================================
# # 5. Main Loop (No Training Required)
# # =====================================
# def main():
#     cap = cv2.VideoCapture(0)  # or a video file path

#     tracker = CentroidTracker(max_disappeared=15)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Could not read from camera. Exiting...")
#             break

#         frame = cv2.flip(frame, 1)  # mirror the webcam
#         orig_frame = frame.copy()
#         H, W = frame.shape[:2]

#         # -------- Step 1: Detect People with YOLOv8n --------
#         results = yolo_model(frame, verbose=False)[0]
#         boxes = []
#         for box in results.boxes:
#             cls_id = int(box.cls[0])
#             if cls_id == 0:  # "person"
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 x1, y1 = max(0, x1), max(0, y1)
#                 x2, y2 = min(W, x2), min(H, y2)
#                 boxes.append((x1, y1, x2, y2))

#         # -------- Step 2: Update Tracker --------
#         tracked_objects = tracker.update(boxes)

#         # -------- Step 3: For Each Tracked Person, Extract Face Features & Apply Heuristics --------
#         for objectID, (centroid, bbox) in tracked_objects.items():
#             x1, y1, x2, y2 = bbox

#             # Initialize suspicious buffer for new object
#             if objectID not in person_suspicious_buffers:
#                 person_suspicious_buffers[objectID] = []

#             # Draw a placeholder bounding box (gray)
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 1)
#             cv2.putText(frame, f"ID {objectID}", (x1, y1 - 5),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

#             roi = orig_frame[y1:y2, x1:x2]
#             if roi.size == 0:
#                 # If the ROI is empty
#                 continue

#             # Convert ROI to RGB for MediaPipe
#             roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
#             face_results = face_mesh.process(roi_rgb)

#             # Extract face features
#             if face_results.multi_face_landmarks:
#                 # Use only the first face in the ROI
#                 face_landmarks = face_results.multi_face_landmarks[0]
#                 feats = compute_face_features(face_landmarks)

#                 if feats is not None:
#                     head_ratio, mar = feats
#                     # Check suspicious
#                     is_suspicious = check_suspicious(head_ratio, mar)
#                     person_suspicious_buffers[objectID].append(is_suspicious)

#                     # Limit buffer size
#                     if len(person_suspicious_buffers[objectID]) > WINDOW_SIZE:
#                         person_suspicious_buffers[objectID] = person_suspicious_buffers[objectID][-WINDOW_SIZE:]

#                     # Decide "cheating" or "not_cheating"
#                     label = get_cheating_label(objectID)
#                     if label == "cheating":
#                         box_color = (0, 0, 255)      # Red
#                         text_color = (255, 255, 255)
#                         # Log if we want
#                         # If you'd like to log every frame, do it here. Typically we log on transition from not_cheating -> cheating
#                         # cheating_log.append((objectID, time.strftime("%Y-%m-%d %H:%M:%S")))
#                     else:
#                         box_color = (0, 255, 0)      # Green
#                         text_color = (255, 255, 255)

#                     # Draw final bounding box
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
#                     cv2.putText(frame, label, (x1, y1 - 10),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
#                 else:
#                     # Landmark extraction failed
#                     cv2.putText(frame, "No face features", (x1, y2 + 20),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
#             else:
#                 # No face detected
#                 cv2.putText(frame, "No face", (x1, y2 + 20),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

#         # -------- Step 4: Show the Frame --------
#         cv2.imshow("Heuristic Cheating Detection (Press 'q' to exit)", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Cleanup
#     cap.release()
#     face_mesh.close()
#     cv2.destroyAllWindows()

#     # Print the log if you decided to fill it in
#     if len(cheating_log) > 0:
#         print("\nCheating Log (objectID, timestamp):")
#         for entry in cheating_log:
#             print(entry)
#     else:
#         print("\nNo cheating events logged in the code above.")

# if __name__ == "__main__":
#     main()












# import cv2
# import mediapipe as mp
# import numpy as np
# import tensorflow as tf
# import time
# from ultralytics import YOLO
# from scipy.optimize import linear_sum_assignment

# # ====================================
# # 1. Parameters & Initialization
# # ====================================
# WINDOW_SIZE = 30  # Number of consecutive frames to form a window
# FEATURE_DIM = 4   # Example: (head_ratio, mar, mouth_movement, gaze_angle) - you can define more

# # YOLO Person Detector
# yolo_model = YOLO("yolov8n.pt")  # Make sure 'ultralytics' is installed

# # MediaPipe FaceMesh or Holistic initialization
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(
#     static_image_mode=False,
#     max_num_faces=10,
#     refine_landmarks=True,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# # (Optional) If you want upper-body or full pose, you could do:
# # mp_pose = mp.solutions.pose
# # pose = mp_pose.Pose(...)

# # Load or create an LSTM model. We'll do a multi-class dummy approach if no model is found.
# NUM_CLASSES = 4  # e.g. 0=not_cheating, 1=talking, 2=looking_around, 3=cheating
# label_map = {0: "not_cheating", 1: "talking", 2: "looking_around", 3: "cheating"}

# try:
#     lstm_model = tf.keras.models.load_model("multi_class_lstm_model.h5")
#     use_lstm = True
#     print("Loaded LSTM model successfully.")
# except:
#     print("No multi-class LSTM model found. Using dummy classifier.")
#     use_lstm = False

# # ====================================
# # 2. Centroid Tracker (for multi-person)
# # ====================================
# class CentroidTracker:
#     def __init__(self, max_disappeared=15):
#         self.nextObjectID = 0
#         # objectID -> (centroid, bbox)
#         self.objects = {}
#         # objectID -> number of consecutive frames missing
#         self.disappeared = {}
#         self.max_disappeared = max_disappeared

#     def register(self, centroid, bbox):
#         self.objects[self.nextObjectID] = (centroid, bbox)
#         self.disappeared[self.nextObjectID] = 0
#         self.nextObjectID += 1

#     def deregister(self, objectID):
#         del self.objects[objectID]
#         del self.disappeared[objectID]

#     def update(self, rects):
#         # rects: list of bounding boxes (x1, y1, x2, y2)
#         if len(rects) == 0:
#             # increment disappeared count for existing objects
#             for objectID in list(self.disappeared.keys()):
#                 self.disappeared[objectID] += 1
#                 if self.disappeared[objectID] > self.max_disappeared:
#                     self.deregister(objectID)
#             return self.objects

#         # compute input centroids
#         inputCentroids = []
#         for (x1, y1, x2, y2) in rects:
#             cX = int((x1 + x2) / 2.0)
#             cY = int((y1 + y2) / 2.0)
#             inputCentroids.append((cX, cY))

#         # if no existing objects, register all
#         if len(self.objects) == 0:
#             for i in range(len(inputCentroids)):
#                 self.register(inputCentroids[i], rects[i])
#         else:
#             objectIDs = list(self.objects.keys())
#             objectCentroids = [self.objects[oid][0] for oid in objectIDs]

#             # compute distance matrix between objectCentroids and inputCentroids
#             D = np.linalg.norm(np.array(objectCentroids)[:, np.newaxis] - np.array(inputCentroids), axis=2)

#             # Hungarian (Munkres) assignment
#             rows, cols = linear_sum_assignment(D)
#             assignedCols = set()

#             for (row, col) in zip(rows, cols):
#                 objectID = objectIDs[row]
#                 self.objects[objectID] = (inputCentroids[col], rects[col])
#                 self.disappeared[objectID] = 0
#                 assignedCols.add(col)

#             # unmatched objects
#             unmatchedObjectIDs = set(range(len(objectIDs))) - set(rows)
#             for idx in unmatchedObjectIDs:
#                 objectID = objectIDs[idx]
#                 self.disappeared[objectID] += 1
#                 if self.disappeared[objectID] > self.max_disappeared:
#                     self.deregister(objectID)

#             # unmatched new detections
#             for i in range(len(inputCentroids)):
#                 if i not in assignedCols:
#                     self.register(inputCentroids[i], rects[i])

#         return self.objects

# # ====================================
# # 3. Feature Extraction
# # ====================================
# def compute_face_features(face_landmarks):
#     """
#     Example features:
#     1) head_ratio       = ratio of nose->left_eye / nose->right_eye
#     2) mouth_aspect     = mouth aspect ratio
#     3) mouth_movement   = difference from previous mouth position (placeholder)
#     4) gaze_angle       = rough 'angle' of the head or eye gaze (placeholder)
#     """
#     try:
#         # Use MediaPipe Landmarks:
#         # Nose Tip: index=1
#         # Left Eye Outer Corner: 33
#         # Right Eye Outer Corner: 263
#         # Upper Lip: 13
#         # Lower Lip: 14
#         nose = face_landmarks.landmark[1]
#         left_eye = face_landmarks.landmark[33]
#         right_eye = face_landmarks.landmark[263]
#         upper_lip = face_landmarks.landmark[13]
#         lower_lip = face_landmarks.landmark[14]
#     except IndexError:
#         return None

#     # Head ratio
#     d_left = np.linalg.norm([nose.x - left_eye.x, nose.y - left_eye.y])
#     d_right = np.linalg.norm([nose.x - right_eye.x, nose.y - right_eye.y])
#     head_ratio = d_left / (d_right + 1e-5)

#     # Mouth aspect ratio (MAR)
#     mouth_distance = np.linalg.norm([upper_lip.x - lower_lip.x, upper_lip.y - lower_lip.y])
#     eye_distance = np.linalg.norm([left_eye.x - right_eye.x, left_eye.y - right_eye.y])
#     mar = mouth_distance / (eye_distance + 1e-5)

#     # Dummy placeholders: in practice you'd measure actual mouth movement or gaze angle
#     mouth_movement = mar  # For example, you might track difference from the previous frame
#     gaze_angle = head_ratio  # Or you’d compute real angles w.r.t. nose or shoulders

#     return [head_ratio, mar, mouth_movement, gaze_angle]

# # ====================================
# # 4. Multi-Class Classification
# # ====================================
# def classify_behavior(feature_sequence):
#     """
#     Input: feature_sequence shape = [WINDOW_SIZE, FEATURE_DIM]
#     Output: (label, confidence)
#       - label in { "not_cheating", "talking", "looking_around", "cheating" }
#       - confidence in [0,1]
#     """
#     sequence_np = np.expand_dims(np.array(feature_sequence), axis=0)  # shape (1, window_size, feature_dim)

#     if use_lstm:
#         # Real multi-class model
#         # E.g. model output shape: (batch_size, NUM_CLASSES)
#         preds = lstm_model.predict(sequence_np)
#         # pred vector: [p_not_cheating, p_talking, p_looking_around, p_cheating]
#         class_idx = int(np.argmax(preds[0]))
#         confidence = float(preds[0][class_idx])
#         label = label_map[class_idx]
#         return label, confidence
#     else:
#         # Dummy multi-class logic
#         # For instance, if mouth_aspect is large => "talking",
#         # if ratio is certain range => "looking_around", etc.
#         # Or pick "cheating" if mar is extremely large
#         feats = np.mean(feature_sequence, axis=0)  # average across the window
#         head_ratio_avg, mar_avg, mouth_move_avg, gaze_angle_avg = feats

#         # Example (completely made up) heuristics:
#         if mar_avg > 0.35:
#             return "talking", 0.8
#         elif abs(gaze_angle_avg - 1.0) > 0.2:
#             return "looking_around", 0.7
#         elif mar_avg > 0.3 and head_ratio_avg < 0.8:
#             return "cheating", 0.9
#         else:
#             return "not_cheating", 0.6

# # ====================================
# # 5. Temporal Buffers & Logging
# # ====================================
# person_buffers = {}  # objectID -> list of feature vectors
# behavior_log = []    # (objectID, timestamp, label)

# # ====================================
# # 6. Main Video Loop
# # ====================================
# cap = cv2.VideoCapture(0)  # or video file path

# tracker = CentroidTracker(max_disappeared=15)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = cv2.flip(frame, 1)  # mirror view
#     orig_frame = frame.copy()
#     H, W = frame.shape[:2]

#     # ---- 6.1: Person Detection via YOLO ----
#     results = yolo_model(frame, verbose=False)[0]
#     boxes = []
#     for box in results.boxes:
#         cls_id = int(box.cls[0])
#         if cls_id == 0:  # "person" class
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             boxes.append((x1, y1, x2, y2))

#     # ---- 6.2: Multi-Person Tracking ----
#     objects = tracker.update(boxes)

#     # ---- 6.3: Extract Features & Classify for Each Tracked Person ----
#     for objectID, (centroid, bbox) in objects.items():
#         (x1, y1, x2, y2) = bbox
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 2)
#         cv2.putText(frame, f"ID {objectID}", (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)

#         # Crop ROI for face analysis
#         roi = orig_frame[y1:y2, x1:x2]
#         if roi.size == 0:
#             continue

#         roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
#         face_results = face_mesh.process(roi_rgb)

#         features = None
#         if face_results.multi_face_landmarks:
#             # Assume first face in ROI
#             f_landmarks = face_results.multi_face_landmarks[0]
#             features = compute_face_features(f_landmarks)

#         if features is not None:
#             if objectID not in person_buffers:
#                 person_buffers[objectID] = []
#             person_buffers[objectID].append(features)

#             # Keep buffer size fixed
#             if len(person_buffers[objectID]) > WINDOW_SIZE:
#                 person_buffers[objectID] = person_buffers[objectID][-WINDOW_SIZE:]

#             # If buffer is full, classify
#             if len(person_buffers[objectID]) == WINDOW_SIZE:
#                 label, confidence = classify_behavior(person_buffers[objectID])

#                 # Color logic
#                 if label == "cheating":
#                     box_color = (0, 0, 255)    # Red
#                 elif label == "talking":
#                     box_color = (0, 255, 255) # Yellowish
#                 elif label == "looking_around":
#                     box_color = (255, 0, 255) # Magenta
#                 else:
#                     box_color = (0, 255, 0)   # Green for not_cheating

#                 # Draw bounding box
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
#                 cv2.putText(frame, f"{label} ({confidence:.2f})",
#                             (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX,
#                             0.6, box_color, 2)

#                 # Log if cheating or up to you
#                 # Here, we log every classification
#                 behavior_log.append((objectID, time.strftime("%Y-%m-%d %H:%M:%S"), label))
#         else:
#             # If no face
#             cv2.putText(frame, "No face",
#                         (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX,
#                         0.6, (0, 255, 255), 2)

#     # ---- 6.4: Display ----
#     cv2.imshow("Multi-Person Cheating Detection", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# face_mesh.close()
# cv2.destroyAllWindows()

# # ====================================
# # 7. Print Behavior Log
# # ====================================
# print("Behavior Log (objectID, timestamp, label):")
# for entry in behavior_log:
#     print(entry)














# import cv2
# import mediapipe as mp
# import math

# # -------------------------------
# # Helper Function: Euclidean Distance
# # -------------------------------
# def calculate_distance(point1, point2):
#     return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# # -------------------------------
# # 1. Initialize MediaPipe Modules
# # -------------------------------
# mp_face_mesh = mp.solutions.face_mesh
# mp_pose = mp.solutions.pose

# # Configure FaceMesh: 
# #   - static_image_mode=False for video stream processing,
# #   - max_num_faces=1 (assuming one subject),
# #   - refine_landmarks=True to get extra details (for mouth detection)
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
#                                   max_num_faces=1,
#                                   refine_landmarks=True,
#                                   min_detection_confidence=0.5,
#                                   min_tracking_confidence=0.5)

# # Configure Pose
# pose = mp_pose.Pose(static_image_mode=False,
#                     min_detection_confidence=0.5,
#                     min_tracking_confidence=0.5)

# # -------------------------------
# # 2. Start Video Capture (Webcam)
# # -------------------------------
# cap = cv2.VideoCapture(0)  # Change 0 to video file path if needed

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Flip the frame for a mirror view (optional)
#     frame = cv2.flip(frame, 1)
#     h, w, _ = frame.shape

#     # Convert frame color for MediaPipe
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Process with FaceMesh and Pose
#     results_face = face_mesh.process(frame_rgb)
#     results_pose = pose.process(frame_rgb)

#     # Initialize flags for behavior detection
#     head_turn_flag = False
#     talking_flag = False
#     leaning_flag = False

#     # -------------------------------
#     # 3. FaceMesh Processing for Head & Mouth Analysis
#     # -------------------------------
#     if results_face.multi_face_landmarks:
#         for face_landmarks in results_face.multi_face_landmarks:
#             # Get landmark positions in normalized coordinates
#             # Using recommended landmark indices:
#             #   - Nose Tip: index 1
#             #   - Left Eye Outer Corner: index 33
#             #   - Right Eye Outer Corner: index 263
#             #   - Upper Lip: index 13
#             #   - Lower Lip: index 14
#             nose = face_landmarks.landmark[1]
#             left_eye = face_landmarks.landmark[33]
#             right_eye = face_landmarks.landmark[263]
#             upper_lip = face_landmarks.landmark[13]
#             lower_lip = face_landmarks.landmark[14]

#             # Convert normalized coordinates to pixels
#             nose_coord = (int(nose.x * w), int(nose.y * h))
#             left_eye_coord = (int(left_eye.x * w), int(left_eye.y * h))
#             right_eye_coord = (int(right_eye.x * w), int(right_eye.y * h))
#             upper_lip_coord = (int(upper_lip.x * w), int(upper_lip.y * h))
#             lower_lip_coord = (int(lower_lip.x * w), int(lower_lip.y * h))

#             # OPTIONAL: Draw key landmarks on frame for visualization
#             cv2.circle(frame, nose_coord, 2, (0, 255, 0), -1)
#             cv2.circle(frame, left_eye_coord, 2, (0, 255, 0), -1)
#             cv2.circle(frame, right_eye_coord, 2, (0, 255, 0), -1)
#             cv2.circle(frame, upper_lip_coord, 2, (0, 255, 0), -1)
#             cv2.circle(frame, lower_lip_coord, 2, (0, 255, 0), -1)

#             # --- Head Turning Detection ---
#             # Heuristic: Compare distance from nose to each eye.
#             d_left = calculate_distance((nose.x, nose.y), (left_eye.x, left_eye.y))
#             d_right = calculate_distance((nose.x, nose.y), (right_eye.x, right_eye.y))
#             ratio = d_left / (d_right + 1e-5)  # avoid division by zero

#             # If ratio deviates significantly from 1, assume head is turned.
#             if ratio < 0.8 or ratio > 1.2:
#                 head_turn_flag = True

#             # --- Talking Detection (Mouth Open) ---
#             # Compute mouth aspect ratio (MAR) using vertical distance between lips,
#             # normalized by the eye distance.
#             mouth_distance = calculate_distance((upper_lip.x, upper_lip.y), (lower_lip.x, lower_lip.y))
#             eye_distance = calculate_distance((left_eye.x, left_eye.y), (right_eye.x, right_eye.y))
#             if eye_distance > 0:
#                 mar = mouth_distance / eye_distance
#                 # If MAR exceeds threshold (e.g., 0.3), assume mouth is open (talking)
#                 if mar > 0.3:
#                     talking_flag = True

#     # -------------------------------
#     # 4. Pose Processing for Leaning Detection
#     # -------------------------------
#     if results_pose.pose_landmarks:
#         landmarks = results_pose.pose_landmarks.landmark
#         # Get left and right shoulder landmarks (indices from MediaPipe Pose)
#         left_shoulder = landmarks[11]  # left shoulder
#         right_shoulder = landmarks[12]  # right shoulder
#         # Use nose from Pose as an alternative head position (landmark 0)
#         nose_pose = landmarks[0]

#         # Calculate the midpoint of shoulders (normalized coordinates)
#         mid_shoulder_x = (left_shoulder.x + right_shoulder.x) / 2

#         # If the horizontal difference between the nose and shoulder midpoint exceeds a threshold, flag leaning.
#         if abs(nose_pose.x - mid_shoulder_x) > 0.1:
#             leaning_flag = True

#     # -------------------------------
#     # 5. Determine Final Cheating Status
#     # -------------------------------
#     if head_turn_flag or talking_flag or leaning_flag:
#         status = "Cheating"
#         status_color = (0, 0, 255)  # Red for cheating
#     else:
#         status = "Not Cheating"
#         status_color = (0, 255, 0)  # Green for normal behavior

#     # Display the status on the frame
#     cv2.putText(frame, status, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

#     # OPTIONAL: Show individual flags for debugging
#     debug_text = f"HeadTurn: {head_turn_flag}  Talking: {talking_flag}  Leaning: {leaning_flag}"
#     cv2.putText(frame, debug_text, (30, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

#     # -------------------------------
#     # 6. Display Frame
#     # -------------------------------
#     cv2.imshow("Cheating Detection", frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# # -------------------------------
# # 7. Clean Up
# # -------------------------------
# cap.release()
# cv2.destroyAllWindows()
# face_mesh.close()
# pose.close()
