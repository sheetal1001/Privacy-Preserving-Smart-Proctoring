import cv2
import mediapipe as mp
import numpy as np
import av 
import csv
from datetime import datetime
import os
from streamlit_webrtc import VideoProcessorBase
from ultralytics import YOLO 

# Log file for recording session logs
# If log file is not present, create and append the headers
LOG_FILE = "session_logs.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Violation Type", "Severity"])

def get_distance(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

class ProctorProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.face_mesh = None 
        self.segmentation = None 
        self.yolo = None 
        self.last_status = "Initializing..."
        self.last_color = (0, 255, 0)
        self.last_yaw = 0
        self.last_pitch_ratio = 1.0  
        self.face_detected = False
        self.previous_status = "Initializing..."
        self.prohibited_object = None 

    def log_violation(self, status, severity):
        # Logging only when status changes and only voilations.
        if status != self.previous_status and status != "No Unusual Activity":
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(LOG_FILE, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, status, severity])
        self.previous_status = status

    def recv(self, frame):
        # A video is just a fast sequence of still images, converting the images to numpy array for processing
        # .copy() creates a writeable clone of the original read-only frame to prevent crashes if we draw directly on it.
        img = frame.to_ndarray(format="bgr24").copy()

        try:
            if self.face_mesh is None:
                # Importing face mesh from mp.solutions, refine_landmark=True for iris tracking
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    max_num_faces=2, refine_landmarks=True,
                    min_detection_confidence=0.5, min_tracking_confidence=0.5 
                )
                # SelfieSegmentation is MediaPipe's built-in background removal AI
                # Assigns value close to 1 to any pixel it thinks belongs to a human, and 0 to everything else
                # model_selection=1 (highly optimized for speed and is specifically trained for scenarios
                # where a single person is sitting close to the camera)
                self.mp_selfie = mp.solutions.selfie_segmentation
                self.segmentation = self.mp_selfie.SelfieSegmentation(model_selection=1)
                
                # Load the YOLOv8 Nano model for prohibited objects detection like phones
                self.yolo = YOLO('yolov8n.pt') 

            self.frame_count += 1
            # Physical pixel dimensions of the raw video frame coming directly from the webcam hardware
            h, w, _ = img.shape

            # Mediapipe needs RGB but openCV uses BGR by default. Convert to RGB.
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # YOLO OBJECT DETECTION (Every 15th frame on SHARP image)
            if self.frame_count % 15 == 0:
                self.prohibited_object = None # Reset flag
                # Class: 67 = cell phone
                # verbose=False stops YOLO from spamming the terminal
                results_yolo = self.yolo.predict(rgb_img, classes=[67], verbose=False)
                
                for r in results_yolo:
                    if len(r.boxes) > 0:
                        # Get the name of the first detected object
                        cls_id = int(r.boxes.cls[0])
                        self.prohibited_object = self.yolo.names[cls_id]

            # PRIVACY BLUR (Applied AFTER YOLO looks at the sharp frame)
            # seg_results: grid of decimal numbers from 0.0 (background) to 1.0 (person)
            seg_results = self.segmentation.process(rgb_img)

            # duplicated the seg mask (2D) to 3D by *3 and stacked along last axis (color channels) 
            # 0.1 --> if more than 10% sure, it is a person, mark them as person
            condition = np.stack((seg_results.segmentation_mask,) * 3, axis=-1) > 0.1

            # Creates a blurred image by averaging a 99x99 pixel grid around each pixel, 
            # letting OpenCV auto-calculate the standard deviation (0 flag)
            blurred_bg = cv2.GaussianBlur(img, (99, 99), 0)

            # np.where(CONDITION, PICK_THIS_IF_TRUE, PICK_THIS_IF_FALSE)
            img = np.where(condition, img, blurred_bg).astype(np.uint8)

            # FACE & BEHAVIOR TRACKING (Every 5th frame) 
            if self.frame_count % 5 == 0:
                small_img = cv2.resize(rgb_img, (480, int(480 * h / w)))
                results = self.face_mesh.process(small_img)

                if results.multi_face_landmarks:
                    self.face_detected = True
                    
                    # Highest Priority Alert: Object Detection overrides everything
                    if self.prohibited_object:
                        current_status = f"Prohibited Object: {self.prohibited_object.upper()}"
                        self.last_color = (0, 0, 255)
                        severity = "Critical"
                        
                    elif len(results.multi_face_landmarks) > 1:
                        current_status = "Multiple Persons Detected!"
                        self.last_color = (0, 0, 255)
                        severity = "Critical"
                        
                    else:
                        # Exact 478 3D coordinate points mapped to examinee's face.
                        face_landmarks = results.multi_face_landmarks[0]
                        
                        # Eye Tracking for suspicious eyes tracking
                        
                        # iris_r = centre of right eye's iris, landmark 468
                        # inner_r = inner corner of right eye, landmark 133
                        # outer_r = outer corner of right eye, landmark 33
                        # ratio_r = position of iris (middle or drifted)
                    
                        iris_r, inner_r, outer_r = face_landmarks.landmark[468], face_landmarks.landmark[133], face_landmarks.landmark[33]
                        dist_inner_r, dist_outer_r = get_distance(iris_r, inner_r), get_distance(iris_r, outer_r)
                        ratio_r = dist_inner_r / (dist_inner_r + dist_outer_r) if (dist_inner_r + dist_outer_r) > 0 else 0.5
                        
                        iris_l, inner_l, outer_l = face_landmarks.landmark[473], face_landmarks.landmark[362], face_landmarks.landmark[263]
                        dist_inner_l, dist_outer_l = get_distance(iris_l, inner_l), get_distance(iris_l, outer_l)
                        ratio_l = dist_inner_l / (dist_inner_l + dist_outer_l) if (dist_inner_l + dist_outer_l) > 0 else 0.5
                        
                        # ideally 0.5 if both iris in middle
                        gaze_ratio = (ratio_r + ratio_l) / 2.0
                        
                        # Mouth Tracking
                        top_lip, bottom_lip = face_landmarks.landmark[13].y, face_landmarks.landmark[14].y
                        forehead, chin = face_landmarks.landmark[10].y, face_landmarks.landmark[152].y
                        mouth_ratio = (bottom_lip - top_lip) / (chin - forehead) if (chin - forehead) > 0 else 0

                        # Pitch Logic (Vertical Ratio)
                        nose_y = face_landmarks.landmark[1].y
                        top_face = nose_y - forehead
                        bottom_face = chin - nose_y
                        pitch_ratio = top_face / bottom_face if bottom_face > 0 else 1.0
                        self.last_pitch_ratio = pitch_ratio
                        
                        # Yaw Logic (solvePnP)
                        # why 3D for Yaw -> head turn left and right
                        # If we only measured the 2D pixel distance between the eyes to see if the 
                        # head is turning, it would become inaccurate if the student simply leans 
                        # forward or backward (the eyes look closer together when leaning back).
                        # Using 3D solvePnP fixes this by calculating true rotational degrees.

                        # Note on Coordinate Conversion:
                        # MediaPipe outputs landmarks as a percentage of the screen (0.0 to 1.0).
                        # OpenCV's solvePnP requires absolute physical pixel coordinates.
                        # Therefore, we multiply the x/y values by the frame width (w) and height (h).
                        image_points = np.array([
                            (face_landmarks.landmark[1].x * w, face_landmarks.landmark[1].y * h), # nose tip
                            (face_landmarks.landmark[152].x * w, face_landmarks.landmark[152].y * h), # chin
                            (face_landmarks.landmark[33].x * w, face_landmarks.landmark[33].y * h), # left eye 
                            (face_landmarks.landmark[263].x * w, face_landmarks.landmark[263].y * h), # right eye
                            (face_landmarks.landmark[61].x * w, face_landmarks.landmark[61].y * h), # left corner of mouth
                            (face_landmarks.landmark[291].x * w, face_landmarks.landmark[291].y * h) # right corner of mouth
                        ], dtype=np.float32)
                        
                        model_points = np.array([
                            (0.0, 0.0, 0.0), # Nose Tip
                            (0.0, -330.0, -65.0), # Chin
                            (-225.0, 170.0, -135.0), # The Left Eye 
                            (225.0, 170.0, -135.0), # The Right Eye 
                            (-150.0, -150.0, -125.0), # left corner of mouth
                            (150.0, -150.0, -125.0) # right corner of mouth
                        ], dtype=np.float32)
                        
                        # Camera matrix estimation
                        # Telling openCv that the (w/2 and h/2) is the opticaal center of the camera
                        cam_matrix = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype=np.float32)
                        # Calculating distortion is computationally heavy and hence a matrix of zeros 
                        dist_coeffs = np.zeros((4, 1), dtype=np.float32)
                        success, rot_vec, trans_vec = cv2.solvePnP(model_points, image_points, cam_matrix, dist_coeffs)

                        if success:
                            # unpacks compressed Rotation Vector into 3*3 matrix
                            rmat, _ = cv2.Rodrigues(rot_vec)

                            # stacking 3*3 rmat and stacking translation vector
                            proj_matrix = np.hstack((rmat, trans_vec))

                            # Extracting the Human Angles (Euler Angles)
                            _, _, _, _, _, _, eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)
                            _, yaw, _ = eulerAngles.flatten() 
                            self.last_yaw = yaw
                            
                            # Hierarchy of standard violations
                            current_status = "No Unusual Activity"
                            self.last_color = (0, 255, 0) # Green (R, G, B)
                            severity = "Low"
                            
                            # ideally 0.5, safe space for reading content on screen. Can be adjusted as required.
                            if gaze_ratio < 0.4 or gaze_ratio > 0.6:
                                current_status = "Suspicious Eye Movement"
                                self.last_color = (0, 0, 255)
                                severity = "High"
                            elif mouth_ratio > 0.10: 
                                current_status = "Speaking Detected"
                                self.last_color = (0, 0, 255) 
                                severity = "High"
                            elif abs(yaw) > 30:
                                current_status = "Looking Away (Left/Right)"
                                self.last_color = (0, 0, 255) 
                                severity = "High"
                            elif pitch_ratio < 0.9 or pitch_ratio > 2: 
                                current_status = "Looking Up/Down"
                                self.last_color = (0, 0, 255) 
                                severity = "High"
                    
                    self.last_status = current_status
                    self.log_violation(current_status, severity)

                else:
                    self.face_detected = False

            # Drawing Logic
            cv2.putText(img, "AI ACTIVE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if self.face_detected:
                cv2.putText(img, f"Status: {self.last_status}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.last_color, 2)
                cv2.putText(img, f"Yaw: {int(self.last_yaw)} | Pitch Ratio: {self.last_pitch_ratio:.2f}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            else:
                cv2.putText(img, "No Face Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")
            
        except Exception as e:
            cv2.putText(img, f"ERROR: {str(e)}", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")