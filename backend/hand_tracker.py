from pathlib import Path
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandTracker:

    def __init__(
            self,
            num_hands = 1,
            detection_confidence = 0.7,
            presence_confidence = 0.7,
            tracking_confidence = 0.7
    ):
        
        #-------------- Get project root directory -----------------
        self.base_dir = Path(__file__).resolve().parent.parent

        #-------------- Build path to the hand landmark model file -----------------
        model_path = self.base_dir / "models" / "hand_landmarker.task"

        #-------------- Raise error if model file is missing -----------------
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found.\n{model_path}"
            )
        
        #-------------- Load the MediaPipe hand landmark model -----------------
        base_options = python.BaseOptions(
            model_asset_path = str(model_path)
        )

        #-------------- Configure hand landmarker settings -----------------
        options = vision.HandLandmarkerOptions(
            base_options = base_options,
            running_mode = vision.RunningMode.VIDEO,
            num_hands = num_hands,
            min_hand_detection_confidence = detection_confidence,
            min_hand_presence_confidence = presence_confidence,
            min_tracking_confidence = tracking_confidence
        )

        #-------------- Create the hand landmark detector -----------------
        self.detector = vision.HandLandmarker.create_from_options(options)


    def detect(self, frame):

        #-------------- Convert frame from BGR to RGB for MediaPipe -----------------
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)\
        
        #-------------- Wrap frame into a MediaPipe Image object -----------------
        mp_image = mp.Image(
            image_format= mp.ImageFormat.SRGB,
            data = frame_rgb
        )

        #-------------- Get current timestamp in milliseconds -----------------
        timestamp = int(time.time() * 1000)

        #-------------- Run hand detection on the current video frame -----------------
        result = self.detector.detect_for_video(
            mp_image,
            timestamp
        )

        return frame_rgb, result
    
    def get_index_tip(self, result, frame_shape):

        #-------------- No hand detected, nothing to return -----------------
        if not result.hand_landmarks:
            return None
        
        #-------------- Use the first detected hand -----------------
        hand = result.hand_landmarks[0]

        h, w, _ = frame_shape

        #-------------- Landmark 8 is the index fingertip -----------------
        index_tip = hand[8]

        #-------------- Convert normalized coordinates to pixel coordinates -----------------
        x = int(index_tip.x * w)
        y = int(index_tip.y * h)

        return (x, y)
    

    def is_drawing(self, result):

        #-------------- No hand detected, not drawing -----------------
        if not result.hand_landmarks:
            return False
        
        hand = result.hand_landmarks[0]

        #-------------- Landmark IDs for fingertips -----------------
        INDEX_TIP = 8
        MIDDLE_TIP = 12
        RING_TIP = 16
        PINKY_TIP = 20

        #-------------- Landmark IDs for finger base joints (PIP) -----------------
        INDEX_PIP = 6
        MIDDLE_PIP = 10
        RING_PIP = 14
        PINKY_PIP = 18

        #-------------- Check which fingers are raised (tip above pip) -----------------
        index_up = hand[INDEX_TIP].y < hand[INDEX_PIP].y
        middle_up = hand[MIDDLE_TIP].y < hand[MIDDLE_PIP].y
        ring_up = hand[RING_TIP].y < hand[RING_PIP].y
        pinky_up = hand[PINKY_TIP].y < hand[PINKY_PIP].y

        #-------------- Drawing gesture: only index finger raised -----------------
        return index_up and not middle_up and not ring_up and not pinky_up
    

    def is_open_palm(self, result):

        #-------------- No hand detected, not an open palm -----------------
        if not result.hand_landmarks:
            return False
        
        hand = result.hand_landmarks[0]

        #-------------- Pairs of (tip, pip) landmark IDs for each finger -----------------
        fingers = [
            (8,6),
            (12,10),
            (16,14),
            (20,18)
        ]

        fingers_up = 0

        #-------------- Count how many fingers are raised -----------------
        for tip,pip in fingers:

            if hand[tip].y < hand[pip].y:

                fingers_up += 1

        #-------------- Open palm gesture: all 4 fingers raised -----------------
        return fingers_up == 4
    
    def is_predict_gesture(self, result):

        #-------------- No hand detected, not a predict gesture -----------------
        if not result.hand_landmarks:
            return False
        
        hand = result.hand_landmarks[0]

        #-------------- Landmark IDs for fingertips -----------------
        INDEX_TIP = 8
        MIDDLE_TIP = 12
        RING_TIP = 16
        PINKY_TIP = 20

        #-------------- Landmark IDs for finger base joints (PIP) -----------------
        INDEX_PIP = 6
        MIDDLE_PIP = 10
        RING_PIP = 14
        PINKY_PIP = 18

        #-------------- Check which fingers are raised (tip above pip) -----------------
        index_up = hand[INDEX_TIP].y < hand[INDEX_PIP].y
        middle_up = hand[MIDDLE_TIP].y < hand[MIDDLE_PIP].y
        ring_up = hand[RING_TIP].y < hand[RING_PIP].y
        pinky_up = hand[PINKY_TIP].y < hand[PINKY_PIP].y
    
        #-------------- Predict gesture: index, middle, ring raised, pinky down -----------------
        return (index_up and middle_up and ring_up and not pinky_up)

    
    def close(self):
        
        #-------------- Release the hand landmark detector -----------------
        self.detector.close()