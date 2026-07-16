import av
import cv2
import time

from backend.hand_tracker import HandTracker
from backend.canvas import Canvas
from backend.preprocessing import Preprocessor
from backend.predictor import Predictor


class FrameProcessor:

    def __init__(self):

        #-------------- Initialize hand tracking and drawing canvas -----------------
        self.tracker = HandTracker()
        self.canvas = Canvas()

        #-------------- Initialize feature extractor and predictor -----------------
        self.preprocessor = Preprocessor()
        self.predictor = Predictor()

        #-------------- Timers for hold-to-clear and hold-to-predict gestures -----------------
        self.clear_start_timer = None
        self.predict_start_timer = None

        #-------------- Store current prediction and lock state -----------------
        self.prediction = ""
        self.prediction_locked = False

        self.confidence = 0.0

    def recv(self, frame):

        #-------------- Convert incoming video frame to a numpy array -----------------
        image = frame.to_ndarray(format="bgr24")

        #-------------- Flip frame horizontally for mirror-view -----------------
        image = cv2.flip(image, 1)

        #-------------- Run hand detection on the current frame -----------------
        _, result = self.tracker.detect(image)

        # ----------------------------
        # Draw Hand Landmarks
        # ----------------------------

        if result.hand_landmarks:

            h, w, _ = image.shape

            #-------------- Draw a dot on every detected hand landmark -----------------
            for hand in result.hand_landmarks:

                for landmark in hand:

                    x = int(landmark.x * w)
                    y = int(landmark.y * h)

                    cv2.circle(
                        image,
                        (x, y),
                        5,
                        (0, 255, 0),
                        -1,
                    )

        # ----------------------------
        # Get Index Finger
        # ----------------------------

        #-------------- Get index fingertip position for drawing -----------------
        index_tip = self.tracker.get_index_tip(result, image.shape)

        #-------------- Check if the drawing gesture is active -----------------
        drawing = self.tracker.is_drawing(result)

        # ----------------------------
        # Air Drawing
        # ----------------------------

        if drawing:

            #-------------- Draw on canvas at fingertip position -----------------
            changed = self.canvas.draw(index_tip)

            #-------------- Unlock prediction if canvas was updated -----------------
            if changed:
                self.prediction_locked = False

        else:
            #-------------- No drawing, reset stroke continuity -----------------
            self.canvas.draw(None)

        # ----------------------------
        # Cursor
        # ----------------------------

        if index_tip:

            #-------------- Red cursor while drawing, green otherwise -----------------
            color = (0, 0, 255) if drawing else (0, 255, 0)

            #-------------- Draw white outer ring for cursor -----------------
            cv2.circle(
                image,
                index_tip,
                12,
                (255, 255, 255),
                -1,
            )

            #-------------- Draw colored inner circle for cursor -----------------
            cv2.circle(
                image,
                index_tip,
                8,
                color,
                -1,
            )

        # ----------------------------
        # Clear Gesture
        # ----------------------------

        #-------------- Check if open palm gesture is active -----------------
        clear = self.tracker.is_open_palm(result)

        if clear and not self.canvas.is_empty():

            #-------------- Start timer when clear gesture begins -----------------
            if self.clear_start_timer is None:
                self.clear_start_timer = time.time()

            elapsed = time.time() - self.clear_start_timer

            #-------------- Show countdown text while holding gesture -----------------
            cv2.putText(
                image,
                f"Hold to Clear: {max(0, 1 - elapsed):.1f}s",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
            )

            #-------------- Clear canvas once gesture is held long enough -----------------
            if elapsed >= 1:

                self.canvas.clear()

                self.prediction = ""
                self.confidence = 0.0
                self.prediction_locked = False

                self.clear_start_timer = None

        else:
            #-------------- Reset timer if gesture stops -----------------
            self.clear_start_timer = None

        # ----------------------------
        # Predict Gesture
        # ----------------------------

        #-------------- Check if predict gesture (3 fingers) is active -----------------
        predict = self.tracker.is_predict_gesture(result)

        if predict and not self.canvas.is_empty() and not self.prediction_locked:

            #-------------- Start timer when predict gesture begins -----------------
            if self.predict_start_timer is None:
                self.predict_start_timer = time.time()

            elapsed = time.time() - self.predict_start_timer

            #-------------- Show countdown text while holding gesture -----------------
            cv2.putText(
                image,
                f"Hold to Predict: {max(0, 1 - elapsed):.1f}s",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )

            #-------------- Run prediction once gesture is held long enough -----------------
            if elapsed >= 1:

                #-------------- Crop the drawn character from the canvas -----------------
                cropped = self.canvas.get_cropped_character()

                if cropped is not None:

                    #-------------- Extract features and predict the letter -----------------
                    features = self.preprocessor.process(cropped)

                    self.prediction, self.confidence = self.predictor.predict(features)

                    #-------------- Lock prediction until next drawing/clear -----------------
                    self.prediction_locked = True

                self.predict_start_timer = None

        else:
            #-------------- Reset timer if gesture stops -----------------
            self.predict_start_timer = None

        # ----------------------------
        # Prediction
        # ----------------------------

        if self.prediction:

            #-------------- Display predicted letter on frame -----------------
            cv2.putText(
                image,
                f"Prediction: {self.prediction}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2,
            )

            #-------------- Display prediction confidence on frame -----------------
            cv2.putText(
                image,
                f"Confidence: {self.confidence:.1f}%",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

        # ----------------------------
        # Overlay Canvas
        # ----------------------------

        #-------------- Blend camera feed with the drawing canvas -----------------
        overlay = cv2.addWeighted(
            image,
            0.8,
            self.canvas.get_canvas(),
            0.8,
            0,
        )

        #-------------- Return processed frame back to the video stream -----------------
        return av.VideoFrame.from_ndarray(
            overlay,
            format="bgr24",
        )