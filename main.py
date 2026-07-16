import cv2
import time

from backend.hand_tracker import HandTracker
from backend.canvas import Canvas
from backend.preprocessing import Preprocessor
from backend.predictor import Predictor

#-------------- Initialize hand tracker, canvas, and model components -----------------
tracker = HandTracker()
canvas = Canvas()
preprocessor = Preprocessor()
predictor = Predictor()

#-------------- Timers for hold-to-clear and hold-to-predict gestures -----------------
clear_start_timer = None
predict_start_timer = None

#-------------- Prediction state -----------------
prediction_locked = False
prediction = ""

#-------------- Open the default webcam -----------------
cap = cv2.VideoCapture(0)

#-------------- Stop if webcam fails to open -----------------
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

while True:

    #-------------- Read a frame from the webcam -----------------
    success, frame = cap.read()

    if not success:
        break

    #-------------- Mirror the webcam feed -----------------
    frame = cv2.flip(frame, 1)

    #-------------- Detect hand landmarks in the frame -----------------
    frame_rgb, result = tracker.detect(frame)

    #-------------- Get index fingertip coordinates -----------------
    index_tip = tracker.get_index_tip(result, frame.shape)

    #-------------- Check if user wants to draw -----------------
    drawing = tracker.is_drawing(result)

    #-------------- Check if user wants to clear the canvas -----------------
    clear = tracker.is_open_palm(result)

    #-------------- Check if user wants to predict the drawn letter -----------------
    predict = tracker.is_predict_gesture(result)

    if clear and not canvas.is_empty():

        #-------------- Start timer when clear gesture begins -----------------
        if clear_start_timer is None:
            clear_start_timer = time.time()

        elapsed = time.time() - clear_start_timer

        #-------------- Show countdown text while holding gesture -----------------
        cv2.putText(
            frame,
            f"Hold to Clear: {1 - elapsed:.1f}s",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )

        #-------------- Clear canvas once gesture is held long enough -----------------
        if elapsed >= 1:
            canvas.clear()

            # Reset prediction
            prediction = ""
            prediction_locked = False

            clear_start_timer = None

    else:
        #-------------- Reset timer if gesture stops -----------------
        clear_start_timer = None


    
    if predict and not canvas.is_empty() and not prediction_locked:

        #-------------- Start timer when predict gesture begins -----------------
        if predict_start_timer is None:
            predict_start_timer = time.time()

        elapsed = time.time() - predict_start_timer

        #-------------- Show countdown text while holding gesture -----------------
        cv2.putText(
            frame,
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
            cropped = canvas.get_cropped_character()

            if cropped is not None:

                #-------------- Extract features and predict the letter -----------------
                features = preprocessor.process(cropped)

                prediction, confidence = predictor.predict(features)

                #-------------- Lock prediction until next drawing/clear -----------------
                prediction_locked = True

                # cv2.imshow("Cropped", cropped)            # Use this see the cropped Picture

            predict_start_timer = None

    else:
        #-------------- Reset timer if gesture stops -----------------
        predict_start_timer = None

    # ----------------------------
    # Draw hand landmarks
    # ----------------------------
    if result.hand_landmarks:

        #-------------- Draw a dot on every detected hand landmark -----------------
        for hand in result.hand_landmarks:

            h, w, _ = frame.shape

            for landmark in hand:

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)

    # ----------------------------
    # Highlight index finger
    # ----------------------------
    if index_tip:

        #-------------- Red cursor while drawing, green otherwise -----------------
        color = (0, 0, 255) if drawing else (0, 255, 0)

        # White border
        cv2.circle(frame, index_tip, 12, (255, 255, 255), -1)

        # Colored center
        cv2.circle(frame, index_tip, 8, color, -1)

    # ----------------------------
    # Air Drawing
    # ----------------------------
    if drawing:

        #-------------- Draw on canvas at fingertip position -----------------
        changed = canvas.draw(index_tip)

        #-------------- Unlock prediction if canvas was updated -----------------
        if changed:
            prediction_locked = False

        cv2.putText(
            frame,
            "DRAWING",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    else:
        #-------------- No drawing, reset stroke continuity -----------------
        canvas.draw(None)

    # ----------------------------
    # Display Windows
    # ----------------------------
    if prediction:

        #-------------- Display predicted letter and confidence -----------------
        cv2.putText(
            frame,
            f"Prediction: {prediction}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Confidence: {confidence:.2f}%",
            (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )

    #-------------- Blend camera feed with the drawing canvas -----------------
    overlay = cv2.addWeighted(
    frame,
    0.8,
    canvas.get_canvas(),
    0.8,
    0,
    )

    #-------------- Show the final combined frame in a window -----------------
    cv2.imshow("Air Writing", overlay)

    # ----------------------------
    # Keyboard Controls
    # ----------------------------
    #-------------- Wait for a key press -----------------
    key = cv2.waitKey(1) & 0xFF

    #-------------- 'c' key clears the canvas manually -----------------
    if key == ord("c"):
        canvas.clear()

        prediction = ""
        prediction_locked = False

    #-------------- 'q' key quits the program -----------------
    elif key == ord("q"):
        break

#-------------- Release resources when the loop ends -----------------
tracker.close()
cap.release()
cv2.destroyAllWindows()