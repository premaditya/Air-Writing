import cv2
import numpy as np

class Canvas:

    def __init__(self, width =640, height =480):
        #-------------- Store canvas dimensions -----------------
        self.width = width
        self.height = height

        #-------------- Create a blank black canvas (BGR) -----------------
        self.canvas = np.zeros((height,width,3), dtype=np.uint8)

        #-------------- Track last drawn point for line continuity -----------------
        self.previous_point = None

    def draw(self, point):

        #-------------- No point detected, reset stroke -----------------
        if point is None:
            self.previous_point = None
            return False

        #-------------- First point of a new stroke, nothing to connect yet -----------------
        if self.previous_point is None:
            self.previous_point = point
            return False

        #-------------- Draw a line from previous point to current point -----------------
        cv2.line(
            self.canvas,
            self.previous_point,
            point,
            (255, 255, 255),
            8,
            cv2.LINE_AA
        )

        #-------------- Update previous point for next segment -----------------
        self.previous_point = point

        return True

    def clear(self):

        #-------------- Reset canvas to blank and clear stroke history -----------------
        self.canvas[:] = 0
        self.previous_point = None

    def is_empty(self):
        #-------------- Check if canvas has any drawing on it -----------------
        return not self.canvas.any()

    def get_canvas(self):
        #-------------- Return the full canvas -----------------
        return self.canvas
    

    def get_cropped_character(self):

        #-------------- Convert canvas to grayscale for contour detection -----------------
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)

        #-------------- Find outer contours of the drawn character -----------------
        contours,_ = cv2.findContours(
            gray,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        #-------------- No contours found, nothing was drawn -----------------
        if not contours:
            return None
        
        #-------------- Pick the largest contour (main character shape) -----------------
        largest = max(contours, key=cv2.contourArea)

        #-------------- Get bounding box around the character -----------------
        x, y, w, h = cv2.boundingRect(largest)

        #-------------- Crop canvas to just the character region -----------------
        cropped = self.canvas[y:y+h, x:x+w]

        return cropped