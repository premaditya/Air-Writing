from pathlib import Path

import cv2
import joblib
from skimage.feature import hog

class Preprocessor:

    def __init__(self):

        #-------------- Get project root directory -----------------
        base_dir = Path(__file__).resolve().parent.parent

        #-------------- Build path to saved HOG parameters -----------------
        hog_path = base_dir / "models" / "hog_params.pkl"

        #-------------- Load the HOG parameters used during training -----------------
        self.hog_params = joblib.load(hog_path)

        #-------------- Target image size (must match training data) -----------------
        self.image_size = (34, 34)

    def process(self, image):

        #-------------- Resize image to match model's expected input size -----------------
        image = cv2.resize(image, self.image_size)

        #-------------- Convert image to grayscale -----------------
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #-------------- Normalize pixel values to 0-1 range -----------------
        image = image.astype("float32") / 255.0

        #-------------- Extract HOG features using the same params as training -----------------
        features = hog(
            image,
            **self.hog_params
        )

        return features