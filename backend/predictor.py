from pathlib import Path

import joblib

class Predictor:

    def __init__(self):
        
        #-------------- Get project root and models directory -----------------
        base_dir = Path(__file__).resolve().parent.parent
        model_dir = base_dir / "models"

        #-------------- Load the trained KNN model -----------------
        self.model = joblib.load(model_dir / "knn_gesture_model.pkl")

        #-------------- Load the label encoder to map predictions back to letters -----------------
        self.label_encoder = joblib.load(model_dir / "label_encoder.pkl")


    def predict(self, features):

        #-------------- Predict the class (encoded label) from features -----------------
        prediction = self.model.predict([features])[0]

        #-------------- Get prediction probabilities for all classes -----------------
        probabilities = self.model.predict_proba([features])[0]

        #-------------- Convert highest probability into a confidence percentage -----------------
        confidence = probabilities.max() * 100

        #-------------- Convert encoded prediction back to the actual letter -----------------
        letter = self.label_encoder.inverse_transform([prediction])[0]

        return letter, confidence