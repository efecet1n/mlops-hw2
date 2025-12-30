# Model Loading and Inference
# MLOps HW2 - Efe Çetin

import pickle
import os
from typing import Optional


class FlightDelayModel:
    """Wrapper for the trained flight delay prediction model."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize model wrapper.
        
        Args:
            model_path: Path to the pickled model file
        """
        self.model = None
        self.scaler = None
        self.feature_columns = None
        self.loaded = False
        
        if model_path:
            self.load(model_path)
    
    def load(self, model_path: str) -> None:
        """
        Load model from pickle file.
        
        Args:
            model_path: Path to the pickled model file
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            bundle = pickle.load(f)
        
        self.model = bundle['model']
        self.scaler = bundle['scaler']
        self.feature_columns = bundle['feature_columns']
        self.loaded = True
    
    def predict(self, features: list) -> int:
        """
        Make a prediction.
        
        Args:
            features: List of feature values
        
        Returns:
            Predicted delay category (0, 1, or 2)
        """
        if not self.loaded:
            raise RuntimeError("Model not loaded")
        
        import numpy as np
        features_scaled = self.scaler.transform([features])
        prediction = self.model.predict(features_scaled)
        return int(prediction[0])
    
    def predict_proba(self, features: list) -> list:
        """
        Get prediction probabilities.
        
        Args:
            features: List of feature values
        
        Returns:
            List of probabilities for each class
        """
        if not self.loaded:
            raise RuntimeError("Model not loaded")
        
        import numpy as np
        features_scaled = self.scaler.transform([features])
        proba = self.model.predict_proba(features_scaled)
        return proba[0].tolist()


# Singleton model instance
_model_instance: Optional[FlightDelayModel] = None


def get_model() -> FlightDelayModel:
    """Get or create the singleton model instance."""
    global _model_instance
    if _model_instance is None:
        model_path = os.environ.get('MODEL_PATH', 'model/flight_delay_model.pkl')
        _model_instance = FlightDelayModel(model_path)
    return _model_instance
