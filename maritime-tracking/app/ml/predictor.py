from typing import Dict, Optional
import pandas as pd
import numpy as np
from .deep_learning import DeepVesselPredictor
from .feature_engineering import MaritimeFeatureEngineering

class VesselPredictor:
    def __init__(self, use_gpu: bool = True):
        self.feature_engineer = MaritimeFeatureEngineering()
        self.deep_predictor = DeepVesselPredictor(use_gpu)
        self.is_trained = False

    def train(self, vessel_data: pd.DataFrame, weather_data: Optional[pd.DataFrame] = None) -> Dict:
        """Train the prediction system"""
        try:
            # Apply feature engineering
            engineered_data = self.feature_engineer.calculate_derived_features(vessel_data)
            
            # Add weather features if available
            if weather_data is not None:
                engineered_data = self.feature_engineer.add_weather_features(
                    engineered_data, weather_data
                )
            
            # Train deep learning model
            training_history = self.deep_predictor.train(engineered_data)
            self.is_trained = True
            
            return {
                "status": "success",
                "training_history": training_history
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    def predict(self, vessel_data: pd.DataFrame, 
                weather_data: Optional[pd.DataFrame] = None) -> Dict:
        """Make predictions for vessel trajectory"""
        try:
            # Feature engineering
            engineered_data = self.feature_engineer.calculate_derived_features(vessel_data)
            
            # Add weather features if available
            if weather_data is not None:
                engineered_data = self.feature_engineer.add_weather_features(
                    engineered_data, weather_data
                )
            
            # Get predictions
            predictions = self.deep_predictor.predict(engineered_data)
            
            # Process results
            return {
                "status": "success",
                "predictions": {
                    "position": predictions.tolist(),
                    "vessel_info": vessel_data.iloc[-1][
                        ['MMSI', 'VesselName', 'VesselType', 'Length', 'Width']
                    ].to_dict(),
                    "environmental_factors": weather_data.iloc[-1].to_dict() 
                    if weather_data is not None else None
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    def get_status(self) -> Dict:
        """Get predictor status"""
        return {
            "is_trained": self.is_trained,
            "deep_learning_status": self.deep_predictor.is_trained,
            "features_available": self.feature_engineer.get_available_features()
        }