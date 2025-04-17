from typing import Dict, Optional, Any
import pandas as pd
import numpy as np
from .deep_learning import DeepVesselPredictor
from .feature_engineering import MaritimeFeatureEngineering
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VesselPredictor:
    """
    High-level orchestrator for vessel prediction tasks.
    Combines feature engineering and deep learning models to train and predict vessel behavior.
    Manages the instances of MaritimeFeatureEngineering and DeepVesselPredictor.
    """
    def __init__(self, sequence_length: int = 10, use_gpu: bool = True):
        """
        Initializes the VesselPredictor.

        Args:
            sequence_length (int): The sequence length to use for the deep learning model.
            use_gpu (bool): Whether to attempt using GPU acceleration.
        """
        logging.info("Initializing VesselPredictor...")
        # Initialize the feature engineering component
        self.feature_engineer = MaritimeFeatureEngineering()
        # Initialize the deep learning prediction component
        self.deep_predictor = DeepVesselPredictor(sequence_length=sequence_length, use_gpu=use_gpu)
        # Flag to track if the system (specifically the deep_predictor) has been trained
        self.is_trained = False
        logging.info("VesselPredictor initialized.")

    def train(self,
              vessel_data: pd.DataFrame,
              validation_data: Optional[pd.DataFrame] = None,
              weather_data: Optional[pd.DataFrame] = None, # Placeholder for future use
              epochs: int = 50,
              batch_size: int = 64
             ) -> Dict[str, Any]:
        """
        Trains the prediction system using provided vessel data.
        Steps:
        1. Performs feature engineering on the training data.
        2. (Optionally) Adds weather features.
        3. Trains the deep learning model using the engineered features.

        Args:
            vessel_data (pd.DataFrame): DataFrame containing training AIS data.
            validation_data (Optional[pd.DataFrame]): Optional DataFrame for validation.
            weather_data (Optional[pd.DataFrame]): Optional DataFrame with weather data (currently unused).
            epochs (int): Number of training epochs for the deep learning model.
            batch_size (int): Batch size for training.

        Returns:
            Dict[str, Any]: A dictionary indicating the status ('success' or 'error')
                            and potentially including training history or error messages.
        """
        logging.info(f"Starting training process with {len(vessel_data)} training samples.")
        if validation_data is not None:
             logging.info(f"Using {len(validation_data)} validation samples.")

        try:
            # 1. Apply feature engineering to training data
            logging.info("Performing feature engineering on training data...")
            engineered_train_data = self.feature_engineer.process_features(vessel_data)
            logging.info("Feature engineering on training data complete.")

            # Apply feature engineering to validation data (if provided)
            engineered_validation_data = None
            if validation_data is not None:
                 logging.info("Performing feature engineering on validation data...")
                 engineered_validation_data = self.feature_engineer.process_features(validation_data)
                 logging.info("Feature engineering on validation data complete.")


            # 2. Add weather features (placeholder)
            if weather_data is not None:
                logging.warning("Weather data integration is not yet implemented.")
                # engineered_train_data = self.feature_engineer.add_weather_features(
                #     engineered_train_data, weather_data
                # )
                # if engineered_validation_data is not None:
                #      engineered_validation_data = self.feature_engineer.add_weather_features(...)


            # 3. Train the deep learning model
            logging.info("Starting deep learning model training...")
            # Pass both engineered train and validation data to the trainer
            training_history = self.deep_predictor.train(
                train_data=engineered_train_data,
                feature_engineer=self.feature_engineer, # Pass the engineer instance
                validation_data=engineered_validation_data,
                epochs=epochs,
                batch_size=batch_size
            )

            # Check if training was successful (e.g., history object is valid)
            if training_history:
                self.is_trained = self.deep_predictor.is_trained # Update status from deep predictor
                logging.info("Training process completed successfully.")
                return {
                    "status": "success",
                    "message": "Training completed successfully.",
                    "training_history": training_history.history # Return the history dict
                }
            else:
                 # Handle cases where training might fail internally but not raise exception
                 self.is_trained = False
                 logging.error("Training process failed internally in DeepVesselPredictor.")
                 return {
                     "status": "error",
                     "message": "Training failed. Check logs for details."
                 }

        except Exception as e:
            # Catch any exceptions during the process
            self.is_trained = False
            logging.error(f"Error during training: {e}", exc_info=True) # Log traceback
            return {
                "status": "error",
                "message": f"An error occurred during training: {str(e)}"
            }

    def predict(self, input_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Makes predictions using the trained system.
        Requires input data containing the necessary features for the last `sequence_length` time steps.

        Args:
            input_data (pd.DataFrame): DataFrame containing the recent AIS data (at least sequence_length points)
                                       for the vessel(s) to predict. Must contain required raw features.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing the predictions from the deep learning model,
                                      or None if prediction fails or the model is not trained.
                                      Includes keys like 'position_pred', 'motion_pred', 'status_pred'.
        """
        logging.info(f"Starting prediction process for {len(input_data)} input samples.")
        if not self.is_trained:
            logging.error("Predictor is not trained. Cannot make predictions.")
            return {"status": "error", "message": "Model not trained."}

        try:
            # 1. Apply feature engineering to the input data
            # Important: Use the *same* feature engineer instance used for training
            # to ensure consistent processing (scaling, encoding).
            logging.info("Performing feature engineering on prediction input data...")
            engineered_input_data = self.feature_engineer.process_features(input_data)
            logging.info("Feature engineering on prediction input data complete.")

            # 2. (Optional) Add weather features if implemented and used in training

            # 3. Make predictions using the deep learning model
            logging.info("Making predictions with the deep learning model...")
            predictions = self.deep_predictor.predict(
                input_data=engineered_input_data,
                feature_engineer=self.feature_engineer # Pass the engineer instance
            )

            if predictions is not None:
                logging.info("Prediction process completed successfully.")
                return {
                    "status": "success",
                    "predictions": predictions # Contains dict from deep_predictor.predict
                }
            else:
                logging.error("Prediction failed internally in DeepVesselPredictor.")
                return {
                    "status": "error",
                    "message": "Prediction failed. Check logs."
                }

        except Exception as e:
            logging.error(f"Error during prediction: {e}", exc_info=True)
            return {
                "status": "error",
                "message": f"An error occurred during prediction: {str(e)}"
            }

    def save(self, model_path: str):
        """
        Saves the state of the predictor, primarily the trained deep learning model.
        Feature engineering scalers/encoders are typically saved/loaded implicitly
        if using libraries like scikit-learn pipelines or saving the feature engineer object itself.

        Args:
            model_path (str): Path where the deep learning model should be saved.
        """
        logging.info(f"Saving model to {model_path}...")
        try:
            self.deep_predictor.save_model(model_path)
            # Optionally save the feature engineer instance if needed (e.g., using pickle)
            # import pickle
            # with open(feature_engineer_path, 'wb') as f:
            #     pickle.dump(self.feature_engineer, f)
            logging.info("Predictor state (model) saved.")
        except Exception as e:
            logging.error(f"Failed to save predictor state: {e}", exc_info=True)


    def load(self, model_path: str):
        """
        Loads the state of the predictor, primarily the trained deep learning model.

        Args:
            model_path (str): Path from where the deep learning model should be loaded.
        """
        logging.info(f"Loading model from {model_path}...")
        try:
            self.deep_predictor.load_model(model_path)
            self.is_trained = self.deep_predictor.is_trained # Update status after loading
            # Optionally load the feature engineer instance
            # import pickle
            # with open(feature_engineer_path, 'rb') as f:
            #     self.feature_engineer = pickle.load(f)
            if self.is_trained:
                 logging.info("Predictor state (model) loaded successfully.")
            else:
                 logging.warning("Model loaded, but predictor status indicates not trained (check load process).")

        except Exception as e:
            self.is_trained = False
            logging.error(f"Failed to load predictor state: {e}", exc_info=True)