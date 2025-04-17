import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate, TimeDistributed, BatchNormalization
from tensorflow.keras.optimizers import Adam
import pandas as pd
import os
import logging
from typing import List, Dict, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DeepVesselPredictor:
    """
    Implements deep learning models (LSTM-based) for vessel trajectory prediction.
    Handles model building, data preparation for sequences, training, and prediction.
    Uses multiple inputs (sequential, categorical, binary, temporal) and multiple outputs.
    """
    def __init__(self, sequence_length: int = 10, use_gpu: bool = True):
        """
        Initializes the predictor, configures GPU usage, defines feature sets,
        and builds the LSTM model structure.

        Args:
            sequence_length (int): The number of time steps for LSTM input sequences.
            use_gpu (bool): Flag to attempt using GPU for training and inference.
        """
        self.sequence_length = sequence_length
        self.use_gpu = use_gpu
        self._configure_gpu() # Setup GPU/CPU device usage

        # Define feature groups based on how they'll be used in the model
        # These names should match columns in the DataFrame passed for training/prediction
        # Continuous features used in the LSTM sequence input
        self.sequential_continuous_features = [
            'LAT', 'LON',           # Position (potentially scaled)
            'SOG', 'COG', 'Heading', # Motion (potentially scaled)
            'acceleration', 'turn_rate' # Derived kinematic features (potentially scaled)
            # Add other relevant continuous features like 'wind_speed', 'wave_height' if available and scaled
        ]

        # Categorical features (potentially one-hot encoded) used as static input alongside LSTM output
        # These names should be the *original* categorical column names before encoding
        self.static_categorical_features = [
            'VesselType', # Or 'vessel_category' if using the derived name
            'Status',     # Or 'status_name'
            'Cargo',      # If available
            'TransceiverClass',
            'speed_category',
            'movement_direction',
            'size_category'
            # Add other relevant categorical features
        ]
        # Placeholder for the actual encoded feature names (determined after fitting encoder)
        self._encoded_categorical_feature_names: Optional[List[str]] = None

        # Binary features used as static input
        self.static_binary_features = [
            'is_night',
            'is_weekend',
            'is_underway',
            'is_at_anchor',
            'is_moored',
            'is_fishing_vessel', # Renamed from is_fishing
            'is_restricted',
            'is_passenger',
            'is_cargo',
            'is_tanker',
            'is_class_a',
            'is_class_b'
            # Add other relevant binary flags
        ]

        # Temporal features (cyclical encoding might be beneficial) used as static input
        self.static_temporal_features = [
            'hour', # Consider sin/cos transform
            'day_of_week', # Consider sin/cos transform
            'month' # Consider sin/cos transform
        ]

        # --- Model Building ---
        # Determine input shapes based on feature lists
        self.num_sequential_features = len(self.sequential_continuous_features)
        # Note: num_categorical_features needs to be set *after* fitting the encoder
        self.num_categorical_features_encoded = None # Will be set later
        self.num_binary_features = len(self.static_binary_features)
        self.num_temporal_features = len(self.static_temporal_features)

        # Build the Keras model(s)
        self.lstm_model: Optional[Model] = None # Initialize model as None
        # self.rl_model = self._build_enhanced_rl_model() # RL model build placeholder
        self.is_trained = False # Flag to track if the model has been trained

    def _configure_gpu(self):
        """Configures TensorFlow to use GPU if available and requested, otherwise CPU."""
        if self.use_gpu:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    # Enable memory growth to prevent TensorFlow from allocating all GPU memory at once
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logging.info(f"Using GPU: {gpus[0].name}")
                except RuntimeError as e:
                    # Memory growth must be set before GPUs have been initialized
                    logging.error(f"Error setting memory growth for GPU: {e}. Falling back to CPU.")
                    self.use_gpu = False
                    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Explicitly disable GPU
            else:
                logging.info("No GPU found, falling back to CPU.")
                self.use_gpu = False
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        else:
            logging.info("Using CPU for computations.")
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Explicitly disable GPU

    def build_lstm_model(self, num_encoded_categorical: int):
        """
        Builds the enhanced LSTM model architecture using the Keras Functional API.
        Takes multiple inputs (sequential, categorical, binary, temporal) and produces multiple outputs.

        Args:
            num_encoded_categorical (int): The number of features after one-hot encoding categorical inputs.
                                           This is needed to define the categorical input layer shape.
        """
        logging.info("Building enhanced LSTM model...")
        self.num_categorical_features_encoded = num_encoded_categorical

        # --- Define Input Layers ---
        # Input for sequential continuous data (e.g., scaled position, motion over time)
        seq_input = Input(shape=(self.sequence_length, self.num_sequential_features), name='sequential_input')

        # Input for static categorical data (one-hot encoded, taken from the last time step of the sequence)
        cat_input = Input(shape=(self.num_categorical_features_encoded,), name='categorical_input')

        # Input for static binary data (from the last time step)
        binary_input = Input(shape=(self.num_binary_features,), name='binary_input')

        # Input for static temporal data (from the last time step, potentially cyclically encoded)
        temporal_input = Input(shape=(self.num_temporal_features,), name='temporal_input')

        # --- Process Sequential Data ---
        # Stacked LSTMs to capture temporal dependencies
        lstm_out = LSTM(128, return_sequences=True, name='lstm_1')(seq_input)
        lstm_out = BatchNormalization(name='bn_lstm_1')(lstm_out) # Add Batch Norm
        lstm_out = Dropout(0.3, name='dropout_lstm_1')(lstm_out) # Increase dropout
        lstm_out = LSTM(64, return_sequences=False, name='lstm_2')(lstm_out) # Last LSTM returns single vector
        lstm_out = BatchNormalization(name='bn_lstm_2')(lstm_out)
        lstm_out = Dropout(0.3, name='dropout_lstm_2')(lstm_out)
        # lstm_out now holds the encoded temporal information -> shape (batch_size, 64)

        # --- Process Static Data ---
        # Simple Dense layers for static features before concatenation
        cat_features = Dense(32, activation='relu', name='dense_cat_1')(cat_input)
        # cat_features = BatchNormalization(name='bn_cat_1')(cat_features) # Optional BN
        cat_features = Dense(16, activation='relu', name='dense_cat_2')(cat_features)

        binary_features = Dense(16, activation='relu', name='dense_bin_1')(binary_input) # Increased size
        # binary_features = BatchNormalization(name='bn_bin_1')(binary_features) # Optional BN
        binary_features = Dense(8, activation='relu', name='dense_bin_2')(binary_features)

        temporal_features = Dense(16, activation='relu', name='dense_temp_1')(temporal_input) # Increased size
        # temporal_features = BatchNormalization(name='bn_temp_1')(temporal_features) # Optional BN
        temporal_features = Dense(8, activation='relu', name='dense_temp_2')(temporal_features)

        # --- Combine Features ---
        # Concatenate the output of the LSTM stack and the processed static features
        combined = Concatenate(name='concatenate_features')([
            lstm_out,
            cat_features,
            binary_features,
            temporal_features
        ])
        combined = BatchNormalization(name='bn_combined')(combined) # BN after concatenation

        # --- Dense Layers for Final Processing ---
        combined = Dense(128, activation='relu', name='dense_combined_1')(combined) # Increased size
        combined = Dropout(0.4, name='dropout_combined_1')(combined) # Increased dropout
        combined = Dense(64, activation='relu', name='dense_combined_2')(combined)
        combined = BatchNormalization(name='bn_combined_2')(combined)
        combined = Dropout(0.4, name='dropout_combined_2')(combined)

        # --- Define Output Layers (Prediction Heads) ---
        # Output for next position (Latitude, Longitude) - Regression
        position_output = Dense(2, name='position_output')(combined) # Linear activation for regression

        # Output for next motion state (SOG, COG, Heading) - Regression
        motion_output = Dense(3, name='motion_output')(combined) # Linear activation

        # Output for next status (e.g., probability of being underway) - Classification
        # Using sigmoid for binary classification (underway vs not underway)
        status_output = Dense(1, activation='sigmoid', name='status_output')(combined)

        # --- Create and Compile Model ---
        self.lstm_model = Model(
            inputs=[seq_input, cat_input, binary_input, temporal_input],
            outputs=[position_output, motion_output, status_output],
            name='MultiInput_MultiOutput_LSTM_Predictor'
        )

        # Define losses for each output head
        losses = {
            'position_output': 'mse', # Mean Squared Error for position regression
            'motion_output': 'mse',   # Mean Squared Error for motion regression
            'status_output': 'binary_crossentropy' # Binary Crossentropy for status classification
        }
        # Define weights for each loss (optional, adjust to prioritize certain outputs)
        loss_weights = {
            'position_output': 1.0,
            'motion_output': 0.5, # Example: Give less weight to motion prediction
            'status_output': 0.2  # Example: Give less weight to status prediction
        }
        # Define metrics to monitor for each output
        metrics = {
            'position_output': ['mae', 'mse'], # Mean Absolute Error, Mean Squared Error
            'motion_output': ['mae', 'mse'],
            'status_output': ['accuracy', tf.keras.metrics.AUC(name='auc')] # Accuracy, Area Under Curve
        }

        # Compile the model
        optimizer = Adam(learning_rate=0.001) # Standard Adam optimizer
        self.lstm_model.compile(
            optimizer=optimizer,
            loss=losses,
            loss_weights=loss_weights, # Apply loss weights
            metrics=metrics
        )

        logging.info("LSTM model built and compiled successfully.")
        print(self.lstm_model.summary()) # Print model summary

    # Placeholder for RL model building
    # def _build_enhanced_rl_model(self):
    #     # ... Implementation for RL model ...
    #     logging.warning("RL model building not implemented yet.")
    #     return None

    def prepare_data_for_lstm(self,
                              df: pd.DataFrame,
                              feature_engineer: 'MaritimeFeatureEngineering', # Use forward reference
                              fit_scalers: bool = True
                             ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Prepares data from a DataFrame into the format required by the multi-input LSTM model.
        Handles scaling, encoding, and sequence creation.

        Args:
            df (pd.DataFrame): DataFrame containing processed features (including derived ones).
            feature_engineer (MaritimeFeatureEngineering): The fitted feature engineering instance
                                                           containing scalers and encoders.
            fit_scalers (bool): Whether to fit scalers/encoders (True for training) or just transform
                                (False for validation/testing/prediction).

        Returns:
            Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
                - A dictionary of input arrays for the model (keys match input layer names).
                - A dictionary of target arrays for the model (keys match output layer names).
        """
        logging.info(f"Preparing data for LSTM. Fit scalers: {fit_scalers}")
        try:
            # 1. Scale/Encode Features using the Feature Engineer
            # Apply scaling to continuous features
            continuous_to_scale = self.sequential_continuous_features # Add others if needed
            if fit_scalers:
                 df[continuous_to_scale] = feature_engineer.continuous_scaler.fit_transform(df[continuous_to_scale])
                 logging.info("Fitted and transformed continuous features.")
                 # Fit encoder and get encoded names/shape
                 encoded_cats = feature_engineer.categorical_encoder.fit_transform(df[self.static_categorical_features].astype(str))
                 self._encoded_categorical_feature_names = feature_engineer.categorical_encoder.get_feature_names_out(self.static_categorical_features).tolist()
                 self.num_categorical_features_encoded = encoded_cats.shape[1]
                 logging.info(f"Fitted categorical encoder. Encoded shape: {encoded_cats.shape}")

            else:
                 if not hasattr(feature_engineer.continuous_scaler, 'mean_') or \
                    not hasattr(feature_engineer.categorical_encoder, 'categories_'):
                     raise RuntimeError("Scalers/encoders must be fitted before transforming. Call with fit_scalers=True first.")
                 df[continuous_to_scale] = feature_engineer.continuous_scaler.transform(df[continuous_to_scale])
                 encoded_cats = feature_engineer.categorical_encoder.transform(df[self.static_categorical_features].astype(str))
                 # Ensure num_categorical_features_encoded is set if not fitting
                 if self.num_categorical_features_encoded is None:
                      self.num_categorical_features_encoded = encoded_cats.shape[1]

                 logging.info("Transformed continuous and categorical features using existing scalers/encoders.")


            # Add encoded categorical features back to DataFrame for easier sequence creation (optional)
            # Or handle them separately during sequence building
            encoded_cat_df = pd.DataFrame(encoded_cats, index=df.index, columns=self._encoded_categorical_feature_names)


            # 2. Create Sequences and Targets
            sequences = []
            cat_inputs = []
            binary_inputs = []
            temporal_inputs = []
            position_targets = []
            motion_targets = []
            status_targets = []

            # Ensure data is sorted for sequencing
            df_sorted = df.sort_values(['MMSI', 'BaseDateTime'])
            encoded_cat_df_sorted = encoded_cat_df.loc[df_sorted.index] # Keep encoded df aligned

            grouped = df_sorted.groupby('MMSI')
            for mmsi, vessel_data in grouped:
                if len(vessel_data) < self.sequence_length + 1:
                    continue # Skip short tracks

                # Extract features for this vessel
                seq_feature_values = vessel_data[self.sequential_continuous_features].values
                cat_feature_values = encoded_cat_df_sorted.loc[vessel_data.index].values # Use encoded values
                binary_feature_values = vessel_data[self.static_binary_features].values
                temporal_feature_values = vessel_data[self.static_temporal_features].values

                # Extract target values
                pos_target_values = vessel_data[['LAT', 'LON']].values
                mot_target_values = vessel_data[['SOG', 'COG', 'Heading']].values
                # Use the derived boolean flag directly as the target
                stat_target_values = vessel_data['is_underway'].values # Target: 1 if underway, 0 otherwise

                # Create sequences using sliding window
                for i in range(len(vessel_data) - self.sequence_length):
                    # Input sequence (continuous features)
                    sequences.append(seq_feature_values[i : i + self.sequence_length])

                    # Static inputs (taken from the *last* time step of the input sequence)
                    last_step_index = i + self.sequence_length - 1
                    cat_inputs.append(cat_feature_values[last_step_index])
                    binary_inputs.append(binary_feature_values[last_step_index])
                    temporal_inputs.append(temporal_feature_values[last_step_index])

                    # Targets (taken from the step *after* the input sequence ends)
                    target_index = i + self.sequence_length
                    position_targets.append(pos_target_values[target_index])
                    motion_targets.append(mot_target_values[target_index])
                    status_targets.append(stat_target_values[target_index])


            if not sequences:
                 logging.warning("No sequences generated during data preparation.")
                 # Return empty dicts or raise error
                 return {}, {}

            # Convert lists to numpy arrays
            model_inputs = {
                'sequential_input': np.array(sequences),
                'categorical_input': np.array(cat_inputs),
                'binary_input': np.array(binary_inputs),
                'temporal_input': np.array(temporal_inputs)
            }
            model_targets = {
                'position_output': np.array(position_targets),
                'motion_output': np.array(motion_targets),
                'status_output': np.array(status_targets)
            }

            logging.info(f"Data preparation complete. Generated {len(sequences)} sequences.")
            return model_inputs, model_targets

        except KeyError as e:
             logging.error(f"KeyError in prepare_data_for_lstm: Missing column {e}. Ensure input df has required columns.", exc_info=True)
             raise
        except Exception as e:
            logging.error(f"Error preparing data for LSTM: {e}", exc_info=True)
            raise

    def train(self,
              train_data: pd.DataFrame,
              feature_engineer: 'MaritimeFeatureEngineering',
              validation_data: Optional[pd.DataFrame] = None,
              epochs: int = 50,
              batch_size: int = 64 # Increased batch size
             ) -> tf.keras.callbacks.History:
        """
        Trains the LSTM model using prepared training data.

        Args:
            train_data (pd.DataFrame): DataFrame for training.
            feature_engineer (MaritimeFeatureEngineering): The feature engineering instance.
            validation_data (Optional[pd.DataFrame]): Optional DataFrame for validation during training.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.

        Returns:
            tf.keras.callbacks.History: Training history object.
        """
        if self.lstm_model is None:
             # Build the model only if it hasn't been built yet. Requires knowing the encoded cat feature count.
             # This implies prepare_data_for_lstm(fit_scalers=True) must be called first.
             if self.num_categorical_features_encoded is None:
                  raise RuntimeError("Model must be built before training. Call prepare_data_for_lstm with fit_scalers=True first.")
             self.build_lstm_model(self.num_categorical_features_encoded)


        logging.info("Starting model training...")
        try:
            # Prepare training data (fit scalers/encoders)
            train_inputs, train_targets = self.prepare_data_for_lstm(train_data, feature_engineer, fit_scalers=True)

            # Prepare validation data (use existing scalers/encoders)
            val_inputs, val_targets = None, None
            validation_set = None
            if validation_data is not None:
                logging.info("Preparing validation data...")
                val_inputs, val_targets = self.prepare_data_for_lstm(validation_data, feature_engineer, fit_scalers=False)
                if val_inputs: # Check if validation sequences were generated
                     validation_set = (
                         [val_inputs['sequential_input'], val_inputs['categorical_input'], val_inputs['binary_input'], val_inputs['temporal_input']],
                         [val_targets['position_output'], val_targets['motion_output'], val_targets['status_output']]
                     )
                else:
                     logging.warning("No validation sequences generated. Training without validation set.")


            # Define callbacks for better training
            early_stopping = EarlyStopping(
                monitor='val_loss', # Monitor validation loss
                patience=10,        # Stop after 10 epochs with no improvement
                restore_best_weights=True, # Restore weights from the best epoch
                verbose=1
            )
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2, # Reduce LR by a factor of 5
                patience=5, # Reduce LR after 5 epochs with no improvement
                min_lr=1e-6, # Minimum learning rate
                verbose=1
            )

            # Determine device context for training
            device = '/GPU:0' if self.use_gpu and tf.config.list_physical_devices('GPU') else '/CPU:0'
            logging.info(f"Training on device: {device}")

            with tf.device(device):
                history = self.lstm_model.fit(
                    # Provide inputs as a list matching the order in Model definition
                    [train_inputs['sequential_input'], train_inputs['categorical_input'], train_inputs['binary_input'], train_inputs['temporal_input']],
                    # Provide targets as a list matching the order in Model definition
                    [train_targets['position_output'], train_targets['motion_output'], train_targets['status_output']],
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=validation_set, # Pass validation data tuple
                    callbacks=[early_stopping, reduce_lr], # Add callbacks
                    verbose=1 # Show progress bar
                )

            self.is_trained = True
            logging.info("Model training completed.")
            return history

        except Exception as e:
            logging.error(f"Error during model training: {e}", exc_info=True)
            raise

    def predict(self,
                input_data: pd.DataFrame,
                feature_engineer: 'MaritimeFeatureEngineering'
               ) -> Optional[Dict[str, np.ndarray]]:
        """
        Makes predictions using the trained LSTM model.

        Args:
            input_data (pd.DataFrame): DataFrame with the necessary features for prediction.
                                       Should contain at least `sequence_length` rows for each vessel.
            feature_engineer (MaritimeFeatureEngineering): The fitted feature engineering instance.

        Returns:
            Optional[Dict[str, np.ndarray]]: Dictionary containing predictions for each output head,
                                             or None if prediction fails.
                                             Keys match output layer names.
        """
        if not self.is_trained or self.lstm_model is None:
            logging.error("Model is not trained yet. Cannot make predictions.")
            return None

        logging.info("Starting prediction...")
        try:
            # Prepare the input data using existing scalers/encoders
            # Note: For prediction, we typically need the *last* sequence available for each vessel.
            # prepare_data_for_lstm creates all possible sequences. We need a modified version
            # or filter the output to get only the latest sequence per vessel.
            # Let's assume prepare_data_for_lstm is adapted or we handle selection here.

            pred_inputs, _ = self.prepare_data_for_lstm(input_data, feature_engineer, fit_scalers=False)

            if not pred_inputs:
                 logging.warning("No input sequences generated for prediction.")
                 return None

            # Select the latest sequence for each vessel if multiple were generated (simplistic approach)
            # A more robust way would track indices during preparation.
            # This example assumes pred_inputs contains sequences for potentially multiple vessels,
            # and we want prediction for the *last* sequence in the prepared batch.
            # If predicting for a *single* vessel's latest state, ensure input_data is just that vessel's last sequence_length points.

            # Determine device context for prediction
            device = '/GPU:0' if self.use_gpu and tf.config.list_physical_devices('GPU') else '/CPU:0'
            logging.info(f"Predicting on device: {device}")

            with tf.device(device):
                predictions = self.lstm_model.predict(
                     [pred_inputs['sequential_input'], pred_inputs['categorical_input'], pred_inputs['binary_input'], pred_inputs['temporal_input']],
                     batch_size=64 # Use a reasonable batch size for prediction
                )

            # The output 'predictions' is a list of numpy arrays, one for each output head
            position_preds, motion_preds, status_preds = predictions

            logging.info("Prediction completed.")
            return {
                'position_pred': position_preds,
                'motion_pred': motion_preds,
                'status_pred': status_preds
            }

        except Exception as e:
            logging.error(f"Error during prediction: {e}", exc_info=True)
            return None

    def save_model(self, file_path: str):
        """Saves the trained Keras model."""
        if self.lstm_model and self.is_trained:
            try:
                self.lstm_model.save(file_path)
                logging.info(f"Model saved successfully to {file_path}")
            except Exception as e:
                logging.error(f"Error saving model to {file_path}: {e}", exc_info=True)
        else:
            logging.warning("Model is not trained or not built. Cannot save.")

    def load_model(self, file_path: str):
        """Loads a Keras model from a file."""
        try:
            self.lstm_model = tf.keras.models.load_model(file_path)
            self.is_trained = True # Assume loaded model is trained
            # Re-extract input/output layer info if needed, or assume it matches saved model
            logging.info(f"Model loaded successfully from {file_path}")
            # It might be necessary to re-extract feature counts if they weren't saved with the model structure implicitly
            # Or ensure the __init__ parameters match the loaded model structure.
        except Exception as e:
            logging.error(f"Error loading model from {file_path}: {e}", exc_info=True)
            self.lstm_model = None
            self.is_trained = False

# Environment classes (VesselEnvironment and EnhancedVesselEnvironment) remain unchanged...

class VesselEnvironment:
    """Basic environment for RL training"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.state = np.zeros(7)  # [lat, lon, sog, cog, target_lat, target_lon, distance_to_target]
        return self.state
    
    def step(self, action):
        # Simulate vessel movement based on action
        # action: [maintain, adjust_course, adjust_speed]
        
        # Update state based on action
        if action[1] > 0.5:  # adjust course
            self.state[3] += np.random.normal(0, 5)  # change course
        if action[2] > 0.5:  # adjust speed
            self.state[2] += np.random.normal(0, 1)  # change speed
            
        # Update position based on speed and course
        dt = 1.0  # time step
        self.state[0] += self.state[2] * np.cos(np.radians(self.state[3])) * dt
        self.state[1] += self.state[2] * np.sin(np.radians(self.state[3])) * dt
        
        # Calculate distance to target
        self.state[6] = np.sqrt(
            (self.state[0] - self.state[4])**2 + 
            (self.state[1] - self.state[5])**2
        )
        
        # Calculate reward
        reward = -self.state[6]  # negative distance as reward
        
        # Check if done
        done = self.state[6] < 0.1  # within 0.1 nautical miles
        
        return self.state, reward, done, {}

class EnhancedVesselEnvironment(VesselEnvironment):
    """Enhanced environment with realistic vessel behavior"""
    def __init__(self, vessel_data=None):
        super().__init__()
        self.vessel_data = vessel_data
        self.reset()
        
    def reset(self):
        if self.vessel_data is not None:
            # Initialize with real vessel data
            vessel = self.vessel_data.sample(1).iloc[0]
            self.state = np.array([
                vessel['LAT'],
                vessel['LON'],
                vessel['SOG'],
                vessel['COG'],
                vessel['Heading'],
                vessel['Length'],
                vessel['Width'],
                vessel['Draft'],
                0, 0,  # target position (will be set)
                0,     # distance to target
                0      # traffic density
            ])
        else:
            self.state = np.zeros(12)
        return self.state

    def step(self, action):
        # Enhanced physics model
        dt = 1.0
        
        # Update based on vessel characteristics
        length_factor = self.state[5] / 100  # Length impact on maneuverability
        draft_factor = self.state[7] / 10    # Draft impact on speed
        
        # Process actions with physical constraints
        if action[1] > 0.5:  # adjust course
            max_turn_rate = 20 * (1 - length_factor)  # longer vessels turn slower
            course_change = np.random.normal(0, max_turn_rate)
            self.state[3] += course_change
            
        if action[2] > 0.5:  # adjust speed
            max_speed_change = 2 * (1 - draft_factor)  # deeper draft = slower acceleration
            speed_change = np.random.normal(0, max_speed_change)
            self.state[2] = np.clip(self.state[2] + speed_change, 0, 30)  # max 30 knots
            
        # Update position with realistic motion model
        self.state[0] += self.state[2] * np.cos(np.radians(self.state[3])) * dt
        self.state[1] += self.state[2] * np.sin(np.radians(self.state[3])) * dt
        
        # Update distance to target
        self.state[10] = np.sqrt(
            (self.state[0] - self.state[8])**2 + 
            (self.state[1] - self.state[9])**2
        )
        
        # Enhanced reward function
        distance_reward = -self.state[10]
        efficiency_reward = -abs(speed_change) * 0.1 if 'speed_change' in locals() else 0
        stability_reward = -abs(course_change) * 0.1 if 'course_change' in locals() else 0
        
        reward = distance_reward + efficiency_reward + stability_reward
        
        done = self.state[10] < 0.1
        
        return self.state, reward, done, {
            'distance': self.state[10],
            'speed': self.state[2],
            'course': self.state[3]
        }

def train_models(data, use_gpu=True):
    """Train both LSTM and RL models"""
    predictor = DeepVesselPredictor(use_gpu=use_gpu)
    env = EnhancedVesselEnvironment(vessel_data=data)
    
    # Train LSTM
    print("Training LSTM model...")
    lstm_history = predictor.train(data)
    
    # Generate RL training data through simulation
    print("Generating RL training data...")
    states = []
    actions = []
    rewards = []
    
    n_episodes = 1000
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        while not done:
            action = predictor.get_action(state)
            next_state, reward, done, _ = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state

    # Train RL model
    # ... RL training logic ...

    return lstm_history