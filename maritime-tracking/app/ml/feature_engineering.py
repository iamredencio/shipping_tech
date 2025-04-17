import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from scipy.spatial import KDTree
import geopy.distance
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MaritimeFeatureEngineering:
    """
    Handles the preprocessing and feature engineering pipeline for maritime AIS data.
    Includes data validation, calculation of derived features (temporal, kinematic, vessel characteristics),
    and preparation of features for machine learning models (scaling, encoding).
    """
    def __init__(self):
        # Initialize standard scaler for continuous features (zero mean, unit variance)
        self.continuous_scaler = StandardScaler()
        # Initialize one-hot encoder for categorical features
        # sparse_output=False returns a dense numpy array
        # handle_unknown='ignore' skips unknown categories during transform
        self.categorical_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

        # Define lists of column names based on their type for easier processing
        # Base continuous features expected directly from AIS data
        self.continuous_features = [
            'LAT', 'LON', 'SOG', 'COG', 'Heading', # Core position and motion
            'Length', 'Width', 'Draft'             # Vessel dimensions
        ]

        # Base categorical features expected directly from AIS data
        self.categorical_features = [
            'VesselType', 'Status', 'Cargo', 'TransceiverClass'
        ]

        # Identifier columns - these should not be scaled or encoded
        self.identifier_features = [
            'MMSI', 'IMO', 'CallSign', 'VesselName', 'BaseDateTime' # Added BaseDateTime here
        ]

        # Predefined labels for categorical bins created during feature engineering
        self.speed_categories = ['Stationary', 'Slow', 'Medium', 'Fast', 'Very Fast']
        self.direction_categories = ['North', 'Northeast', 'East', 'Southeast', 'South', 'Southwest', 'West', 'Northwest'] # Updated for 8 directions
        self.size_categories = ['Very Small', 'Small', 'Medium', 'Large', 'Very Large']

        # Mapping from numerical VesselType codes (AIS standard) to broader categories
        # Example ranges, adjust based on specific AIS documentation/needs
        self.vessel_type_ranges = {
            'WIG': list(range(20, 30)), # Wing in ground effect
            'Pilot': [31],
            'SearchAndRescue': [32],
            'Tug': [33, 52], # Added 52
            'PortTender': [34],
            'AntiPollution': [35],
            'LawEnforcement': [36],
            'Medical': [37], # And spare
            'SpecialCraft': [50, 51, 53, 54, 55, 58, 59], # Added various special craft types
            'Passenger': list(range(60, 70)),
            'Cargo': list(range(70, 80)),
            'Tanker': list(range(80, 90)),
            'Fishing': [30],
            'Sailing': [38], # Corrected typo from 36
            'PleasureCraft': [39], # Corrected typo from 37
            'HSC': list(range(40, 50)), # High-speed craft
            'Other': list(range(90, 100)) + [0] # Includes 'Not available' or 'Reserved'
        }
        # Reverse mapping for easier lookup (code -> category name)
        self.vessel_type_map = {str(code): category for category, codes in self.vessel_type_ranges.items() for code in codes}


        # Valid navigational status codes (AIS standard)
        self.valid_status_codes = list(map(str, range(16))) # 0-15 are defined statuses
        # Mapping for readable status names
        self.status_map = {
            '0': 'Under way using engine', '1': 'At anchor', '2': 'Not under command',
            '3': 'Restricted manoeuverability', '4': 'Constrained by her draught',
            '5': 'Moored', '6': 'Aground', '7': 'Engaged in Fishing', '8': 'Under way sailing',
            '9': 'Reserved for future amendment (HSC)', '10': 'Reserved for future amendment (WIG)',
            '11': 'Reserved for future use', '12': 'Reserved for future use',
            '13': 'Reserved for future use', '14': 'AIS-SART is active',
            '15': 'Not defined' # Default/unknown
        }


    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validates the input DataFrame to ensure required columns exist and have plausible data types.

        Args:
            df (pd.DataFrame): The input DataFrame containing AIS data.

        Returns:
            bool: True if validation passes.

        Raises:
            ValueError: If required columns are missing or if BaseDateTime cannot be parsed.
        """
        logging.info("Validating input data...")
        # Define the minimum set of columns required for basic processing
        required_columns = ['MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'COG', 'Heading']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Check if BaseDateTime is already a datetime type
        if 'BaseDateTime' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['BaseDateTime']):
            logging.warning("BaseDateTime column is not datetime type. Attempting conversion.")
            # Attempt conversion, raise error if it fails
            try:
                # Example: Try parsing common formats if needed, or just pd.to_datetime
                pd.to_datetime(df['BaseDateTime']) # Test conversion without modifying df yet
                logging.info("BaseDateTime conversion test successful.")
            except Exception as e:
                raise ValueError(f"Invalid DateTime format in BaseDateTime column: {e}")

        # Add more checks: e.g., Lat/Lon ranges, SOG/COG ranges if necessary
        if 'LAT' in df.columns and not df['LAT'].between(-90, 90).all():
             logging.warning("Latitude values outside expected range [-90, 90] detected.")
        if 'LON' in df.columns and not df['LON'].between(-180, 180).all():
             logging.warning("Longitude values outside expected range [-180, 180] detected.")
        if 'SOG' in df.columns and not df['SOG'].between(0, 102.3).all(): # AIS max SOG is 102.3 knots
             logging.warning("SOG values outside expected range [0, 102.3] detected.")
        if 'COG' in df.columns and not df['COG'].between(0, 360).all():
             logging.warning("COG values outside expected range [0, 360] detected.")
        if 'Heading' in df.columns and not df['Heading'].between(0, 359).all(): # Heading 0-359, 511=unavailable
             # Allow 511 for unavailable heading
             if not df['Heading'].apply(lambda x: 0 <= x <= 359 or x == 511).all():
                 logging.warning("Heading values outside expected range [0, 359] or 511 detected.")


        logging.info("Data validation successful.")
        return True

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Fills missing values using appropriate strategies. """
        logging.info("Handling missing values...")
        processed = df.copy()

        # Fill missing continuous features with median (robust to outliers) or mean
        for feature in self.continuous_features:
            if feature in processed.columns and processed[feature].isnull().any():
                median_val = processed[feature].median()
                processed[feature] = processed[feature].fillna(median_val)
                logging.debug(f"Filled NaNs in '{feature}' with median value: {median_val}")

        # Fill missing categorical features with a placeholder like 'Unknown' or the mode
        for feature in self.categorical_features:
             if feature in processed.columns and processed[feature].isnull().any():
                 # Convert to string first to handle mixed types if necessary
                 processed[feature] = processed[feature].astype(str).fillna('Unknown')
                 logging.debug(f"Filled NaNs in '{feature}' with 'Unknown'")

        # Special handling for Heading=511 (unavailable) -> NaN might be better for calculations
        if 'Heading' in processed.columns:
            processed['Heading'] = processed['Heading'].replace(511, np.nan)
            # Optionally fill NaN Heading, e.g., with COG or median/mean if appropriate
            if processed['Heading'].isnull().any():
                 # Example: Fill with COG if available, otherwise median
                 if 'COG' in processed.columns:
                     processed['Heading'] = processed['Heading'].fillna(processed['COG'])
                 median_heading = processed['Heading'].median() # Calculate after potential COG fill
                 processed['Heading'] = processed['Heading'].fillna(median_heading)
                 logging.debug(f"Filled NaN Heading values (potentially from 511 or original NaNs).")


        # Handle potential NaNs introduced in derived features later if needed
        logging.info("Missing value handling complete.")
        return processed


    def process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main pipeline function to process AIS data.
        Validates, handles missing values, converts types, and calculates derived features.

        Args:
            df (pd.DataFrame): Input DataFrame with raw AIS data.

        Returns:
            pd.DataFrame: Processed DataFrame with original and derived features.
        """
        logging.info(f"Starting feature processing for DataFrame with shape {df.shape}.")
        # 1. Validate data
        self.validate_data(df)
        processed = df.copy()

        # 2. Convert BaseDateTime if not already datetime
        if 'BaseDateTime' in processed.columns and not pd.api.types.is_datetime64_any_dtype(processed['BaseDateTime']):
            processed['BaseDateTime'] = pd.to_datetime(processed['BaseDateTime'])
            logging.info("Converted 'BaseDateTime' column to datetime objects.")

        # 3. Convert continuous features to numeric, coercing errors to NaN
        logging.info("Converting continuous features to numeric...")
        for feature in self.continuous_features:
            if feature in processed.columns:
                processed[feature] = pd.to_numeric(processed[feature], errors='coerce')

        # 4. Handle missing values (including those from coercion errors)
        processed = self._handle_missing_values(processed)

        # 5. Calculate derived features
        logging.info("Calculating derived features...")
        processed = self.calculate_derived_features(processed)
        logging.info(f"Finished feature processing. DataFrame shape: {processed.shape}.")

        # 6. Final check for NaNs introduced by derived feature calculations
        if processed.isnull().any().any():
             logging.warning(f"NaN values detected after calculating derived features. Check calculations (e.g., diff()). Filling remaining NaNs with 0 or appropriate value.")
             # Example: Fill any remaining numeric NaNs with 0
             numeric_cols = processed.select_dtypes(include=np.number).columns
             processed[numeric_cols] = processed[numeric_cols].fillna(0)


        return processed

    def calculate_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates various derived features based on the processed AIS data.
        Includes time-based, kinematic (movement), vessel characteristics, and status features.

        Args:
            df (pd.DataFrame): DataFrame with validated and type-converted AIS data.
                               Must include 'BaseDateTime', 'MMSI', and relevant base features.

        Returns:
            pd.DataFrame: DataFrame with added derived feature columns.
        """
        try:
            derived = df.copy()
            # Ensure data is sorted by vessel and time for accurate diff() calculations
            derived = derived.sort_values(['MMSI', 'BaseDateTime'])

            logging.debug("Calculating time-based features...")
            # Time-based features
            derived['hour'] = derived['BaseDateTime'].dt.hour
            derived['day_of_week'] = derived['BaseDateTime'].dt.dayofweek # Monday=0, Sunday=6
            derived['month'] = derived['BaseDateTime'].dt.month
            derived['is_weekend'] = derived['day_of_week'].isin([5, 6]) # Saturday or Sunday
            # is_night: Check if hour is between 6 PM (18) and 6 AM (6) inclusive? Adjust as needed.
            # This logic might be tricky across midnight. A simpler way:
            derived['is_night'] = ~derived['hour'].between(7, 17) # Assume day is 7 AM to 5 PM

            # Calculate time difference between consecutive messages for rate calculations
            # Group by vessel, calculate time diff, convert to seconds
            derived['time_diff_seconds'] = derived.groupby('MMSI')['BaseDateTime'].diff().dt.total_seconds()
            # Fill NaN for the first message of each vessel, maybe with a default (e.g., average interval) or 0
            derived['time_diff_seconds'] = derived['time_diff_seconds'].fillna(0) # Or median/mean if preferred


            logging.debug("Calculating vessel movement features...")
            # Vessel movement features (require SOG, COG, Heading, time_diff_seconds)
            if 'SOG' in derived.columns:
                # Speed categories using predefined bins and labels
                speed_bins = [-np.inf, 0.1, 5, 10, 15, np.inf] # Bins: (..., 0.1], (0.1, 5], ..., (15, ...]
                derived['speed_category'] = pd.cut(
                    derived['SOG'],
                    bins=speed_bins,
                    labels=self.speed_categories,
                    right=True, # Intervals are closed on the right
                    ordered=False # Treat as nominal categories
                )

                # Acceleration: Change in speed over time difference
                # Ensure time_diff is not zero to avoid division by zero
                sog_diff = derived.groupby('MMSI')['SOG'].diff().fillna(0)
                derived['acceleration'] = np.where(
                    derived['time_diff_seconds'] > 0,
                    sog_diff / derived['time_diff_seconds'],
                    0 # Assign 0 acceleration if time diff is 0 or first point
                )


            if all(col in derived.columns for col in ['COG', 'Heading']):
                 # Handle potential circular nature of angles (e.g., 359 deg vs 1 deg)
                 # Course deviation: Difference between COG and Heading
                 # Calculate angular difference correctly (shortest angle)
                 angle_diff = derived['COG'] - derived['Heading']
                 derived['course_deviation'] = np.abs((angle_diff + 180) % 360 - 180)


                 # Turn rate: Change in COG over time difference
                 # Need careful handling of angle wrapping (e.g., 350 -> 10 is +20 deg turn)
                 cog_diff = derived.groupby('MMSI')['COG'].diff().fillna(0)
                 # Correct for angle wrapping
                 cog_diff_corrected = (cog_diff + 180) % 360 - 180
                 derived['turn_rate'] = np.where(
                     derived['time_diff_seconds'] > 0,
                     cog_diff_corrected / derived['time_diff_seconds'], # degrees per second
                     0
                 )

                 # Movement direction categories (8 directions) based on COG
                 # Bins: (337.5, 22.5], (22.5, 67.5], ..., (292.5, 337.5]
                 direction_bins = [-np.inf] + list(np.arange(22.5, 360, 45)) + [np.inf]
                 # Use COG % 360 to handle potential values slightly outside 0-360
                 derived['movement_direction'] = pd.cut(
                     derived['COG'] % 360,
                     bins=direction_bins,
                     labels=self.direction_categories,
                     right=False, # Intervals are [a, b) - important for 0/360 boundary
                     ordered=False
                 )
                 # Handle the wrap-around case for North (assign first bin label)
                 derived.loc[derived['COG'] % 360 >= 337.5, 'movement_direction'] = self.direction_categories[0] # North


            logging.debug("Calculating vessel characteristic features...")
            # Vessel characteristics features (require Length, Width, Draft)
            if all(col in derived.columns for col in ['Length', 'Width', 'Draft']):
                # Size ratios, protect against division by zero if Width or Length can be 0
                # Replace 0 with NaN before division, then fill resulting NaN if needed
                derived['length_width_ratio'] = derived['Length'] / derived['Width'].replace(0, np.nan)
                derived['draft_ratio'] = derived['Draft'] / derived['Length'].replace(0, np.nan)
                # Fill potential NaNs from division by zero, e.g., with 0 or median
                derived['length_width_ratio'] = derived['length_width_ratio'].fillna(0)
                derived['draft_ratio'] = derived['draft_ratio'].fillna(0)


                # Size categories based on Length
                size_bins = [-np.inf, 50, 100, 150, 200, np.inf] # Bins: (..., 50], (50, 100], ...
                derived['size_category'] = pd.cut(
                    derived['Length'],
                    bins=size_bins,
                    labels=self.size_categories,
                    right=True,
                    ordered=False
                )

            logging.debug("Calculating status and type features...")
            # Status and type features (require Status, VesselType, TransceiverClass)
            if 'Status' in derived.columns:
                # Ensure Status is string for consistent mapping/checking
                status_str = derived['Status'].astype(str)
                # Map status code to readable name
                derived['status_name'] = status_str.map(self.status_map).fillna('Not defined')
                # Create boolean flags based on status
                derived['is_underway'] = status_str.isin(['0', '8']) # Under way engine or sailing
                derived['is_at_anchor'] = status_str == '1'
                derived['is_moored'] = status_str == '5'
                derived['is_fishing'] = status_str == '7'
                derived['is_restricted'] = status_str.isin(['2', '3', '4', '6']) # NUC, Restricted, Constrained, Aground


            if 'VesselType' in derived.columns:
                 # Ensure VesselType is string
                 vessel_type_str = derived['VesselType'].astype(str)
                 # Map type code to category name
                 derived['vessel_category'] = vessel_type_str.map(self.vessel_type_map).fillna('Other')
                 # Create boolean flags for major categories
                 derived['is_passenger'] = derived['vessel_category'] == 'Passenger'
                 derived['is_cargo'] = derived['vessel_category'] == 'Cargo'
                 derived['is_tanker'] = derived['vessel_category'] == 'Tanker'
                 derived['is_fishing_vessel'] = derived['vessel_category'] == 'Fishing' # Renamed from is_fishing


            if 'TransceiverClass' in derived.columns:
                # Boolean flags for AIS transceiver class (A or B)
                derived['is_class_a'] = derived['TransceiverClass'].astype(str).str.upper() == 'A'
                derived['is_class_b'] = derived['TransceiverClass'].astype(str).str.upper() == 'B'


            # Fill NaNs that might have been created in categorical derived features
            # (e.g., from pd.cut if input was NaN, or map if key not found)
            categorical_derived_cols = ['speed_category', 'movement_direction', 'size_category', 'status_name', 'vessel_category']
            for col in categorical_derived_cols:
                 if col in derived.columns and derived[col].isnull().any():
                     # Convert to string before filling if not already
                     if pd.api.types.is_categorical_dtype(derived[col]):
                         derived[col] = derived[col].astype(str).fillna('Unknown')
                     else:
                         derived[col] = derived[col].fillna('Unknown')


            logging.debug("Finished calculating derived features.")
            return derived

        except Exception as e:
            logging.error(f"Error in calculate_derived_features: {e}", exc_info=True) # Log traceback
            raise # Re-raise the exception


    def prepare_model_features(self, df: pd.DataFrame, fit_scalers: bool = True) -> Dict[str, np.ndarray]:
        """
        Prepares the final feature set for input into a machine learning model.
        Selects relevant features, scales continuous ones, and encodes categorical ones.

        Args:
            df (pd.DataFrame): The DataFrame containing all processed features.
            fit_scalers (bool): If True, fit the scalers/encoders. If False, only transform (use for test/prediction data).

        Returns:
            Dict[str, np.ndarray]: A dictionary containing scaled continuous features,
                                   encoded categorical features, and feature names.
                                   Example: {'continuous': ndarray, 'categorical': ndarray, 'feature_names': {...}}
        """
        logging.info(f"Preparing model features. Fit scalers: {fit_scalers}")
        try:
            # Select only the base continuous and categorical features for scaling/encoding
            # Derived features might be used directly or might need separate handling depending on the model
            continuous_data = df[self.continuous_features].values
            categorical_data = df[self.categorical_features].astype(str).values # Ensure string type for encoder

            if fit_scalers:
                # Fit and transform the scalers/encoders
                scaled_continuous = self.continuous_scaler.fit_transform(continuous_data)
                encoded_categorical = self.categorical_encoder.fit_transform(categorical_data)
                logging.info("Fitted scalers and encoders.")
            else:
                # Only transform using previously fitted scalers/encoders
                scaled_continuous = self.continuous_scaler.transform(continuous_data)
                encoded_categorical = self.categorical_encoder.transform(categorical_data)
                logging.info("Transformed data using existing scalers/encoders.")

            # Get feature names after one-hot encoding
            encoded_categorical_names = self.categorical_encoder.get_feature_names_out(self.categorical_features).tolist()

            # Combine scaled continuous and encoded categorical features into a single array (optional)
            # model_input_features = np.hstack((scaled_continuous, encoded_categorical))
            # all_feature_names = self.continuous_features + encoded_categorical_names

            logging.info("Model features prepared successfully.")
            return {
                'continuous': scaled_continuous,
                'categorical': encoded_categorical,
                # 'combined': model_input_features, # Optionally return combined array
                'feature_names': {
                    'continuous': self.continuous_features,
                    'categorical_encoded': encoded_categorical_names,
                    # 'combined': all_feature_names
                }
            }
        except Exception as e:
            logging.error(f"Error in prepare_model_features: {e}", exc_info=True)
            raise


    def prepare_sequence_data(self, df: pd.DataFrame, sequence_length: int = 10, features_to_use: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepares sequence data suitable for recurrent models like LSTMs.
        Creates sequences of features and corresponding targets (e.g., next position).

        Args:
            df (pd.DataFrame): DataFrame with processed features, sorted by MMSI and BaseDateTime.
            sequence_length (int): The number of time steps in each input sequence.
            features_to_use (Optional[List[str]]): List of feature column names to include in the sequences.
                                                   If None, uses self.continuous_features.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                                           - sequences (np.ndarray): Shape (num_sequences, sequence_length, num_features)
                                           - targets (np.ndarray): Shape (num_sequences, num_target_dims) e.g., (num_sequences, 2) for Lat/Lon
        """
        logging.info(f"Preparing sequence data with sequence length {sequence_length}...")
        try:
            # Ensure data is sorted correctly
            df_sorted = df.sort_values(['MMSI', 'BaseDateTime'])

            # Select features for the sequences
            if features_to_use is None:
                features_to_use = self.continuous_features # Default to continuous features
            logging.debug(f"Using features for sequences: {features_to_use}")

            # --- Scaling should happen *before* creating sequences ---
            # Apply scaling to the selected features across the entire dataset first
            # Note: This assumes scaling is desired for sequence features.
            # If using prepare_model_features output, sequences would be built from scaled data.
            # Let's assume df contains *unscaled* features for now, and scale here.
            scaler = StandardScaler() # Use a local scaler or the class one if appropriate
            df_sorted[features_to_use] = scaler.fit_transform(df_sorted[features_to_use])
            logging.info("Scaled features selected for sequences.")
            # --- End Scaling ---


            all_sequences = []
            all_targets = []

            # Group by vessel to create sequences only within each vessel's track
            grouped = df_sorted.groupby('MMSI')

            for mmsi, vessel_data in grouped:
                # Check if vessel track is long enough to create at least one sequence + target
                if len(vessel_data) < sequence_length + 1:
                    logging.debug(f"Skipping MMSI {mmsi}: track length {len(vessel_data)} is less than sequence_length+1 ({sequence_length+1})")
                    continue

                # Extract the feature values and target values (e.g., next Lat/Lon)
                feature_values = vessel_data[features_to_use].values
                # Target is typically the state at the step *after* the sequence ends
                target_values = vessel_data[['LAT', 'LON']].values # Example: Predict next Lat/Lon

                # Use sliding window to create sequences
                for i in range(len(vessel_data) - sequence_length):
                    # Input sequence: features from index i to i + sequence_length - 1
                    sequence = feature_values[i : i + sequence_length]
                    # Target: Lat/Lon at index i + sequence_length
                    target = target_values[i + sequence_length]

                    all_sequences.append(sequence)
                    all_targets.append(target)

            if not all_sequences:
                # Handle case where no sequences could be created (e.g., all tracks too short)
                logging.warning("No valid sequences could be created from the provided data.")
                # Return empty arrays or raise error, depending on desired behavior
                return np.array([]), np.array([])
                # raise ValueError("No valid sequences could be created from the data")

            logging.info(f"Created {len(all_sequences)} sequences.")
            # Convert lists of sequences and targets to numpy arrays
            return np.array(all_sequences), np.array(all_targets)

        except KeyError as e:
             logging.error(f"KeyError in prepare_sequence_data: Missing column {e}. Ensure input df has required columns.", exc_info=True)
             raise
        except Exception as e:
            logging.error(f"Error in prepare_sequence_data: {e}", exc_info=True)
            raise

    # Example method to add weather features (if weather data is available)
    def add_weather_features(self, ais_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merges weather data onto AIS data based on timestamp and location (nearest).
        Requires weather_df with 'Timestamp', 'Lat', 'Lon', and weather features.
        NOTE: This is a placeholder and needs a proper implementation (e.g., using spatial indexing).
        """
        logging.warning("add_weather_features is a placeholder and needs implementation.")
        # Placeholder: merge based on exact timestamp (unlikely to work well)
        # A proper implementation would involve time interpolation and spatial nearest neighbor search.
        # merged_df = pd.merge_asof(ais_df.sort_values('BaseDateTime'),
        #                           weather_df.sort_values('Timestamp'),
        #                           left_on='BaseDateTime', right_on='Timestamp',
        #                           direction='nearest', tolerance=pd.Timedelta('1hour'))
        # return merged_df
        return ais_df # Return unchanged df for now

    def get_feature_info(self) -> Dict:
        """Returns a dictionary describing the features handled by this class."""
        # Generate list of derived feature names dynamically if possible, or list manually
        # This requires knowing which columns are added in calculate_derived_features
        derived_feature_names = [
            'hour', 'day_of_week', 'month', 'is_weekend', 'is_night', 'time_diff_seconds',
            'speed_category', 'acceleration', 'course_deviation', 'turn_rate', 'movement_direction',
            'length_width_ratio', 'draft_ratio', 'size_category',
            'status_name', 'is_underway', 'is_at_anchor', 'is_moored', 'is_fishing', 'is_restricted',
            'vessel_category', 'is_passenger', 'is_cargo', 'is_tanker', 'is_fishing_vessel',
            'is_class_a', 'is_class_b'
            # Add any other derived features here
        ]
        # Get encoded names if encoder has been fitted
        encoded_categorical_names = []
        if hasattr(self.categorical_encoder, 'get_feature_names_out'):
             try:
                 encoded_categorical_names = self.categorical_encoder.get_feature_names_out(self.categorical_features).tolist()
             except Exception: # NotFittedError etc.
                 pass


        return {
            'base_continuous': self.continuous_features,
            'base_categorical': self.categorical_features,
            'identifiers': self.identifier_features,
            'derived': derived_feature_names,
            'categorical_encoded': encoded_categorical_names,
            'discretization_bins': {
                'speed': [-np.inf, 0.1, 5, 10, 15, np.inf],
                'direction': [-np.inf] + list(np.arange(22.5, 360, 45)) + [np.inf],
                'size': [-np.inf, 50, 100, 150, 200, np.inf]
            },
            'category_labels': {
                'speed': self.speed_categories,
                'direction': self.direction_categories,
                'size': self.size_categories
            }
        }