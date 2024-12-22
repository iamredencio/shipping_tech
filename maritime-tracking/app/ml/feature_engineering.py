import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
from scipy.spatial import KDTree
import geopy.distance
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import logging

class MaritimeFeatureEngineering:
    """
    Feature engineering pipeline for maritime vessel tracking data
    Handles data preprocessing, feature extraction, and validation
    """
    def __init__(self):
        # Initialize scalers and encoders for feature preprocessing
        self.continuous_scaler = StandardScaler()
        self.categorical_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        
        # Define feature groups for different data types
        self.continuous_features = [
            'LAT', 'LON', 'SOG', 'COG', 'Heading',
            'Length', 'Width', 'Draft'
        ]
        
        self.categorical_features = [
            'VesselType', 'Status', 'Cargo', 'TransceiverClass'
        ]
        
        # Identifier features that should not be transformed
        self.identifier_features = [
            'MMSI', 'IMO', 'CallSign', 'VesselName'
        ]

        # Predefined categories for discretization
        self.speed_categories = ['Stationary', 'Slow', 'Medium', 'Fast', 'Very Fast']
        self.direction_categories = ['North', 'East', 'South', 'West', 'North']
        self.size_categories = ['Very Small', 'Small', 'Medium', 'Large', 'Very Large']
        
        # Define vessel type mappings
        self.vessel_type_ranges = {
            'passenger': list(range(60, 65)),
            'cargo': list(range(70, 80)),
            'tanker': list(range(80, 90))
        }
        
        # Define valid status codes
        self.valid_status_codes = ['0', '1', '2', '3', '4', '5']

    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate input data before processing"""
        required_columns = ['MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'COG', 'Heading']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        if not pd.api.types.is_datetime64_any_dtype(df['BaseDateTime']):
            try:
                pd.to_datetime(df['BaseDateTime'])
            except Exception as e:
                raise ValueError(f"Invalid DateTime format in BaseDateTime column: {e}")
        
        return True

    def process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process all features from AIS data"""
        # Validate data first
        self.validate_data(df)
        processed = df.copy()
        
        # Convert BaseDateTime if needed
        if not pd.api.types.is_datetime64_any_dtype(processed['BaseDateTime']):
            processed['BaseDateTime'] = pd.to_datetime(processed['BaseDateTime'])
        
        # Process continuous features
        for feature in self.continuous_features:
            if feature in processed.columns:
                processed[feature] = pd.to_numeric(processed[feature], errors='coerce')
                processed[feature] = processed[feature].fillna(processed[feature].median())

        # Calculate derived features
        processed = self.calculate_derived_features(processed)

        return processed

    def calculate_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived features from AIS data"""
        try:
            derived = df.copy()
            
            # Time-based features
            derived['hour'] = derived['BaseDateTime'].dt.hour
            derived['day_of_week'] = derived['BaseDateTime'].dt.dayofweek
            derived['month'] = derived['BaseDateTime'].dt.month
            derived['is_night'] = derived['hour'].between(18, 6)
            
            # Vessel movement features
            if 'SOG' in derived.columns:
                # Speed categories with fixed bins
                speed_bins = [-np.inf, 0.1, 5, 10, 15, np.inf]  # 6 edges for 5 categories
                derived['speed_category'] = pd.cut(
                    derived['SOG'],
                    bins=speed_bins,
                    labels=self.speed_categories,
                    ordered=False
                )
                
                # Acceleration
                derived['acceleration'] = derived.groupby('MMSI')['SOG'].diff() / \
                                        derived.groupby('MMSI')['BaseDateTime'].diff().dt.total_seconds()
                derived['acceleration'] = derived['acceleration'].fillna(0)

            # Course and heading features
            if all(col in derived.columns for col in ['COG', 'Heading']):
                # Course deviation
                derived['course_deviation'] = abs(derived['COG'] - derived['Heading'])
                
                # Turn rate
                derived['turn_rate'] = derived.groupby('MMSI')['COG'].diff() / \
                                     derived.groupby('MMSI')['BaseDateTime'].diff().dt.total_seconds()
                derived['turn_rate'] = derived['turn_rate'].fillna(0)
                
                # Movement direction
                direction_bins = [-np.inf, 45, 135, 225, 315, np.inf]  # 6 edges for 5 categories
                derived['movement_direction'] = pd.cut(
                    derived['COG'] % 360,
                    bins=direction_bins,
                    labels=self.direction_categories,
                    ordered=False
                )

            # Vessel characteristics features
            if all(col in derived.columns for col in ['Length', 'Width', 'Draft']):
                # Size ratios with protection against zero division
                derived['length_width_ratio'] = derived['Length'] / derived['Width'].replace(0, np.nan)
                derived['draft_ratio'] = derived['Draft'] / derived['Length'].replace(0, np.nan)
                
                # Size categories
                size_bins = [0, 50, 100, 150, 200, np.inf]  # 6 edges for 5 categories
                derived['size_category'] = pd.cut(
                    derived['Length'],
                    bins=size_bins,
                    labels=self.size_categories,
                    ordered=False
                )

            # Status and type features
            if 'Status' in derived.columns:
                status_str = derived['Status'].astype(str)
                derived['is_underway'] = status_str.isin(self.valid_status_codes)
                derived['is_anchored'] = status_str.isin(['1', '5'])
                derived['is_moored'] = status_str.isin(['5'])
                derived['is_restricted'] = status_str.isin(['3', '4'])

            # Cargo and vessel type features
            if 'VesselType' in derived.columns:
                vessel_type_str = derived['VesselType'].astype(str)
                derived['is_passenger'] = vessel_type_str.isin([str(x) for x in self.vessel_type_ranges['passenger']])
                derived['is_cargo'] = vessel_type_str.isin([str(x) for x in self.vessel_type_ranges['cargo']])
                derived['is_tanker'] = vessel_type_str.isin([str(x) for x in self.vessel_type_ranges['tanker']])

            # TransceiverClass features
            if 'TransceiverClass' in derived.columns:
                derived['is_class_a'] = derived['TransceiverClass'] == 'A'
                derived['is_class_b'] = derived['TransceiverClass'] == 'B'

            # Handle NaN values
            # For numeric columns
            numeric_cols = derived.select_dtypes(include=[np.number]).columns
            derived[numeric_cols] = derived[numeric_cols].fillna(0)
            
            # For categorical columns
            categorical_cols = derived.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                if col not in self.identifier_features:  # Don't modify identifier columns
                    derived[col] = derived[col].astype(str)
                    derived[col] = derived[col].fillna('Unknown')

            return derived
            
        except Exception as e:
            logging.error(f"Error in calculate_derived_features: {e}")
            raise

    def prepare_model_features(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Prepare features for model training"""
        try:
            # Scale continuous features
            continuous_data = df[self.continuous_features].values
            scaled_continuous = self.continuous_scaler.fit_transform(continuous_data)
            
            # Encode categorical features
            categorical_data = df[self.categorical_features].values
            encoded_categorical = self.categorical_encoder.fit_transform(categorical_data)
            
            return {
                'continuous': scaled_continuous,
                'categorical': encoded_categorical,
                'feature_names': {
                    'continuous': self.continuous_features,
                    'categorical': self.categorical_encoder.get_feature_names_out(self.categorical_features).tolist()
                }
            }
        except Exception as e:
            logging.error(f"Error in prepare_model_features: {e}")
            raise

    def prepare_sequence_data(self, df: pd.DataFrame, sequence_length: int = 10) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Prepare sequence data for model training"""
        try:
            df_sorted = df.sort_values(['MMSI', 'BaseDateTime'])
            
            sequences_continuous = []
            sequences_categorical = []
            targets = []
            
            for mmsi in df_sorted['MMSI'].unique():
                vessel_data = df_sorted[df_sorted['MMSI'] == mmsi]
                
                if len(vessel_data) < sequence_length + 1:
                    continue
                    
                # Process features for this vessel
                processed_features = self.prepare_model_features(vessel_data)
                
                for i in range(len(vessel_data) - sequence_length):
                    # Get sequence of continuous features
                    seq_continuous = processed_features['continuous'][i:i+sequence_length]
                    sequences_continuous.append(seq_continuous)
                    
                    # Get categorical features (use last timestep's values)
                    seq_categorical = processed_features['categorical'][i+sequence_length-1]
                    sequences_categorical.append(seq_categorical)
                    
                    # Target is next position
                    target = vessel_data[['LAT', 'LON']].iloc[i+sequence_length].values
                    targets.append(target)
            
            if not sequences_continuous:
                raise ValueError("No valid sequences could be created from the data")
                
            return {
                'continuous': np.array(sequences_continuous),
                'categorical': np.array(sequences_categorical)
            }, np.array(targets)
            
        except Exception as e:
            logging.error(f"Error in prepare_sequence_data: {e}")
            raise

    def get_feature_info(self) -> Dict:
        """Get information about available features"""
        return {
            'continuous_features': self.continuous_features,
            'categorical_features': self.categorical_features,
            'identifier_features': self.identifier_features,
            'derived_features': [
                'time_features',
                'movement_features',
                'vessel_characteristics',
                'status_features',
                'cargo_features',
                'transceiver_features'
            ]
        }