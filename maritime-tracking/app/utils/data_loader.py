import pandas as pd
import numpy as np
from datetime import datetime

class AISDataLoader:
    """Handles loading and preprocessing of AIS data"""
    def __init__(self, file_path='AIS_2020_01_01.csv'):
        self.file_path = file_path
        self.data = None
        
    def load_data(self, nrows=None):
        """Load AIS data with correct datatypes"""
        print(f"Loading data from {self.file_path}")
        
        dtypes = {
            'MMSI': str,
            'LAT': float,
            'LON': float,
            'SOG': float,
            'COG': float,
            'Heading': float,
            'VesselName': str,
            'IMO': str,
            'CallSign': str,
            'VesselType': 'Int64',  # nullable integer
            'Status': 'Int64',      # nullable integer
            'Length': float,
            'Width': float,
            'Draft': float,
            'Cargo': str,
            'TransceiverClass': str
        }
        
        try:
            self.data = pd.read_csv(
                self.file_path,
                dtype=dtypes,
                parse_dates=['BaseDateTime'],
                nrows=nrows
            )
            
            print("\nData Summary:")
            print(f"Total records: {len(self.data)}")
            print(f"Unique vessels: {self.data['MMSI'].nunique()}")
            print(f"Time range: {self.data['BaseDateTime'].min()} to {self.data['BaseDateTime'].max()}")
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
            
    def get_vessel_data(self, mmsi):
        """Get all data for a specific vessel"""
        if self.data is None:
            return None
        return self.data[self.data['MMSI'] == mmsi].sort_values('BaseDateTime')

    def get_state_vector(self, row):
        """Convert a row to state vector format"""
        return np.array([
            row['LAT'],
            row['LON'],
            row['SOG'],
            row['COG'],
            row['Heading'],
            row['Length'],
            row['Width'],
            row['Draft']
        ])