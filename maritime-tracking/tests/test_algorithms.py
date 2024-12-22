import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from app.ml.deep_learning import DeepVesselPredictor, EnhancedVesselEnvironment
from app.ml.feature_engineering import MaritimeFeatureEngineering

@pytest.fixture
def sample_vessel_data():
    """Create sample vessel data for testing"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='5min')
    
    return pd.DataFrame({
        'MMSI': ['123456789'] * 100,
        'BaseDateTime': dates,
        'LAT': 38.25 + np.random.normal(0, 0.001, 100),
        'LON': -76.29 + np.random.normal(0, 0.001, 100),
        'SOG': 10 + np.random.normal(0, 0.5, 100),
        'COG': 45 + np.random.normal(0, 1, 100),
        'Heading': 45 + np.random.normal(0, 1, 100),
        'VesselName': ['Test Vessel'] * 100,
        'IMO': ['IMO123456'] * 100,
        'CallSign': ['TEST123'] * 100,
        'VesselType': [70] * 100,
        'Status': [0] * 100,
        'Length': [100] * 100,
        'Width': [20] * 100,
        'Draft': [5] * 100,
        'Cargo': ['General Cargo'] * 100,
        'TransceiverClass': ['A'] * 100
    })

def test_deep_vessel_predictor_initialization():
    """Test DeepVesselPredictor initialization"""
    predictor = DeepVesselPredictor(use_gpu=False)
    assert predictor is not None
    assert predictor.lstm_model is not None
    assert predictor.rl_model is not None

def test_feature_engineering(sample_vessel_data):
    """Test feature engineering pipeline"""
    feature_eng = MaritimeFeatureEngineering()
    processed_data = feature_eng.process_features(sample_vessel_data)
    
    # Check derived features exist
    assert 'speed_to_length_ratio' in processed_data.columns
    assert 'course_deviation' in processed_data.columns
    assert 'is_underway' in processed_data.columns

def test_lstm_prediction(sample_vessel_data):
    """Test LSTM prediction"""
    predictor = DeepVesselPredictor(use_gpu=False)
    
    # Prepare data
    processed_data = predictor.prepare_enhanced_data(sample_vessel_data)
    
    # Train model with small number of epochs for testing
    history = predictor.train(sample_vessel_data, epochs=2)
    
    # Make prediction
    sequence = processed_data['inputs']['sequences'][0]
    cat_data = processed_data['inputs']['categorical'][0]
    binary_data = processed_data['inputs']['binary'][0]
    temporal_data = processed_data['inputs']['temporal'][0]
    
    position, motion, status = predictor.predict_trajectory(
        sequence, cat_data, binary_data, temporal_data
    )
    
    assert position.shape == (2,)  # LAT, LON
    assert motion.shape == (3,)    # SOG, COG, Heading
    assert isinstance(status, (float, np.ndarray))

def test_vessel_environment():
    """Test vessel environment simulation"""
    env = EnhancedVesselEnvironment()
    initial_state = env.reset()
    
    # Test step with random action
    action = np.array([0.5, 0.6, 0.3, 0.1])  # Example action
    next_state, reward, done, info = env.step(action)
    
    assert next_state.shape == initial_state.shape
    assert isinstance(reward, (int, float))
    assert isinstance(done, bool)
    assert isinstance(info, dict)

def test_model_training_pipeline(sample_vessel_data):
    """Test complete model training pipeline"""
    predictor, lstm_history, rl_history = train_models(
        sample_vessel_data, use_gpu=False
    )
    
    assert lstm_history is not None
    assert rl_history is not None
    assert 'loss' in lstm_history.history
    
    # Test prediction with trained model
    vessel_data = sample_vessel_data.iloc[:10]
    processed_data = predictor.prepare_enhanced_data(vessel_data)
    
    # Make prediction
    sequence = processed_data['inputs']['sequences'][0]
    cat_data = processed_data['inputs']['categorical'][0]
    binary_data = processed_data['inputs']['binary'][0]
    temporal_data = processed_data['inputs']['temporal'][0]
    
    position, motion, status = predictor.predict_trajectory(
        sequence, cat_data, binary_data, temporal_data
    )
    
    assert position is not None
    assert motion is not None
    assert status is not None

if __name__ == '__main__':
    pytest.main([__file__])