import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate, TimeDistributed
from tensorflow.keras.optimizers import Adam
import pandas as pd
import os
import logging

class DeepVesselPredictor:
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu
        if use_gpu:
            if tf.config.list_physical_devices('GPU'):
                print("Using GPU for computations")
                for gpu in tf.config.list_physical_devices('GPU'):
                    tf.config.experimental.set_memory_growth(gpu, True)
            else:
                print("No GPU found, falling back to CPU")
                self.use_gpu = False
        else:
            print("Using CPU for computations")
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        # Base features from AIS data
        self.continuous_features = [
            'LAT', 'LON',           # Position
            'SOG', 'COG', 'Heading', # Motion
            'Length', 'Width', 'Draft' # Vessel dimensions
        ]
        
        self.categorical_features = [
            'VesselType',
            'Status',
            'Cargo',
            'TransceiverClass'
        ]
        
        self.binary_features = [
            'is_night',
            'is_underway',
            'is_anchored',
            'is_moored',
            'is_restricted',
            'is_passenger',
            'is_cargo',
            'is_tanker',
            'is_class_a',
            'is_class_b'
        ]
        
        self.temporal_features = [
            'hour',
            'day_of_week',
            'month'
        ]
        
        self.sequence_length = 10
        self.lstm_model = self._build_enhanced_lstm()
        self.rl_model = self._build_enhanced_rl_model()
        self.is_trained = False

    def _build_enhanced_lstm(self):
        """Build enhanced LSTM model with all AIS features"""
        try:
            # Sequential features input (continuous data over time)
            seq_input = Input(shape=(self.sequence_length, len(self.continuous_features)))
            
            # Categorical features input (static per sequence)
            cat_input = Input(shape=(len(self.categorical_features),))
            
            # Binary features input
            binary_input = Input(shape=(len(self.binary_features),))
            
            # Temporal features input
            temporal_input = Input(shape=(len(self.temporal_features),))
            
            # Process sequential data
            x = LSTM(128, return_sequences=True)(seq_input)
            x = Dropout(0.2)(x)
            x = LSTM(64, return_sequences=True)(x)
            x = Dropout(0.2)(x)
            x = LSTM(32)(x)
            
            # Process categorical data
            cat_features = Dense(32, activation='relu')(cat_input)
            cat_features = Dense(16, activation='relu')(cat_features)
            
            # Process binary data
            binary_features = Dense(8, activation='relu')(binary_input)
            
            # Process temporal data
            temporal_features = Dense(8, activation='relu')(temporal_input)
            
            # Combine all features
            combined = Concatenate()([
                x,
                cat_features,
                binary_features,
                temporal_features
            ])
            
            # Dense layers for combined processing
            combined = Dense(64, activation='relu')(combined)
            combined = Dropout(0.2)(combined)
            combined = Dense(32, activation='relu')(combined)
            
            # Multiple prediction heads
            position_output = Dense(2, name='position')(combined)  # LAT, LON
            motion_output = Dense(3, name='motion')(combined)      # SOG, COG, Heading
            status_output = Dense(1, activation='sigmoid', name='status')(combined)  # Vessel status prediction
            
            model = Model(
                inputs=[seq_input, cat_input, binary_input, temporal_input],
                outputs=[position_output, motion_output, status_output]
            )
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss={
                    'position': 'mse',
                    'motion': 'mse',
                    'status': 'binary_crossentropy'
                },
                loss_weights={
                    'position': 1.0,
                    'motion': 0.5,
                    'status': 0.3
                },
                metrics={
                    'position': ['mae'],
                    'motion': ['mae'],
                    'status': ['accuracy']
                }
            )
            return model
        except Exception as e:
            logging.error(f"Error building LSTM model: {e}")
            raise

    def _build_enhanced_rl_model(self):
        """Build enhanced RL model with all state information"""
        try:
            # Enhanced state space including vessel characteristics
            state_dim = 12  # [lat, lon, sog, cog, heading, length, width, draft, 
                           #  target_lat, target_lon, distance, traffic_density]
            
            model = Sequential([
                Input(shape=(state_dim,)),
                Dense(128, activation='relu'),
                Dense(64, activation='relu'),
                Dense(32, activation='relu'),
                Dense(4, activation='softmax')  # [maintain, adjust_course, adjust_speed, emergency_maneuver]
            ])
            
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse'
            )
            return model
        except Exception as e:
            logging.error(f"Error building RL model: {e}")
            raise

    def prepare_enhanced_data(self, data):
        """Prepare all AIS features for model training"""
        try:
            sequences = []
            cat_features = []
            binary_features = []
            temporal_features = []
            position_targets = []
            motion_targets = []
            status_targets = []
            
            for i in range(len(data) - self.sequence_length):
                # Sequence data (continuous features)
                seq = data[self.continuous_features].iloc[i:i+self.sequence_length].values
                sequences.append(seq)
                
                # Categorical features (from last timestep)
                cat = data[self.categorical_features].iloc[i+self.sequence_length-1].values
                cat_features.append(cat)
                
                # Binary features
                binary = data[self.binary_features].iloc[i+self.sequence_length-1].values
                binary_features.append(binary)
                
                # Temporal features
                temporal = data[self.temporal_features].iloc[i+self.sequence_length-1].values
                temporal_features.append(temporal)
                
                # Targets
                pos_target = data[['LAT', 'LON']].iloc[i+self.sequence_length].values
                mot_target = data[['SOG', 'COG', 'Heading']].iloc[i+self.sequence_length].values
                status_target = data['is_underway'].iloc[i+self.sequence_length]
                
                position_targets.append(pos_target)
                motion_targets.append(mot_target)
                status_targets.append(status_target)
            
            return {
                'inputs': {
                    'sequences': np.array(sequences),
                    'categorical': np.array(cat_features),
                    'binary': np.array(binary_features),
                    'temporal': np.array(temporal_features)
                },
                'targets': {
                    'position': np.array(position_targets),
                    'motion': np.array(motion_targets),
                    'status': np.array(status_targets)
                }
            }
        except Exception as e:
            logging.error(f"Error preparing data: {e}")
            raise

    def train(self, data, epochs=50, batch_size=32):
        """Train LSTM with all features"""
        try:
            processed_data = self.prepare_enhanced_data(data)
            
            with tf.device('/GPU:0' if self.use_gpu else '/CPU:0'):
                history = self.lstm_model.fit(
                    [
                        processed_data['inputs']['sequences'],
                        processed_data['inputs']['categorical'],
                        processed_data['inputs']['binary'],
                        processed_data['inputs']['temporal']
                    ],
                    {
                        'position': processed_data['targets']['position'],
                        'motion': processed_data['targets']['motion'],
                        'status': processed_data['targets']['status']
                    },
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.2,
                    verbose=1
                )
            self.is_trained = True
            return history
        except Exception as e:
            logging.error(f"Error training model: {e}")
            raise

    def train_rl(self, states, actions, rewards, epochs=50):
        """Train RL model"""
        try:
            with tf.device('/GPU:0' if self.use_gpu else '/CPU:0'):
                history = self.rl_model.fit(
                    states,
                    actions,
                    sample_weight=rewards,
                    epochs=epochs,
                    verbose=1
                )
            return history
        except Exception as e:
            logging.error(f"Error training RL model: {e}")
            raise
    
    def get_action(self, state):
        """Get optimal action from RL model"""
        try:
            with tf.device('/GPU:0' if self.use_gpu else '/CPU:0'):
                return self.rl_model.predict(state.reshape(1, -1), verbose=0)[0]
        except Exception as e:
            logging.error(f"Error getting action: {e}")
            raise

    def predict_trajectory(self, sequence, categorical_data, binary_data, temporal_data):
        """Predict vessel trajectory using all features"""
        try:
            with tf.device('/GPU:0' if self.use_gpu else '/CPU:0'):
                position, motion, status = self.lstm_model.predict(
                    [
                        sequence.reshape(1, self.sequence_length, -1),
                        categorical_data.reshape(1, -1),
                        binary_data.reshape(1, -1),
                        temporal_data.reshape(1, -1)
                    ],
                    verbose=0
                )
                return position[0], motion[0], status[0]
        except Exception as e:
            logging.error(f"Error predicting trajectory: {e}")
            raise

    def get_model_status(self):
        """Get model status information"""
        return {
            "is_trained": self.is_trained,
            "gpu_available": self.use_gpu,
            "sequence_length": self.sequence_length,
            "feature_counts": {
                "continuous": len(self.continuous_features),
                "categorical": len(self.categorical_features),
                "binary": len(self.binary_features),
                "temporal": len(self.temporal_features)
            }
        }

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
            next_state, reward, done, _ = env