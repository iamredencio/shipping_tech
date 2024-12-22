import numpy as np
import tensorflow as tf
from scipy.linalg import solve

class EnhancedKalmanFilter:
    """
    Enhanced Kalman Filter implementation for maritime vessel tracking
    with improved numerical stability and state estimation
    """
    def __init__(self):
        # State vector components: [lat, lon, sog, cog, heading, length, width, draft]
        self.n_states = 8
        self.state = None
        
        # Covariance matrix with carefully tuned uncertainties for each state variable
        self.P = np.diag([
            0.001,  # lat uncertainty (degrees) - precise GPS measurement
            0.001,  # lon uncertainty (degrees) - precise GPS measurement
            1.0,    # speed uncertainty (knots) - moderate variation expected
            2.0,    # course uncertainty (degrees) - accounts for turning
            2.0,    # heading uncertainty (degrees) - accounts for turning
            0.1,    # length uncertainty (meters) - static vessel property
            0.1,    # width uncertainty (meters) - static vessel property
            0.1     # draft uncertainty (meters) - can vary with cargo
        ])
        
        # Process noise matrix - models system dynamics uncertainty
        self.Q = np.diag([
            1e-6,   # lat - very small process noise
            1e-6,   # lon - very small process noise
            0.1,    # speed - moderate variation possible
            0.2,    # course - higher uncertainty in turns
            0.2,    # heading - higher uncertainty in turns
            1e-8,   # length - almost constant
            1e-8,   # width - almost constant
            0.01    # draft - slow changes possible
        ])
        
        # Measurement noise
        self.R = np.diag([
            1e-5,   # lat
            1e-5,   # lon
            0.5,    # speed
            1.0,    # course
            1.0,    # heading
            1e-4,   # length
            1e-4,   # width
            0.05    # draft
        ])
        
        self.dt = 1.0  # Time step in seconds

    def predict(self):
        """Predict next state"""
        if self.state is None:
            return None

        # State transition matrix
        F = np.eye(self.n_states)
        
        # Update position based on speed and course (if speed is reasonable)
        if 0 <= self.state[2] <= 30:  # Speed between 0 and 30 knots
            speed_ms = self.state[2] * 0.514444  # Convert knots to m/s
            course_rad = np.radians(self.state[3])
            
            # Convert speed to degree changes (approximate)
            lat_change = speed_ms * np.cos(course_rad) * self.dt * 8.99e-6  # degrees latitude per second
            lon_change = speed_ms * np.sin(course_rad) * self.dt * 8.99e-6 / np.cos(np.radians(self.state[0]))
            
            F[0, 2] = lat_change / speed_ms if speed_ms > 0 else 0
            F[1, 2] = lon_change / speed_ms if speed_ms > 0 else 0

        # Predict next state
        self.state = F @ self.state
        
        # Update covariance with stabilized computation
        self.P = F @ self.P @ F.T + self.Q
        
        # Ensure symmetry and positive definiteness
        self.P = (self.P + self.P.T) / 2
        min_cov = 1e-10
        self.P = self.P + np.eye(self.n_states) * min_cov
        
        return self.state

    def update(self, measurement):
        """Update state with new measurement using stable computations"""
        if self.state is None:
            self.state = measurement
            return self.state

        try:
            # Measurement matrix
            H = np.eye(self.n_states)
            
            # Innovation covariance
            S = H @ self.P @ H.T + self.R
            
            # Ensure symmetry
            S = (S + S.T) / 2
            
            # Add small value to diagonal for numerical stability
            S = S + np.eye(self.n_states) * 1e-8
            
            # Compute Kalman gain using solve instead of inverse
            K = solve(S.T, (H @ self.P).T).T
            
            # Update state
            innovation = measurement - H @ self.state
            self.state = self.state + K @ innovation
            
            # Update covariance with Joseph form for better numerical stability
            I = np.eye(self.n_states)
            self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.R @ K.T
            
            # Ensure symmetry and positive definiteness
            self.P = (self.P + self.P.T) / 2
            min_cov = 1e-10
            self.P = self.P + np.eye(self.n_states) * min_cov
            
            return self.state
            
        except Exception as e:
            print(f"Error in Kalman update: {e}")
            # If update fails, return predicted state
            return self.state

class VesselTracker:
    """Combines all tracking algorithms"""
    def __init__(self):
        self.kf = EnhancedKalmanFilter()
        
    def process_vessel(self, vessel_data, data_loader):
        """Process vessel data with improved error handling"""
        try:
            # Convert data to state vectors
            states = []
            filtered_states = []
            
            for _, row in vessel_data.iterrows():
                try:
                    state = data_loader.get_state_vector(row)
                    states.append(state)
                    
                    # Process with Kalman Filter
                    filtered_state = self.kf.update(state)
                    if filtered_state is not None:
                        filtered_states.append(filtered_state)
                        
                except Exception as e:
                    print(f"Error processing row: {e}")
                    continue
            
            if not filtered_states:
                raise ValueError("No valid filtered states produced")
            
            # Simple prediction: extrapolate last two positions
            if len(filtered_states) >= 2:
                last_state = filtered_states[-1]
                prev_state = filtered_states[-2]
                velocity = last_state - prev_state
                predicted_state = last_state + velocity
            else:
                predicted_state = filtered_states[-1]
            
            return {
                'filtered_states': filtered_states,
                'predicted_state': predicted_state
            }
            
        except Exception as e:
            print(f"Error in process_vessel: {e}")
            raise