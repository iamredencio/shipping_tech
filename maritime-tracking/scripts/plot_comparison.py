import requests
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class ResultsVisualizer:
    def __init__(self, base_url="http://localhost:8001"):
        self.base_url = base_url
        
    def fetch_predictions(self, mmsi):
        """Fetch predictions for a vessel"""
        response = requests.get(f"{self.base_url}/api/v1/predict/{mmsi}")
        return response.json()
        
    def plot_trajectory(self, mmsi):
        """Plot actual vs predicted trajectory"""
        data = self.fetch_predictions(mmsi)
        
        plt.figure(figsize=(12, 8))
        
        # Plot actual positions
        actual_positions = np.array(data['vessel_info']['track_history'])
        plt.plot(actual_positions[:, 1], actual_positions[:, 0], 
                'b-', label='Actual Track', linewidth=2)
        
        # Plot predicted positions
        predicted_pos = np.array(data['predictions']['position'])
        plt.plot(predicted_pos[:, 1], predicted_pos[:, 0], 
                'r--', label='Predicted Track', linewidth=2)
        
        plt.title(f"Vessel Trajectory - MMSI: {mmsi}")
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'trajectory_{mmsi}.png')
        plt.close()
        
    def plot_speed_comparison(self, mmsi):
        """Plot actual vs predicted speed"""
        data = self.fetch_predictions(mmsi)
        
        plt.figure(figsize=(12, 6))
        
        actual_speed = [x['SOG'] for x in data['vessel_info']['track_history']]
        predicted_speed = data['predictions']['motion'][:, 0]
        
        plt.plot(actual_speed, label='Actual Speed')
        plt.plot(predicted_speed, label='Predicted Speed', linestyle='--')
        
        plt.title(f"Speed Comparison - MMSI: {mmsi}")
        plt.xlabel('Time Steps')
        plt.ylabel('Speed (knots)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'speed_{mmsi}.png')
        plt.close()
        
    def plot_course_comparison(self, mmsi):
        """Plot actual vs predicted course"""
        data = self.fetch_predictions(mmsi)
        
        plt.figure(figsize=(12, 6))
        
        actual_course = [x['COG'] for x in data['vessel_info']['track_history']]
        predicted_course = data['predictions']['motion'][:, 1]
        
        plt.plot(actual_course, label='Actual Course')
        plt.plot(predicted_course, label='Predicted Course', linestyle='--')
        
        plt.title(f"Course Comparison - MMSI: {mmsi}")
        plt.xlabel('Time Steps')
        plt.ylabel('Course (degrees)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'course_{mmsi}.png')
        plt.close()
        
    def plot_error_analysis(self, mmsi):
        """Plot prediction errors"""
        data = self.fetch_predictions(mmsi)
        
        actual_pos = np.array(data['vessel_info']['track_history'])[-len(data['predictions']['position']):]
        pred_pos = np.array(data['predictions']['position'])
        
        # Calculate errors
        position_error = np.sqrt(np.sum((actual_pos - pred_pos) ** 2, axis=1))
        
        plt.figure(figsize=(12, 6))
        plt.plot(position_error)
        plt.title(f"Position Prediction Error - MMSI: {mmsi}")
        plt.xlabel('Time Steps')
        plt.ylabel('Error (nautical miles)')
        plt.grid(True)
        plt.savefig(f'error_{mmsi}.png')
        plt.close()
        
    def generate_report(self, mmsi_list):
        """Generate comprehensive analysis for multiple vessels"""
        for mmsi in mmsi_list:
            print(f"\nAnalyzing vessel {mmsi}")
            try:
                self.plot_trajectory(mmsi)
                self.plot_speed_comparison(mmsi)
                self.plot_course_comparison(mmsi)
                self.plot_error_analysis(mmsi)
            except Exception as e:
                print(f"Error analyzing vessel {mmsi}: {e}")

if __name__ == "__main__":
    # Example usage
    visualizer = ResultsVisualizer()
    
    # List of vessels to analyze
    test_vessels = [
        '538008468',  # Replace with actual MMSI numbers
        '368120510',
        '368063930'
    ]
    
    visualizer.generate_report(test_vessels)
    print("\nAnalysis complete. Check the output files for plots.")