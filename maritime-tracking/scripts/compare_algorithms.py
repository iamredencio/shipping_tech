import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json

class AlgorithmComparison:
    def __init__(self, base_url="http://localhost:8001"):
        self.base_url = base_url
        self.colors = {
            'Simple': '#1f77b4',
            'Kalman': '#ff7f0e',
            'LSTM': '#2ca02c',
            'RL': '#d62728'
        }
        
    def fetch_data(self, mmsi):
        """Fetch data for a specific vessel"""
        try:
            # Get basic data
            response = requests.get(f"{self.base_url}/vessel/{mmsi}")
            basic_data = response.json()
            
            # Get AI predictions
            response = requests.get(f"{self.base_url}/vessel/{mmsi}/predict")
            ai_data = response.json()
            
            return basic_data, ai_data
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None, None
            
    def calculate_metrics(self, true_positions, predicted_positions):
        """Calculate error metrics"""
        mse = mean_squared_error(true_positions, predicted_positions)
        mae = mean_absolute_error(true_positions, predicted_positions)
        rmse = np.sqrt(mse)
        return {'MSE': mse, 'MAE': mae, 'RMSE': rmse}
        
    def plot_trajectories(self, vessel_data, predictions, title="Trajectory Comparison"):
        """Plot actual vs predicted trajectories"""
        plt.figure(figsize=(12, 8))
        
        # Plot actual trajectory
        actual = vessel_data['tracking_results']['current_position']
        plt.plot(actual[1], actual[0], 'ko-', label='Actual Position', markersize=10)
        
        # Plot predictions from different algorithms
        colors = self.colors
        
        # Simple prediction
        simple_pred = vessel_data['tracking_results']['predicted_position']
        plt.plot(simple_pred[1], simple_pred[0], 'o', color=colors['Simple'], 
                label='Simple Prediction', markersize=8)
        
        # LSTM prediction
        lstm_pred = predictions['lstm_prediction']['next_position']
        plt.plot(lstm_pred['lon'], lstm_pred['lat'], 'o', color=colors['LSTM'],
                label='LSTM Prediction', markersize=8)
        
        plt.title(title)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.legend()
        plt.grid(True)
        
        return plt.gcf()
        
    def plot_error_comparison(self, metrics):
        """Plot error metrics comparison"""
        plt.figure(figsize=(10, 6))
        
        algorithms = list(metrics.keys())
        error_types = ['MSE', 'MAE', 'RMSE']
        
        x = np.arange(len(algorithms))
        width = 0.25
        
        for i, error_type in enumerate(error_types):
            values = [metrics[alg][error_type] for alg in algorithms]
            plt.bar(x + i*width, values, width, label=error_type)
        
        plt.xlabel('Algorithm')
        plt.ylabel('Error Value')
        plt.title('Error Metrics Comparison')
        plt.xticks(x + width, algorithms)
        plt.legend()
        
        return plt.gcf()
        
    def plot_recommendation_analysis(self, predictions):
        """Plot RL recommendations"""
        plt.figure(figsize=(8, 6))
        
        recommendations = predictions['rl_recommendation']
        actions = list(recommendations.keys())
        values = list(recommendations.values())
        
        plt.bar(actions, values, color='skyblue')
        plt.title('RL Action Recommendations')
        plt.xlabel('Action')
        plt.ylabel('Probability')
        plt.xticks(rotation=45)
        
        return plt.gcf()
    
    def run_comparison(self, mmsi_list):
        """Run complete comparison for multiple vessels"""
        results = {}
        
        for mmsi in mmsi_list:
            print(f"\nAnalyzing vessel {mmsi}")
            basic_data, ai_predictions = self.fetch_data(mmsi)
            
            if basic_data is None or ai_predictions is None:
                continue
                
            # Create plots
            trajectory_fig = self.plot_trajectories(basic_data, ai_predictions,
                                                  f"Vessel {mmsi} Trajectory")
            trajectory_fig.savefig(f'trajectory_{mmsi}.png')
            
            recommendation_fig = self.plot_recommendation_analysis(ai_predictions)
            recommendation_fig.savefig(f'recommendations_{mmsi}.png')
            
            # Calculate metrics
            metrics = {
                'Simple': self.calculate_metrics(
                    np.array(basic_data['tracking_results']['current_position']),
                    np.array(basic_data['tracking_results']['predicted_position'])
                ),
                'LSTM': self.calculate_metrics(
                    np.array(basic_data['tracking_results']['current_position']),
                    np.array([ai_predictions['lstm_prediction']['next_position']['lat'],
                             ai_predictions['lstm_prediction']['next_position']['lon']])
                )
            }
            
            error_fig = self.plot_error_comparison(metrics)
            error_fig.savefig(f'errors_{mmsi}.png')
            
            results[mmsi] = {
                'metrics': metrics,
                'predictions': ai_predictions
            }
            
            plt.close('all')
            
        return results

if __name__ == "__main__":
    # Example usage
    comparison = AlgorithmComparison()
    
    # Get list of vessels
    response = requests.get("http://localhost:8001/vessels")
    vessels = response.json()['vessels'][:5]  # Analyze first 5 vessels
    
    print("Starting algorithm comparison...")
    results = comparison.run_comparison(vessels)
    
    # Save results
    with open('comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nComparison complete. Results saved to:")
    print("- comparison_results.json")
    print("- trajectory_*.png")
    print("- recommendations_*.png")
    print("- errors_*.png")