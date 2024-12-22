import requests
import matplotlib.pyplot as plt
import numpy as np
import sys
import traceback

def plot_vessel_track(mmsi):
    BASE_URL = "http://localhost:8001"
    
    try:
        # Check if server is running first
        try:
            response = requests.get(f"{BASE_URL}/")
            print(f"Server status: {response.status_code}")
        except requests.exceptions.ConnectionError:
            print("Error: Could not connect to server. Is it running on localhost:8000?")
            return
            
        # Get vessel data
        print(f"\nRequesting data for vessel {mmsi}")
        response = requests.get(f"{BASE_URL}/vessel/{mmsi}")
        
        # Debug information
        print(f"Response Status Code: {response.status_code}")
        print(f"Response Headers: {response.headers}")
        print(f"Raw Response: {response.text[:500]}...")  # First 500 chars
        
        if response.status_code != 200:
            print(f"Error: Server returned status code {response.status_code}")
            return
            
        # Parse JSON response
        data = response.json()
        print("\nReceived data structure:", data.keys())
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Extract tracking results
        if 'tracking_results' in data:
            results = data['tracking_results']
            
            # Plot current position
            current_pos = results['current_position']
            plt.plot(current_pos[1], current_pos[0], 'bo', 
                    markersize=10, label='Current Position')
            
            # Plot predicted position
            pred_pos = results['predicted_position']
            plt.plot(pred_pos[1], pred_pos[0], 'ro', 
                    markersize=10, label='Predicted Position')
            
            # Add vessel info
            vessel_info = data.get('vessel_info', {})
            vessel_name = vessel_info.get('VesselName', 'Unknown')
            
            plt.title(f"Vessel Track - MMSI: {mmsi}\nName: {vessel_name}")
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.legend()
            plt.grid(True)
            
            # Add speed and course information
            plt.text(0.02, 0.98, 
                    f"Speed: {results.get('current_speed', 'N/A')} knots\n"
                    f"Course: {results.get('current_course', 'N/A')}°",
                    transform=plt.gca().transAxes,
                    verticalalignment='top')
            
            plt.show()
            
            # Print additional information
            print("\nVessel Information:")
            for key, value in vessel_info.items():
                print(f"{key}: {value}")
            
            print("\nTracking Results:")
            for key, value in results.items():
                print(f"{key}: {value}")
        else:
            print("Error: No tracking results in response")
            print("Full response data:", data)
            
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print(f"Raw response: {response.text}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    print(f"Python version: {sys.version}")
    print(f"Starting visualization...\n")
    
    # Try a few different vessels
    test_mmsis = ['368063930', '538008468', '368120510']
    
    for mmsi in test_mmsis:
        print(f"\nTrying vessel MMSI: {mmsi}")
        plot_vessel_track(mmsi)
        input("Press Enter to continue to next vessel...")