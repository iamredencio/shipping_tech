import requests
import pandas as pd
import json
import sys

def test_api():
    BASE_URL = "http://localhost:8001"
    
    try:
        # Test root endpoint
        response = requests.get(f"{BASE_URL}/")
        print("Root endpoint:", response.json())
        
        # Get list of vessels
        response = requests.get(f"{BASE_URL}/vessels")
        vessels = response.json()["vessels"]
        print(f"\nFound {len(vessels)} vessels")
        print("First few vessels:", vessels[:5])
        
        # Get data for first vessel
        if vessels:
            mmsi = vessels[0]
            print(f"\nGetting data for vessel {mmsi}")
            response = requests.get(f"{BASE_URL}/vessel/{mmsi}")
            
            # Debug info
            print("\nResponse status code:", response.status_code)
            print("Response headers:", response.headers)
            print("Raw response text:", response.text)
            
            if response.status_code == 200:
                vessel_data = response.json()
                print("\nVessel Data:")
                print(json.dumps(vessel_data, indent=2))
            else:
                print(f"Error: Received status code {response.status_code}")
                
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server. Is it running?")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        print(f"Error type: {type(e)}")

if __name__ == "__main__":
    print("Python version:", sys.version)
    print("Starting API test...\n")
    test_api()