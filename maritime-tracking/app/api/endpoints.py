from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import pandas as pd
from datetime import datetime
import uvicorn

from ..ml.predictor import VesselPredictor
from ..core.config import settings

app = FastAPI(
    title="Maritime Vessel Tracking API",
    description="API for vessel tracking and prediction using machine learning",
    version="1.0.0",
    docs_url="/docs",   # Swagger UI endpoint
    redoc_url="/redoc"  # ReDoc endpoint
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
predictor = None
vessel_data = None
weather_data = None

@app.on_event("startup")
async def startup_event():
    global predictor, vessel_data, weather_data
    try:
        print("Starting server initialization...")
        
        # Initialize predictor
        predictor = VesselPredictor(use_gpu=settings.USE_GPU)
        print("Predictor initialized")
        
        # Load vessel data
        print("Loading vessel data...")
        vessel_data = pd.read_csv(
            f"{settings.DATA_PATH}/AIS_2020_01_01.csv",
            parse_dates=['BaseDateTime']
        )
        print(f"Loaded {len(vessel_data)} vessel records")
        
        # Try to load weather data
        try:
            weather_data = pd.read_csv(
                f"{settings.DATA_PATH}/weather_data.csv",
                parse_dates=['timestamp']
            )
            print("Weather data loaded")
        except:
            weather_data = None
            print("No weather data available")
            
        # Train predictor
        print("Training predictor...")
        training_result = predictor.train(vessel_data, weather_data)
        if training_result["status"] != "success":
            raise Exception(training_result["message"])
            
        print("Server initialization complete")
            
    except Exception as e:
        print(f"Startup error: {e}")
        raise e

@app.get("/")
async def root():
    """Root endpoint returning API status"""
    return {
        "status": "operational",
        "version": "1.0.0",
        "models_loaded": predictor is not None,
        "data_loaded": vessel_data is not None
    }

@app.get("/api/v1/status")
async def get_status():
    """Get detailed system status"""
    if predictor is None:
        raise HTTPException(status_code=500, detail="System not initialized")
        
    return {
        "status": "operational",
        "predictor_status": predictor.get_status(),
        "data_loaded": vessel_data is not None,
        "weather_data_available": weather_data is not None,
        "total_vessels": len(vessel_data['MMSI'].unique()) if vessel_data is not None else 0,
        "data_timespan": {
            "start": vessel_data['BaseDateTime'].min().isoformat() if vessel_data is not None else None,
            "end": vessel_data['BaseDateTime'].max().isoformat() if vessel_data is not None else None
        },
        "last_update": datetime.now().isoformat()
    }

@app.get("/api/v1/vessels")
async def get_vessels():
    """Get list of all vessels"""
    if vessel_data is None:
        raise HTTPException(status_code=500, detail="Data not loaded")
        
    vessels = vessel_data['MMSI'].unique().tolist()
    return {"vessels": vessels}

@app.get("/api/v1/predict/{mmsi}")
async def predict_vessel(
    mmsi: str,
    hours_ahead: Optional[int] = 24
):
    """Get predictions for specific vessel"""
    try:
        # Validate MMSI
        if vessel_data is None:
            raise HTTPException(status_code=500, detail="Data not loaded")
            
        # Get vessel data
        vessel_subset = vessel_data[vessel_data['MMSI'] == mmsi].copy()
        if len(vessel_subset) == 0:
            raise HTTPException(status_code=404, detail="Vessel not found")
            
        # Get corresponding weather data
        if weather_data is not None:
            weather_subset = weather_data[
                (weather_data['timestamp'] >= vessel_subset['BaseDateTime'].min()) &
                (weather_data['timestamp'] <= vessel_subset['BaseDateTime'].max())
            ]
        else:
            weather_subset = None
            
        # Make prediction
        result = predictor.predict(vessel_subset, weather_subset)
        
        if result["status"] != "success":
            raise HTTPException(status_code=500, detail=result["message"])
            
        return result["predictions"]
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "app.api.endpoints:app",
        host="0.0.0.0",
        port=8001,  # Updated port
        reload=True,
        log_level="info"
    )