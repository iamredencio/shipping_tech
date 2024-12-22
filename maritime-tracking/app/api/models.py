from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime

class VesselBase(BaseModel):
    """Base Vessel Information"""
    mmsi: str
    vessel_name: Optional[str] = None
    imo: Optional[str] = None
    call_sign: Optional[str] = None
    vessel_type: Optional[int] = None
    length: Optional[float] = None
    width: Optional[float] = None
    draft: Optional[float] = None
    cargo: Optional[str] = None
    transceiver_class: Optional[str] = None

class Position(BaseModel):
    """Vessel Position"""
    lat: float = Field(..., description="Latitude")
    lon: float = Field(..., description="Longitude")
    timestamp: datetime

class Motion(BaseModel):
    """Vessel Motion"""
    sog: float = Field(..., description="Speed Over Ground")
    cog: float = Field(..., description="Course Over Ground")
    heading: Optional[float] = Field(None, description="True Heading")

class PredictionRequest(BaseModel):
    """Prediction Request"""
    mmsi: str
    hours_ahead: int = Field(24, description="Hours to predict ahead")
    include_weather: bool = Field(False, description="Include weather in prediction")

class TrajectoryPrediction(BaseModel):
    """Trajectory Prediction"""
    position: Position
    motion: Motion
    confidence: float = Field(..., description="Prediction confidence score")
    timestamp: datetime

class VesselState(BaseModel):
    """Current Vessel State"""
    vessel: VesselBase
    position: Position
    motion: Motion
    status: str
    predictions: List[TrajectoryPrediction]

class ModelMetrics(BaseModel):
    """Model Performance Metrics"""
    mae_position: float
    mae_motion: float
    rmse_position: float
    rmse_motion: float
    prediction_accuracy: float

class SystemStatus(BaseModel):
    """System Status Information"""
    status: str
    models_loaded: bool
    gpu_available: bool
    total_vessels: int
    last_update: datetime
    model_metrics: Optional[ModelMetrics]
    version: str

class WeatherData(BaseModel):
    """Weather Information"""
    timestamp: datetime
    wind_speed: float
    wind_direction: float
    wave_height: Optional[float]
    wave_direction: Optional[float]
    visibility: Optional[float]
    temperature: Optional[float]

class TrafficData(BaseModel):
    """Traffic Information"""
    timestamp: datetime
    location: Position
    vessel_count: int
    average_speed: float
    traffic_density: float

class PredictionResponse(BaseModel):
    """Complete Prediction Response"""
    vessel_id: str
    timestamp: datetime
    current_state: VesselState
    predictions: List[TrajectoryPrediction]
    confidence_intervals: Dict[str, float]
    weather_conditions: Optional[WeatherData]
    traffic_conditions: Optional[TrafficData]