import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
import json
from datetime import datetime

class CustomJsonFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    def format(self, record):
        log_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if hasattr(record, 'vessel_mmsi'):
            log_record['vessel_mmsi'] = record.vessel_mmsi
            
        if hasattr(record, 'model_type'):
            log_record['model_type'] = record.model_type
            
        if record.exc_info:
            log_record['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_record)

def setup_logging(log_level=logging.INFO):
    """Setup application logging"""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(CustomJsonFormatter())
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = RotatingFileHandler(
        "logs/maritime_tracking.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(CustomJsonFormatter())
    logger.addHandler(file_handler)
    
    # Model prediction logger
    prediction_logger = logging.getLogger('predictions')
    prediction_handler = RotatingFileHandler(
        "logs/predictions.log",
        maxBytes=10*1024*1024,
        backupCount=5
    )
    prediction_handler.setFormatter(CustomJsonFormatter())
    prediction_logger.addHandler(prediction_handler)
    
    # Performance metrics logger
    metrics_logger = logging.getLogger('metrics')
    metrics_handler = RotatingFileHandler(
        "logs/metrics.log",
        maxBytes=10*1024*1024,
        backupCount=5
    )
    metrics_handler.setFormatter(CustomJsonFormatter())
    metrics_logger.addHandler(metrics_handler)
    
    return logger

def log_prediction(mmsi: str, prediction_data: dict):
    """Log prediction results"""
    logger = logging.getLogger('predictions')
    logger.info(
        "Prediction made",
        extra={
            'vessel_mmsi': mmsi,
            'prediction_data': prediction_data
        }
    )

def log_model_metrics(model_type: str, metrics: dict):
    """Log model performance metrics"""
    logger = logging.getLogger('metrics')
    logger.info(
        "Model metrics calculated",
        extra={
            'model_type': model_type,
            'metrics': metrics
        }
    )

def log_error(error: Exception, context: dict = None):
    """Log error with context"""
    logger = logging.getLogger()
    logger.error(
        str(error),
        extra={'context': context},
        exc_info=True
    )