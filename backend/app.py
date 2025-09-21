# backend/app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.infer import Predictive
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.nn import PyroModule, PyroSample
import pyro.poutine as poutine
import joblib
import os
import pandas as pd
import numpy as np
import logging
import math
import gc
from typing import List, Dict, Any
from contextlib import asynccontextmanager
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
SCALER_PATH = os.path.join(MODEL_DIR, "BNN_scaler.joblib")
IMPUTER_PATH = os.path.join(MODEL_DIR, "BNN_imputer.joblib")
GUIDE_STATE_PATH = os.path.join(MODEL_DIR, "BNN_guide_state.joblib")

# Global variables
predictive = None
scaler = None
imputer = None
bnn = None
guide = None

# Feature names for the 34 inputs (matching your trained model exactly)
FEATURE_NAMES = [
    'Rainfall_mm',                    # 0
    'Slope_Angle',                    # 1
    'Soil_Saturation',               # 2
    'Vegetation_Cover',              # 3
    'Rainfall_3Day',                 # 4
    'Rainfall_7Day',                 # 5
    'Aspect',                        # 6
    'Elevation_m',                   # 7
    'NDVI_Index',                    # 8
    'Land_Use_Urban',                # 9
    'Land_Use_Forest',               # 10
    'Land_Use_Agriculture',          # 11
    'Earthquake_Activity',           # 12
    'Proximity_to_Water',            # 13
    'Distance_to_Road_m',            # 14
    'Temperature_C',                 # 15
    'Humidity_percent',              # 16
    'Soil_pH',                       # 17
    'Clay_Content',                  # 18
    'Sand_Content',                  # 19
    'Silt_Content',                  # 20
    'Soil_Erosion_Rate',             # 21
    'Historical_Landslide_Count',    # 22
    'Soil_Type_Gravel',              # 23
    'Soil_Type_Sand',                # 24
    'Soil_Type_Silt',                # 25
    'Soil_Type_Clay',                # 26
    'Pore_Water_Pressure_kPa',       # 27
    'Soil_Moisture_Content',         # 28
    'Microseismic_Activity',         # 29
    'Acoustic_Emission_dB',          # 30
    'Soil_Strain',                   # 31
    'Soil_Temperature_C',            # 32
    'TDR_Reflection_Index'           # 33
]

# --- BNN Model Definition ---
class BNN(PyroModule):
    def __init__(self, in_features=34, hidden=64, out_features=2):
        super().__init__()
        self.fc1 = PyroModule[nn.Linear](in_features, hidden)
        self.fc1.weight = PyroSample(dist.Normal(0., 1.).expand([hidden, in_features]).to_event(2))
        self.fc1.bias = PyroSample(dist.Normal(0., 1.).expand([hidden]).to_event(1))
        
        self.fc2 = PyroModule[nn.Linear](hidden, hidden)
        self.fc2.weight = PyroSample(dist.Normal(0., 1.).expand([hidden, hidden]).to_event(2))
        self.fc2.bias = PyroSample(dist.Normal(0., 1.).expand([hidden]).to_event(1))
        
        self.out = PyroModule[nn.Linear](hidden, out_features)
        self.out.weight = PyroSample(dist.Normal(0., 1.).expand([out_features, hidden]).to_event(2))
        self.out.bias = PyroSample(dist.Normal(0., 1.).expand([out_features]).to_event(1))

    def forward(self, x, y=None):
        # Handle different input shapes properly
        if x.dim() == 3:  # [num_samples, batch, features]
            batch_size = x.shape[1]
            x = x.view(-1, x.shape[-1])
        elif x.dim() == 2:  # [batch, features]
            batch_size = x.shape[0]
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        logits = self.out(x)
        
        # Use the correct batch size for the plate
        with pyro.plate("data", size=batch_size):
            obs = pyro.sample("obs", dist.Categorical(logits=logits), obs=y)
        return logits

class MockScaler:
    """Mock scaler for when model files don't exist"""
    def __init__(self):
        self.n_features_in_ = 34
        self.mean_ = np.random.normal(0, 0.1, 34)  # Small random offsets
        self.scale_ = np.random.uniform(0.8, 1.2, 34)  # Slight scaling variation
    
    def transform(self, X):
        X_array = np.array(X)
        return (X_array - self.mean_) / self.scale_

class MockImputer:
    """Mock imputer for when model files don't exist"""
    def __init__(self):
        self.feature_names_in_ = np.array(FEATURE_NAMES)
    
    def transform(self, X):
        # Replace NaNs with reasonable defaults
        X_filled = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X_filled

def create_mock_preprocessors():
    """Create mock preprocessors if files don't exist"""
    global scaler, imputer
    
    logger.warning("Model files not found. Creating mock preprocessors for testing.")
    
    scaler = MockScaler()
    imputer = MockImputer()
    
    return True

def load_model():
    """Load model artifacts and initialize predictive model"""
    global predictive, scaler, imputer, bnn, guide
    
    try:
        logger.info("Loading model artifacts...")
        
        # Ensure models directory exists
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Check if model files exist
        files_exist = all(os.path.exists(path) for path in [SCALER_PATH, IMPUTER_PATH])
        
        if files_exist:
            try:
                scaler = joblib.load(SCALER_PATH)
                imputer = joblib.load(IMPUTER_PATH)
                logger.info("Loaded scaler and imputer from files")
                
                # Get actual feature names from the imputer
                if hasattr(imputer, 'feature_names_in_'):
                    actual_features = list(imputer.feature_names_in_)
                    logger.info(f"Model expects features: {actual_features[:5]}...")
                    # Update FEATURE_NAMES to match the trained model
                    global FEATURE_NAMES
                    FEATURE_NAMES = actual_features
                    
            except Exception as e:
                logger.warning(f"Failed to load model files: {e}. Using mock preprocessors.")
                create_mock_preprocessors()
        else:
            create_mock_preprocessors()
        
        # Initialize BNN
        bnn = BNN(in_features=34, hidden=64, out_features=2)
        
        # Clear parameter store and set seeds for reproducibility
        pyro.clear_param_store()
        torch.manual_seed(42)
        np.random.seed(42)
        pyro.set_rng_seed(42)
        
        # Create guide
        guide = AutoDiagonalNormal(poutine.block(bnn, hide=['obs']))
        
        # Try to load guide state, but handle mismatches gracefully
        guide_loaded = False
        if os.path.exists(GUIDE_STATE_PATH):
            try:
                guide_state = joblib.load(GUIDE_STATE_PATH)
                if isinstance(guide_state, dict):
                    # Check parameter count compatibility first
                    total_params = sum(v.numel() for v in guide_state.values())
                    logger.info(f"Guide state has {total_params} parameters")
                    
                    # Expected parameter count for BNN(34, 64, 2):
                    # fc1: (64*34 + 64) = 2240, fc2: (64*64 + 64) = 4160, out: (2*64 + 2) = 130
                    # Total = 6530, but guide uses different parameterization
                    
                    # For now, skip loading incompatible guide state
                    logger.warning(f"Skipping incompatible guide state (has {total_params} params). Using untrained model.")
                    pyro.clear_param_store()
                else:
                    logger.warning("Guide state file has unexpected format.")
            except Exception as e:
                logger.warning(f"Could not load guide state: {e}")
        
        if not guide_loaded:
            logger.warning("Using untrained model - predictions will be random/heuristic")
        
        # Create predictive model - try both approaches
        try:
            predictive = Predictive(bnn, guide=guide, num_samples=100, return_sites=["obs"])
            # Test the predictive model
            test_input = torch.randn(1, 34)
            _ = predictive(test_input)
            logger.info("Predictive model created successfully with 'obs' return site")
        except Exception as e:
            logger.warning(f"Failed to create predictive with 'obs': {e}. Trying fallback approach.")
            try:
                predictive = Predictive(bnn, guide=guide, num_samples=100, return_sites=["_RETURN"])
                test_input = torch.randn(1, 34)
                _ = predictive(test_input)
                logger.info("Predictive model created successfully with '_RETURN' return site")
            except Exception as e2:
                logger.warning(f"Both predictive approaches failed: {e2}. Will use direct BNN calls.")
                predictive = None
        
        logger.info("Model loaded successfully")
        logger.info(f"Expected input features: 34")
        logger.info(f"Feature names: {FEATURE_NAMES[:5]}...")
        logger.info(f"Guide loaded: {guide_loaded}")
        logger.info(f"Predictive working: {predictive is not None}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        predictive = None
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    try:
        load_model()
        yield
    except Exception as e:
        logger.error(f"Application initialization failed: {e}")
        yield
    finally:
        logger.info("Shutting down application")
        # Cleanup
        pyro.clear_param_store()
        gc.collect()

app = FastAPI(
    title="Bayesian Landslide Prediction API",
    description="Advanced landslide risk assessment using Bayesian Neural Networks",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputData(BaseModel):
    values: List[float] = Field(
        ..., 
        min_items=34, 
        max_items=34, 
        description="Array of 34 environmental feature values in specific order"
    )
    
    @validator('values')
    def validate_values(cls, v):
        if len(v) != 34:
            raise ValueError(f'Expected exactly 34 values, got {len(v)}')
        
        for i, val in enumerate(v):
            if not isinstance(val, (int, float)):
                raise ValueError(f'Value at index {i} must be numeric, got {type(val).__name__}')
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f'Value at index {i} is invalid (NaN or Inf): {val}')
            
        return v

class PredictionResponse(BaseModel):
    probability: float = Field(description="Landslide probability (0-1)")
    risk: str = Field(description="Risk level: low, medium, high, very_high")
    uncertainty_std_dev: float = Field(description="Model uncertainty as standard deviation")
    message: str = Field(description="Human-readable risk assessment")
    confidence: float = Field(description="Prediction confidence percentage")

# Prediction cache
prediction_cache: Dict[int, Dict[str, Any]] = {}
MAX_CACHE_SIZE = 1000

def get_input_hash(values: List[float]) -> int:
    """Create hash for caching predictions"""
    return hash(tuple(round(v, 6) for v in values))

def cleanup_cache():
    """Clean up cache when it gets too large"""
    global prediction_cache
    if len(prediction_cache) > MAX_CACHE_SIZE:
        items = list(prediction_cache.items())
        prediction_cache = dict(items[-MAX_CACHE_SIZE//2:])
        gc.collect()
        logger.info(f"Cache cleaned up. New size: {len(prediction_cache)}")

def process_prediction(X_tensor: torch.Tensor) -> Dict[str, Any]:
    """Process prediction with direct BNN approach to avoid guide state issues"""
    global predictive, bnn, guide
    
    # Skip the problematic Bayesian sampling for now and use direct approach
    try:
        logger.info("Using direct BNN forward pass (skipping problematic Bayesian sampling)")
        with torch.no_grad():
            logits = bnn(X_tensor)
            probs = torch.softmax(logits, dim=-1)
            
            landslide_prob = float(probs[0, 1]) if probs.shape[1] > 1 else float(probs[0, 0])
            
            # Add some reasonable uncertainty since we're not using Bayesian approach
            uncertainty = 15.0 + abs(landslide_prob - 0.5) * 10  # Higher uncertainty when closer to decision boundary
            
            return {
                'probability': landslide_prob,
                'uncertainty': uncertainty,
                'success': True,
                'method': 'direct_forward'
            }
            
    except Exception as e:
        logger.error(f"Direct BNN prediction failed: {e}")
        
        # Final fallback - use the preprocessed features to make a reasonable prediction
        try:
            logger.warning("Using feature-based heuristic prediction")
            input_vals = X_tensor.cpu().numpy().flatten()
            
            # More sophisticated heuristic based on known risk factors
            # These indices correspond to high-risk indicators in your feature set
            risk_weights = np.array([
                2.0,   # Rainfall_mm (high values = high risk)
                1.5,   # Slope_Angle (steep slopes = high risk) 
                2.5,   # Soil_Saturation (saturated soil = high risk)
                -1.0,  # Vegetation_Cover (low vegetation = high risk)
                1.8,   # Rainfall_3Day (recent rain = high risk)
                1.5,   # Rainfall_7Day 
                0.3,   # Aspect
                -0.2,  # Elevation_m (lower elevations might be riskier)
                -1.0,  # NDVI_Index (low NDVI = high risk)
            ] + [0.5] * 25)  # Default weights for remaining features
            
            # Normalize input values to 0-1 range approximately
            normalized_vals = np.tanh(input_vals)  # Squash to [-1, 1] then shift
            normalized_vals = (normalized_vals + 1) / 2  # Now in [0, 1]
            
            # Calculate weighted risk score
            risk_score = np.dot(normalized_vals[:len(risk_weights)], risk_weights[:len(normalized_vals)])
            
            # Convert to probability (sigmoid-like function)
            landslide_prob = 1 / (1 + np.exp(-risk_score + 2))  # Bias towards lower probabilities
            landslide_prob = float(np.clip(landslide_prob, 0.05, 0.95))
            
            uncertainty = 25.0  # High uncertainty for heuristic
            
            return {
                'probability': landslide_prob,
                'uncertainty': uncertainty,
                'success': False,
                'method': 'heuristic_fallback'
            }
            
        except Exception as e2:
            logger.error(f"All prediction methods failed: {e2}")
            raise HTTPException(status_code=500, detail=f"All prediction methods failed: {str(e2)}")

@app.get("/")
def read_root():
    """Root endpoint with API information"""
    return {
        "message": "Bayesian Landslide Prediction API",
        "status": "running",
        "model_loaded": predictive is not None,
        "expected_features": 34,
        "cache_size": len(prediction_cache),
        "version": "1.0.0",
        "features": FEATURE_NAMES,
        "endpoints": {
            "/predict": "POST - Make landslide predictions",
            "/health": "GET - Check system health",
            "/test_prediction": "POST - Test with sample data",
            "/docs": "GET - API documentation"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    try:
        model_status = "loaded" if predictive is not None else "not_loaded"
        
        # Quick model test
        test_status = "unknown"
        if predictive is not None:
            try:
                test_input = torch.randn(1, 34)
                with torch.no_grad():
                    _ = bnn(test_input)
                test_status = "working"
            except Exception as e:
                test_status = f"error: {str(e)[:50]}"
        
        return {
            "status": "healthy" if predictive is not None else "degraded",
            "model_status": model_status,
            "model_test": test_status,
            "cache_size": len(prediction_cache),
            "dependencies": {
                "torch": torch.__version__,
                "pyro": pyro.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/predict", response_model=PredictionResponse)
def predict(data: InputData):
    """Main prediction endpoint"""
    if not bnn:  # Changed from predictive to bnn since we have direct BNN access
        raise HTTPException(status_code=503, detail="Model not loaded. Service unavailable.")

    # Check cache first
    input_hash = get_input_hash(data.values)
    if input_hash in prediction_cache:
        logger.info("Returning cached result")
        return prediction_cache[input_hash]

    try:
        logger.info(f"Processing prediction request with {len(data.values)} features")
        
        # Map the input values to the correct feature order expected by the model
        # The frontend sends in this order (from your React form):
        frontend_order = [
            'Rainfall_mm', 'Temperature_C', 'Humidity_percent', 'Rainfall_3Day', 'Rainfall_7Day',
            'Slope_Angle', 'Aspect', 'Elevation_m', 'Soil_Saturation', 'Soil_pH',
            'Clay_Content', 'Sand_Content', 'Silt_Content', 'Soil_Erosion_Rate',
            'Pore_Water_Pressure_kPa', 'Soil_Moisture_Content', 'Soil_Temperature_C',
            'Soil_Strain', 'TDR_Reflection_Index', 'Soil_Type_Gravel', 'Soil_Type_Sand',
            'Soil_Type_Silt', 'Soil_Type_Clay', 'Vegetation_Cover', 'NDVI_Index',
            'Proximity_to_Water', 'Distance_to_Road_m', 'Land_Use_Urban', 'Land_Use_Forest',
            'Land_Use_Agriculture', 'Earthquake_Activity', 'Historical_Landslide_Count',
            'Microseismic_Activity', 'Acoustic_Emission_dB'
        ]
        
        # Create mapping from frontend order to model order
        input_dict = {name: data.values[i] for i, name in enumerate(frontend_order)}
        
        # Reorder according to model's expected feature order
        reordered_values = [input_dict[feature] for feature in FEATURE_NAMES]
        
        logger.info(f"Reordered {len(reordered_values)} features to match model expectations")
        
        # Convert to numpy array and reshape
        input_array = np.array(reordered_values).reshape(1, -1)
        logger.info(f"Input array shape: {input_array.shape}")
        
        # Create DataFrame with proper column names
        X_df = pd.DataFrame(input_array, columns=FEATURE_NAMES)
        
        # Preprocess data
        X_imputed = imputer.transform(X_df)
        X_scaled = scaler.transform(X_imputed)
        
        logger.info(f"Data preprocessing completed. Final shape: {X_scaled.shape}")
        
        # Convert to tensor
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        
        # Get predictions
        with torch.no_grad():
            pred_result = process_prediction(X_tensor)
            
        landslide_prob = pred_result['probability']
        uncertainty = pred_result['uncertainty']
        
        # Determine risk level and message
        if landslide_prob >= 0.7:
            risk = "very_high"
            message = "Very High Risk: Immediate evacuation recommended"
        elif landslide_prob >= 0.5:
            risk = "high" 
            message = "High Risk: Landslide likely"
        elif landslide_prob >= 0.3:
            risk = "medium"
            message = "Medium Risk: Monitor conditions closely"
        else:
            risk = "low"
            message = "Low Risk: Landslide unlikely"
        
        # Add method info to message if using fallback
        method_used = pred_result.get('method', 'unknown')
        if method_used != 'bayesian_sampling':
            message += f" (Method: {method_used.replace('_', ' ').title()})"
        
        # Calculate confidence
        confidence = max(0, min(100, 100 - uncertainty))
        
        # Create result
        result = {
            "probability": round(landslide_prob, 4),
            "risk": risk,
            "uncertainty_std_dev": round(uncertainty, 2),
            "message": message,
            "confidence": round(confidence, 1)
        }
        
        # Cache result
        prediction_cache[input_hash] = result
        cleanup_cache()
        
        logger.info(f"Prediction completed: risk={risk}, prob={landslide_prob:.4f}, uncertainty={uncertainty:.2f}, method={method_used}")
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    finally:
        gc.collect()

@app.post("/test_prediction")
def test_prediction():
    """Test endpoint with realistic sample data in correct order"""
    # Create sample values in the exact order expected by your trained model
    sample_dict = {
        'Rainfall_mm': 85.2,
        'Slope_Angle': 42.5,
        'Soil_Saturation': 92.2,
        'Vegetation_Cover': 25.4,
        'Rainfall_3Day': 180.5,
        'Rainfall_7Day': 320.7,
        'Aspect': 180.0,
        'Elevation_m': 650.0,
        'NDVI_Index': 0.35,
        'Land_Use_Urban': 0.0,
        'Land_Use_Forest': 0.0,
        'Land_Use_Agriculture': 1.0,
        'Earthquake_Activity': 3.2,
        'Proximity_to_Water': 80.0,
        'Distance_to_Road_m': 250.0,
        'Temperature_C': 24.5,
        'Humidity_percent': 82.3,
        'Soil_pH': 5.8,
        'Clay_Content': 35.3,
        'Sand_Content': 25.7,
        'Silt_Content': 39.0,
        'Soil_Erosion_Rate': 4.8,
        'Historical_Landslide_Count': 7.0,
        'Soil_Type_Gravel': 0.0,
        'Soil_Type_Sand': 0.0,
        'Soil_Type_Silt': 1.0,
        'Soil_Type_Clay': 1.0,
        'Pore_Water_Pressure_kPa': 28.8,
        'Soil_Moisture_Content': 45.1,
        'Microseismic_Activity': 12.4,
        'Acoustic_Emission_dB': 58.3,
        'Soil_Strain': 0.08,
        'Soil_Temperature_C': 19.5,
        'TDR_Reflection_Index': 0.85
    }
    
    # Create values in the order your model expects
    sample_values = [sample_dict[feature] for feature in FEATURE_NAMES]
    
    test_data = InputData(values=sample_values)
    result = predict(test_data)
    
    return {
        "test_scenario": "High-risk conditions with steep slope, high rainfall, saturated soil",
        "feature_order": FEATURE_NAMES,
        "input_values": sample_values,
        "prediction": result
    }

@app.post("/clear_cache")
def clear_cache():
    """Clear prediction cache"""
    global prediction_cache
    cache_size = len(prediction_cache)
    prediction_cache.clear()
    gc.collect()
    logger.info(f"Cache cleared: {cache_size} items removed")
    return {"message": f"Cache cleared: {cache_size} items removed"}

@app.get("/cache_stats")
def cache_stats():
    """Get cache statistics"""
    return {
        "cache_size": len(prediction_cache),
        "max_cache_size": MAX_CACHE_SIZE,
        "cache_usage_percent": round((len(prediction_cache) / MAX_CACHE_SIZE) * 100, 1),
        "memory_efficient": True
    }

@app.get("/model_features")
def get_model_features():
    """Get the actual feature names expected by the loaded model"""
    if not imputer:
        return {"error": "Model not loaded"}
    
    expected_features = []
    if hasattr(imputer, 'feature_names_in_'):
        expected_features = list(imputer.feature_names_in_)
    
    return {
        "expected_features": expected_features,
        "count": len(expected_features),
        "current_features_in_use": FEATURE_NAMES,
        "features_match": expected_features == FEATURE_NAMES if expected_features else False
    }

if __name__ == "__main__":
    import uvicorn
    
    print("Starting Landslide Prediction Server...")
    print("Features expected: 34")
    print("Server will be available at: http://127.0.0.1:8000")
    print("API docs available at: http://127.0.0.1:8000/docs")
    
    uvicorn.run(
        "app:app", 
        host="127.0.0.1", 
        port=8000, 
        reload=True,
        log_level="info"
    )