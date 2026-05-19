from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import onnxruntime as rt
import json
import os

app = FastAPI(
    title="ICU Sepsis Prediction API",
    description="FastAPI microservice serving our ONNX Householder QR Model with SHAP explanations.",
    version="1.0.0"
)

# Load ONNX model at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'sepsis_risk_linearized.onnx')
session = None

@app.on_event("startup")
def load_model():
    global session
    if os.path.exists(MODEL_PATH):
        session = rt.InferenceSession(MODEL_PATH)
    else:
        print(f"Warning: ONNX model not found at {MODEL_PATH}. Please run export_onnx.py first.")

class VitalsInput(BaseModel):
    spo2: float
    heart_rate: float
    temperature: float
    map_pressure: float
    lactate: float

@app.post("/predict")
def predict_risk(vitals: VitalsInput):
    if session is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    # Prepare input tensor
    x = np.array([[vitals.spo2, vitals.heart_rate, vitals.temperature, vitals.map_pressure, vitals.lactate]], dtype=np.float32)
    
    # ONNX Inference
    input_name = session.get_inputs()[0].name
    label_name = session.get_outputs()[0].name
    
    # ONNX gives us the linear prediction Y = A + BX
    pred_onx = session.run([label_name], {input_name: x})[0]
    
    # Convert log-risk back to exponential risk
    risk_score = float(np.exp(pred_onx[0][0]))
    
    return {
        "status": "success",
        "risk_score": risk_score,
        "is_critical": risk_score > 0.8
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": session is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
