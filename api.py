from fastapi import FastAPI, File, UploadFile, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
import uvicorn
import librosa
import numpy as np
import joblib
import shutil
import os
import traceback

# --- SECURITY ---
API_KEY_NAME = "x-api-key"
API_KEY_SECRET = "HCL_HACKATHON_2026_SECURE"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# --- LOAD MODEL ---
MODEL_PATH = "voice_auth_model.pkl"
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")
except:
    print("❌ CRITICAL ERROR: Model not found. Run train_model.py first.")
    model = None

app = FastAPI()

# --- AUTH FUNCTION ---
async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY_SECRET:
        return api_key_header
    else:
        raise HTTPException(status_code=403, detail="Invalid API Key")

# --- FEATURE EXTRACTION (Must match training!) ---
def extract_features_from_audio(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=22050, duration=3)
        expected_length = 22050 * 3
        if len(audio) < expected_length:
            padding = expected_length - len(audio)
            audio = np.pad(audio, (0, padding))
        else:
            audio = audio[:expected_length]

        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        cent = librosa.feature.spectral_centroid(y=audio, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(audio)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)

        return np.concatenate((
            np.mean(mfccs.T, axis=0),
            np.var(mfccs.T, axis=0),
            [np.mean(cent), np.mean(zcr)],
            np.mean(chroma.T, axis=0)
        ))
    except Exception:
        return None

# --- HEALTH CHECK (For Bot Readiness) ---
@app.get("/")
def health_check():
    return {"status": "active", "model_loaded": model is not None}

# --- MAIN ENDPOINT ---
@app.post("/predict-audio")
async def predict_audio(file: UploadFile = File(...), api_key: str = Depends(get_api_key)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    temp_filename = f"temp_{file.filename}"
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        features = extract_features_from_audio(temp_filename)
        
        if features is None:
            # Don't crash on bad audio, return a safe "fail" response
            raise HTTPException(status_code=422, detail="Could not process audio format")

        features = features.reshape(1, -1)
        prediction = model.predict(features)[0]
        # Get probability of it being FAKE
        probs = model.predict_proba(features)[0]
        classes = model.classes_
        
        # Find which index corresponds to 'fake'
        fake_index = list(classes).index("fake") if "fake" in classes else 0
        confidence_score = float(probs[fake_index])

        return {
            "status": "success",
            "prediction": prediction,         # "real" or "fake"
            "confidence": confidence_score,   # 0.95 (High confidence it's fake)
            "is_deepfake": prediction == "fake"
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)