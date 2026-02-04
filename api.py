from fastapi import FastAPI, File, UploadFile, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
import uvicorn
import librosa
import numpy as np
import joblib
import base64
import os
import uuid

# --- CONFIGURATION ---
API_KEY_NAME = "x-api-key"
API_KEY_SECRET = "HCL_HACKATHON_2026_SECURE"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# --- LOAD MODEL ---
MODEL_PATH = "voice_auth_model.pkl"
try:
    model = joblib.load(MODEL_PATH)
except:
    model = None

app = FastAPI()

# --- INPUT MODEL FOR BASE64 ---
class AudioRequest(BaseModel):
    audio: str  # This expects the Base64 string
    language: str = "english"
    format: str = "mp3"

# --- AUTH ---
async def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY_SECRET:
        return api_key_header
    else:
        raise HTTPException(status_code=403, detail="Invalid API Key")

# --- HELPER: FEATURE EXTRACTION ---
def extract_features(file_path):
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

# --- SHARED PREDICTION LOGIC ---
def predict_from_file(file_path):
    features = extract_features(file_path)
    if features is None:
        raise HTTPException(status_code=422, detail="Could not process audio.")
        
    features = features.reshape(1, -1)
    prediction = model.predict(features)[0]
    probs = model.predict_proba(features)[0]
    fake_index = list(model.classes_).index("fake") if "fake" in model.classes_ else 0
    confidence = float(probs[fake_index])
    
    return {
        "status": "success",
        "prediction": prediction,
        "confidence": confidence,
        "is_deepfake": prediction == "fake"
    }

# --- ENDPOINT 1: FILE UPLOAD (Keep this for safety) ---
@app.post("/predict-audio")
async def predict_audio_file(file: UploadFile = File(...), api_key: str = Depends(get_api_key)):
    temp_filename = f"temp_{uuid.uuid4()}.mp3"
    try:
        with open(temp_filename, "wb") as buffer:
            buffer.write(await file.read())
        return predict_from_file(temp_filename)
    finally:
        if os.path.exists(temp_filename): os.remove(temp_filename)

# --- ENDPOINT 2: BASE64 JSON (For the Tester/Bot) ---
@app.post("/predict-base64")
async def predict_audio_base64(request: AudioRequest, api_key: str = Depends(get_api_key)):
    temp_filename = f"temp_{uuid.uuid4()}.mp3"
    try:
        # Decode Base64 string to audio file
        audio_data = base64.b64decode(request.audio)
        with open(temp_filename, "wb") as f:
            f.write(audio_data)
        return predict_from_file(temp_filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid Base64: {str(e)}")
    finally:
        if os.path.exists(temp_filename): os.remove(temp_filename)

# --- HEALTH CHECK ---
@app.get("/docs") # Keeps UptimeRobot happy
def health_check(): return {"status": "active"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)