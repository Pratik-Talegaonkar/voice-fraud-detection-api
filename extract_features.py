import os
import librosa
import numpy as np
import csv

# --- CONFIGURATION ---
DATASET_PATH = "dataset_augmented" # <--- POINTING TO THE NEW DATA
OUTPUT_CSV = "features.csv"
SAMPLE_RATE = 22050
DURATION = 3 
# ---------------------

def extract_features(file_path):
    try:
        # Load audio
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        
        # Pad or Truncate to exactly 3 seconds
        expected_length = SAMPLE_RATE * DURATION
        if len(audio) < expected_length:
            padding = expected_length - len(audio)
            audio = np.pad(audio, (0, padding))
        else:
            audio = audio[:expected_length]

        # 1. MFCC (Texture)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs.T, axis=0)
        mfcc_var = np.var(mfccs.T, axis=0)
        
        # 2. Spectral Centroid (Brightness - good for detecting vocoders)
        cent = librosa.feature.spectral_centroid(y=audio, sr=sr)
        cent_mean = np.mean(cent)
        
        # 3. Zero Crossing Rate (Noisiness)
        zcr = librosa.feature.zero_crossing_rate(audio)
        zcr_mean = np.mean(zcr)
        
        # 4. Chroma (Pitch/Notes)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)
        
        # Combine all into one array
        return np.concatenate((mfcc_mean, mfcc_var, [cent_mean, zcr_mean], chroma_mean))
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    # Define the Header (Must match the features above)
    header = ["filename", "language", "label"]
    
    # MFCCs (1-13 Mean, 1-13 Var)
    for i in range(1, 14): header.append(f"mfcc_mean_{i}")
    for i in range(1, 14): header.append(f"mfcc_var_{i}")
    
    # New Features
    header.append("spectral_centroid")
    header.append("zero_crossing_rate")
    
    # Chroma (12 Pitch Classes)
    for i in range(1, 13): header.append(f"chroma_{i}")

    print(f"🚀 Scanning '{DATASET_PATH}' for features...")
    
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        if not os.path.exists(DATASET_PATH):
            print(f"❌ Error: Folder '{DATASET_PATH}' not found!")
            return

        for lang in os.listdir(DATASET_PATH):
            lang_path = os.path.join(DATASET_PATH, lang)
            if os.path.isdir(lang_path):
                for label in ["real", "fake"]:
                    folder_path = os.path.join(lang_path, label)
                    
                    if os.path.exists(folder_path):
                        print(f"   📂 Processing: {lang.upper()} -> {label}") 
                        files = [f for f in os.listdir(folder_path) if f.endswith(('.mp3', '.wav'))]
                        
                        for filename in files:
                            file_path = os.path.join(folder_path, filename)
                            features = extract_features(file_path)
                            if features is not None:
                                row = [filename, lang, label] + features.tolist()
                                writer.writerow(row)

    print(f"\n🎉 DONE! Saved to '{OUTPUT_CSV}'.")

if __name__ == "__main__":
    main()