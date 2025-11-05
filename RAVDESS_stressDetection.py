import librosa
import os
import pandas as pd
import numpy as np

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=3.0, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        return mfccs_mean
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def parse_label(file_name):
    emotion_labels = {
        '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
        '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
    }
    emotion_code = file_name.split('-')[2]
    return emotion_labels.get(emotion_code, 'unknown')

def map_to_stress_score(emotion):
    if emotion in ['calm', 'neutral']:
        return 1  # Low stress
    elif emotion == 'happy':
        return 2  # Low-moderate stress
    elif emotion == 'sad':
        return 3  # Moderate stress
    elif emotion in ['disgust', 'surprised']:
        return 4  # High stress
    elif emotion in ['angry', 'fearful']:
        return 5  # Very high stress
    else:
        return -1

def process_ravdess_dataset(ravdess_path, output_csv):
    data = []

    print("Starting feature extraction from RAVDESS dataset...")

    for actor_dir in os.listdir(ravdess_path):
        actor_path = os.path.join(ravdess_path, actor_dir)
        if os.path.isdir(actor_path):
            print(f"  Processing actor folder: {actor_dir}")
            
            for file_name in os.listdir(actor_path):
                if file_name.endswith('.wav'):
                    file_path = os.path.join(actor_path, file_name)
                    
                    features = extract_features(file_path)
                    if features is not None:
                        emotion = parse_label(file_name)
                        stress_score = map_to_stress_score(emotion)
                        data.append([file_name, emotion, stress_score] + features.tolist())

    # Create DataFrame
    columns = ['file_name', 'emotion', 'stress_score'] + [f'mfcc_{i+1}' for i in range(40)]
    df = pd.DataFrame(data, columns=columns)

    # Save to CSV
    df.to_csv(output_csv, index=False)
    
    print("\nFeature extraction complete.")
    print(f"Total files processed: {len(df)}")
    print(f"CSV file saved to: {output_csv}")
    print("\nSample rows:")
    print(df.head())

    return df

if __name__ == "__main__":
    RAVDESS_PATH = r'C:\CU\7th Sem\Capstone Project\Research Papers\Audio_Speech_Actors_01-24'
    OUTPUT_CSV = r'C:\CU\7th Sem\Capstone Project\Research Papers\RAVDESS_stress_scores.csv'
    
    if not os.path.exists(RAVDESS_PATH):
        print(f"Error: The directory '{RAVDESS_PATH}' does not exist.")
        print("Please download and unzip the RAVDESS audio-only files and update the RAVDESS_PATH variable.")
    else:
        final_df = process_ravdess_dataset(RAVDESS_PATH, OUTPUT_CSV)

