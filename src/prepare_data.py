import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def audio_to_spectrogram(audio_path, save_path, sr=22050, n_mels=224):
    y, sr = librosa.load(audio_path, sr=sr)
    
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(mel_spec_db, aspect='auto', origin='lower', cmap='viridis')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=72)
    plt.close()
    
    img = Image.open(save_path)
    img = img.resize((224, 224))
    img.save(save_path)

def prepare_dataset(raw_data_dir, processed_data_dir, emotions):
    os.makedirs(processed_data_dir, exist_ok=True)
    
    dataset_info = {
        'train': [],
        'val': [],
        'test': []
    }
    
    for emotion in emotions:
        emotion_dir = os.path.join(raw_data_dir, emotion)
        if not os.path.exists(emotion_dir):
            print(f"Warning: {emotion_dir} does not exist")
            continue
            
        audio_files = [f for f in os.listdir(emotion_dir) if f.endswith(('.wav', '.mp3'))]
        
        train_files, temp_files = train_test_split(audio_files, test_size=0.3, random_state=42)
        val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
        
        for split, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
            split_dir = os.path.join(processed_data_dir, split, emotion)
            os.makedirs(split_dir, exist_ok=True)
            
            for audio_file in tqdm(files, desc=f"Processing {emotion} - {split}"):
                audio_path = os.path.join(emotion_dir, audio_file)
                spec_filename = os.path.splitext(audio_file)[0] + '.png'
                spec_path = os.path.join(split_dir, spec_filename)
                
                try:
                    audio_to_spectrogram(audio_path, spec_path)
                    dataset_info[split].append({
                        'image': spec_path,
                        'label': emotion
                    })
                except Exception as e:
                    print(f"Error processing {audio_path}: {e}")
    
    with open(os.path.join(processed_data_dir, 'dataset_info.json'), 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"\nDataset prepared:")
    print(f"Train: {len(dataset_info['train'])} samples")
    print(f"Val: {len(dataset_info['val'])} samples")
    print(f"Test: {len(dataset_info['test'])} samples")

if __name__ == "__main__":
    RAW_DATA_DIR = "data/raw"
    PROCESSED_DATA_DIR = "data/processed"
    EMOTIONS = ['angry', 'happy', 'sad', 'neutral', 'anxious']
    
    prepare_dataset(RAW_DATA_DIR, PROCESSED_DATA_DIR, EMOTIONS)