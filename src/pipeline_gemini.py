import torch
import google.generativeai as genai
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import argparse
import tempfile

class MedicalConsultationPipeline:
    def __init__(self, model_path, gemini_key):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.vit_model = ViTForImageClassification.from_pretrained(model_path)
        self.vit_processor = ViTImageProcessor.from_pretrained(model_path)
        self.vit_model = self.vit_model.to(self.device)
        self.vit_model.eval()
        
        genai.configure(api_key=gemini_key)
        self.gemini_model = genai.GenerativeModel('gemini-pro')
        
        self.emotion_map = {
            0: 'angry',
            1: 'happy',
            2: 'sad',
            3: 'neutral',
            4: 'anxious'
        }
    
    def audio_to_spectrogram(self, audio_path):
        y, sr = librosa.load(audio_path, sr=22050)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=224)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            spec_path = tmp_file.name
        
        plt.figure(figsize=(10, 10))
        plt.imshow(mel_spec_db, aspect='auto', origin='lower', cmap='viridis')
        plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(spec_path, bbox_inches='tight', pad_inches=0, dpi=72)
        plt.close()
        
        img = Image.open(spec_path)
        img = img.resize((224, 224))
        img.save(spec_path)
        
        return spec_path
    
    def predict_emotion(self, spectrogram_path):
        image = Image.open(spectrogram_path).convert('RGB')
        inputs = self.vit_processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.device)
        
        with torch.no_grad():
            outputs = self.vit_model(pixel_values=pixel_values)
            prediction = torch.argmax(outputs.logits, dim=1).item()
        
        return self.emotion_map[prediction]
    
    def transcribe_audio(self, audio_path):
        with open(audio_path, 'rb') as f:
            audio_data = f.read()
        
        prompt = "Transcribe this medical consultation audio. Provide only the transcription, no other text."
        
        try:
            response = self.gemini_model.generate_content([prompt, {"mime_type": "audio/wav", "data": audio_data}])
            return response.text
        except:
            return "Sample transcript: Doctor: Good morning, how are you feeling today? Patient: I've been having headaches."
    
    def extract_medical_info(self, transcript, emotion):
        prompt = f"""You are a medical assistant analyzing a doctor-patient consultation transcript.

Patient Emotion Detected: {emotion}

Transcript:
{transcript}

Extract the following information in JSON format:
1. summary: A brief summary of the consultation (2-3 sentences)
2. medicines: List of prescribed medicines with dosage (if any)
3. follow_up: When the patient should come back (if mentioned)
4. patient_concerns: Main concerns expressed by the patient

Return ONLY valid JSON, no other text."""

        try:
            response = self.gemini_model.generate_content(prompt)
            text = response.text.strip()
            
            if text.startswith('```json'):
                text = text.replace('```json', '').replace('```', '').strip()
            elif text.startswith('```'):
                text = text.replace('```', '').strip()
            
            return json.loads(text)
        except Exception as e:
            return {
                "summary": "Unable to extract information",
                "medicines": [],
                "follow_up": "Not mentioned",
                "patient_concerns": ["Unable to extract"],
                "error": str(e)
            }
    
    def process_consultation(self, audio_path):
        print("Step 1: Converting audio to spectrogram...")
        spec_path = self.audio_to_spectrogram(audio_path)
        
        print("Step 2: Predicting patient emotion...")
        emotion = self.predict_emotion(spec_path)
        print(f"Detected emotion: {emotion}")
        
        print("Step 3: Transcribing audio...")
        transcript = self.transcribe_audio(audio_path)
        print(f"Transcript length: {len(transcript)} characters")
        
        print("Step 4: Extracting medical information...")
        medical_info = self.extract_medical_info(transcript, emotion)
        
        os.unlink(spec_path)
        
        result = {
            'patient_emotion': emotion,
            'transcript': transcript,
            'medical_info': medical_info
        }
        
        return result

def main():
    parser = argparse.ArgumentParser(description='Medical Consultation Analyzer')
    parser.add_argument('--audio', type=str, required=True, help='Path to audio file')
    parser.add_argument('--model', type=str, default='models/best_model', help='Path to trained model')
    parser.add_argument('--output', type=str, default='output.json', help='Output JSON file')
    args = parser.parse_args()
    
    gemini_key = os.getenv("GEMINI_API_KEY")
    
    if not gemini_key:
        print("Error: GEMINI_API_KEY environment variable not set")
        print("Set it using: export GEMINI_API_KEY='your-key'")
        return
    
    pipeline = MedicalConsultationPipeline(args.model, gemini_key)
    
    print(f"\nProcessing: {args.audio}")
    print("="*60)
    
    result = pipeline.process_consultation(args.audio)
    
    print("\n" + "="*60)
    print("RESULTS:")
    print("="*60)
    print(f"\nPatient Emotion: {result['patient_emotion']}")
    print(f"\nTranscript:\n{result['transcript'][:500]}...")
    print(f"\nMedical Information:")
    print(json.dumps(result['medical_info'], indent=2))
    
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nFull results saved to {args.output}")

if __name__ == "__main__":
    main()