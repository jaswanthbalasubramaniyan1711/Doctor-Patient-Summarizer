import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm

class EmotionDataset:
    def __init__(self, data_info, processor):
        self.data = data_info
        self.processor = processor
        self.label_map = {
            'angry': 0,
            'happy': 1,
            'sad': 2,
            'neutral': 3,
            'anxious': 4
        }
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['image']).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt")
        label = self.label_map[item['label']]
        return {
            'pixel_values': inputs['pixel_values'].squeeze(),
            'labels': torch.tensor(label)
        }

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(pixel_values=pixel_values)
            preds = torch.argmax(outputs.logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels)

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    PROCESSED_DATA_DIR = "data/processed"
    MODEL_DIR = "models/best_model"
    RESULTS_DIR = "models/results"
    BATCH_SIZE = 16
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    with open(os.path.join(PROCESSED_DATA_DIR, 'dataset_info.json'), 'r') as f:
        dataset_info = json.load(f)
    
    processor = ViTImageProcessor.from_pretrained(MODEL_DIR)
    model = ViTForImageClassification.from_pretrained(MODEL_DIR)
    model = model.to(device)
    
    test_dataset = EmotionDataset(dataset_info['test'], processor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print("Evaluating model on test set...")
    preds, labels = evaluate_model(model, test_loader, device)
    
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    
    class_names = ['angry', 'happy', 'sad', 'neutral', 'anxious']
    
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=class_names))
    
    plot_confusion_matrix(labels, preds, class_names, 
                         os.path.join(RESULTS_DIR, 'confusion_matrix.png'))
    
    results = {
        'accuracy': float(accuracy),
        'f1_score': float(f1),
        'classification_report': classification_report(labels, preds, target_names=class_names, output_dict=True)
    }
    
    with open(os.path.join(RESULTS_DIR, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {RESULTS_DIR}")
    print(f"Confusion matrix saved as: {RESULTS_DIR}/confusion_matrix.png")

if __name__ == "__main__":
    main()