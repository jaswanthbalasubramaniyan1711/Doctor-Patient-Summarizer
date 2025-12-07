import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import json
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.tensorboard import SummaryWriter

class EmotionDataset(Dataset):
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

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Training"):
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(pixel_values=pixel_values)
        loss = criterion(outputs.logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, accuracy, f1

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(pixel_values=pixel_values)
            loss = criterion(outputs.logits, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, accuracy, f1, all_preds, all_labels

def main():
    PROCESSED_DATA_DIR = "data/processed"
    MODEL_SAVE_DIR = "models"
    NUM_EPOCHS = 15
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    NUM_CLASSES = 5
    
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    with open(os.path.join(PROCESSED_DATA_DIR, 'dataset_info.json'), 'r') as f:
        dataset_info = json.load(f)
    
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    
    train_dataset = EmotionDataset(dataset_info['train'], processor)
    val_dataset = EmotionDataset(dataset_info['val'], processor)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True
    )
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    writer = SummaryWriter(os.path.join(MODEL_SAVE_DIR, 'runs'))
    
    best_val_acc = 0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1, _, _ = validate_epoch(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('F1/train', train_f1, epoch)
        writer.add_scalar('F1/val', val_f1, epoch)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(os.path.join(MODEL_SAVE_DIR, 'best_model'))
            processor.save_pretrained(os.path.join(MODEL_SAVE_DIR, 'best_model'))
            print(f"Saved best model with validation accuracy: {best_val_acc:.4f}")
    
    model.save_pretrained(os.path.join(MODEL_SAVE_DIR, 'final_model'))
    processor.save_pretrained(os.path.join(MODEL_SAVE_DIR, 'final_model'))
    
    writer.close()
    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    main()