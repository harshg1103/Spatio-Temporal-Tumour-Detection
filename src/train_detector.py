import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from detector import TumorDetector # Imports the architecture from detector.py

# --- CONFIGURATION ---
DATA_DIR = r"E:\EDI\tumor_growth_project\data\processed_detector"
SAVE_PATH = "detector_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 20
BATCH_SIZE = 4

class ClassificationDataset(Dataset):
    def __init__(self):
        label_path = os.path.join(DATA_DIR, "labels.npy")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"labels.npy not found in {DATA_DIR}. Run preprocess_detector.py first.")
            
        self.data = np.load(label_path, allow_pickle=True)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        fname, label = self.data[idx]
        file_path = os.path.join(DATA_DIR, fname)
        
        img = np.load(file_path)
        
        # Convert to Tensor (1 Channel, Depth, Height, Width)
        img_tensor = torch.from_numpy(img).float().unsqueeze(0)
        
        # Label needs to be float for BCELoss
        label_tensor = torch.tensor([float(label)]).float()
        
        return img_tensor, label_tensor

def train():
    print(f"--- Starting Detector Training on {DEVICE} ---")
    
    # 1. Load Data
    try:
        dataset = ClassificationDataset()
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        print(f"Loaded {len(dataset)} samples.")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    # 2. Initialize Model
    model = TumorDetector().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.BCELoss() # Binary Cross Entropy Loss (for 0 vs 1)
    
    # 3. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # Stats
            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
        avg_loss = total_loss / len(loader)
        accuracy = 100 * correct / total
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")
        
    # 4. Save Model
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\n✅ Detector Training Complete.")
    print(f"Model saved to: {SAVE_PATH}")

if __name__ == "__main__":
    train()