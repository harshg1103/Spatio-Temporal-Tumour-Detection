import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time

# --- IMPORTS ---
from dataset_loader import TumorGrowthDataset
from model import UNet3D

# --- CONFIGURATION ---
BATCH_SIZE = 4        
LEARNING_RATE = 1e-3
EPOCHS = 10           
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_PATH = "tumor_model.pth"

# Number of CPU cores to use for loading data.
# Set to 2 or 4. If you get a "BrokenPipe" error on Windows, set back to 0.
NUM_WORKERS = 2       

def dice_coefficient(pred, target, smooth=1e-5):
    pred = torch.sigmoid(pred)
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
    def forward(self, pred, target):
        return 1 - dice_coefficient(pred, target)

def train_model():
    print(f"--- Starting HIGH-PERFORMANCE Training on {DEVICE} ---")
    
    # 1. Load Data
    try:
        dataset = TumorGrowthDataset()
        # pin_memory=True speeds up transfer from RAM to GPU
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                          num_workers=NUM_WORKERS, pin_memory=True)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # 2. Initialize Model
    model = UNet3D(n_channels=2, n_classes=1).to(DEVICE)
    
    # 3. Optimization with Mixed Precision Scaler (Updated Syntax)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # NEW SYNTAX: torch.amp.GradScaler
    scaler = torch.amp.GradScaler('cuda') 
    
    bce_criterion = nn.BCEWithLogitsLoss()
    dice_criterion = DiceLoss()

    print(f"Training for {EPOCHS} epochs with Mixed Precision...")
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            
            # NEW SYNTAX: torch.amp.autocast
            with torch.amp.autocast('cuda'):
                predictions = model(inputs)
                loss_bce = bce_criterion(predictions, targets)
                loss_dice = dice_criterion(predictions, targets)
                total_loss = loss_bce + loss_dice
            
            # Scaled Backward Pass
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += total_loss.item()
            
            # Print every step to show it's alive
            print(f"   E{epoch+1} | Batch {batch_idx+1}/{len(loader)} | Loss: {total_loss.item():.4f}")

        # Summary
        elapsed = time.time() - start_time
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1}/{EPOCHS} Done | Avg Loss: {avg_loss:.4f} | Total Time: {elapsed/60:.1f} min")


    # 5. Save
    torch.save(model.state_dict(), SAVE_PATH)
    print(f"\n✅ Training Complete. Saved to '{SAVE_PATH}'")

if __name__ == "__main__":
    train_model()