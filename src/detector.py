import torch
import torch.nn as nn

class TumorDetector(nn.Module):
    """
    A lightweight 3D CNN for Binary Classification.
    Input: (1, Depth, Height, Width) -> MRI Scan
    Output: Probability (0.0 to 1.0) -> Tumor Presence
    """
    def __init__(self):
        super(TumorDetector, self).__init__()
        
        # Feature Extractor (Convolutional Layers)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2), 
            
            # Block 2
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            # Block 3
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            # Block 4
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(2),
        )
        
        # Classifier (Fully Connected Layers)
        # CALCULATION UPDATE for (64, 64, 32) Input:
        # After 4 MaxPools (dividing by 2 four times = division by 16):
        # 64/16 = 4
        # 64/16 = 4
        # 32/16 = 2
        # Final Flatten Size = 256 channels * 4 * 4 * 2 = 8192
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8192, 512), # <--- UPDATED FROM 65536 TO 8192
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1) 
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return torch.sigmoid(x)