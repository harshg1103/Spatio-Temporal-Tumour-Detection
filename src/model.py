import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    (Conv3D -> BatchNorm -> ReLU) * 2
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """
    Downscaling with MaxPool then DoubleConv
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """
    Upscaling then DoubleConv
    """
    def __init__(self, in_channels, skip_channels, out_channels, trilinear=True):
        super().__init__()

        # 1. Upsampling method
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        # 2. Convolution takes the concatenated size
        # Input to conv = (upsampled_current_layer) + (skip_connection_layer)
        # If trilinear: upsampled keeps 'in_channels' size.
        conv_input_size = in_channels + skip_channels
        
        self.conv = DoubleConv(conv_input_size, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle slight shape mismatch (padding)
        diffD = x2.size(2) - x1.size(2)
        diffH = x2.size(3) - x1.size(3)
        diffW = x2.size(4) - x1.size(4)

        x1 = F.pad(x1, [diffW // 2, diffW - diffW // 2,
                        diffH // 2, diffH - diffH // 2,
                        diffD // 2, diffD - diffD // 2])
        
        # Concatenate: x2 (Skip) + x1 (Upsampled)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # --- Encoder ---
        self.inc = DoubleConv(n_channels, 32)       # Out: 32
        self.down1 = Down(32, 64)                   # Out: 64
        self.down2 = Down(64, 128)                  # Out: 128
        self.down3 = Down(128, 256)                 # Out: 256
        
        # --- Decoder ---
        # Up(Current_In, Skip_In, Output)
        
        # Up1: Takes 256 (from down3), Concats 128 (from down2) -> Out 128
        self.up1 = Up(256, 128, 128, bilinear)
        
        # Up2: Takes 128 (from up1), Concats 64 (from down1) -> Out 64
        self.up2 = Up(128, 64, 64, bilinear)
        
        # Up3: Takes 64 (from up2), Concats 32 (from inc) -> Out 32
        self.up3 = Up(64, 32, 32, bilinear)
        
        # Final Classification
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        
        logits = self.outc(x)
        return logits

if __name__ == "__main__":
    # Test
    model = UNet3D(n_channels=2, n_classes=1)
    dummy = torch.randn(1, 2, 64, 128, 128)
    out = model(dummy)
    print(f"Shape Check: {out.shape}")