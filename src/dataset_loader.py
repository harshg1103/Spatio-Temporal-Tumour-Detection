import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# --- CONFIGURATION ---
PROCESSED_DIR = r"E:\EDI\tumor_growth_project\data\processed"

class TumorGrowthDataset(Dataset):
    def __init__(self, data_dir=PROCESSED_DIR):
        """
        Custom PyTorch Dataset for Tumor Growth Prediction.
        It pairs (Timepoint_N) -> (Timepoint_N+1).
        """
        self.data_dir = data_dir
        self.samples = []
        
        # 1. Scan directory to build pairs
        self._prepare_dataset()

    def _prepare_dataset(self):
        """
        Iterates through patients and creates input-target pairs.
        Checks if files ACTUALLY exist before adding them.
        """
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Processed data directory not found: {self.data_dir}")

        patients = sorted([p for p in os.listdir(self.data_dir) if not p.startswith('.')])
        
        valid_pairs = 0
        
        for patient in patients:
            patient_dir = os.path.join(self.data_dir, patient)
            if not os.path.isdir(patient_dir):
                continue
                
            # Sort timepoints numerically
            timepoints = sorted(
                [d for d in os.listdir(patient_dir) if d.startswith("Timepoint_")],
                key=lambda x: int(x.split('_')[-1])
            )

            if len(timepoints) < 2:
                continue

            # Create pairs: (Current -> Next)
            for i in range(len(timepoints) - 1):
                tp_current = timepoints[i]
                tp_next = timepoints[i+1]
                
                # Define paths
                scan_path = os.path.join(patient_dir, tp_current, "scan.npy")
                mask_curr_path = os.path.join(patient_dir, tp_current, "mask.npy")
                mask_next_path = os.path.join(patient_dir, tp_next, "mask.npy")
                
                # --- ROBUST CHECK: Only add if ALL files exist ---
                if (os.path.exists(scan_path) and 
                    os.path.exists(mask_curr_path) and 
                    os.path.exists(mask_next_path)):
                    
                    self.samples.append({
                        "patient_id": patient,
                        "scan_path": scan_path,
                        "mask_input_path": mask_curr_path,
                        "mask_target_path": mask_next_path
                    })
                    valid_pairs += 1

        print(f"Dataset initialized: {valid_pairs} valid training pairs found.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Loads one sample.
        """
        sample = self.samples[idx]

        try:
            # 1. Load .npy files
            scan = np.load(sample["scan_path"])
            mask_current = np.load(sample["mask_input_path"])
            mask_future = np.load(sample["mask_target_path"])

            # 2. Convert to Tensor
            scan = torch.from_numpy(scan).float()
            mask_current = torch.from_numpy(mask_current).float()
            mask_future = torch.from_numpy(mask_future).float()

            # 3. Handle Dimensions (H, W, D) -> (D, H, W)
            # Assuming saved as (128, 128, 64)
            scan = scan.permute(2, 0, 1)
            mask_current = mask_current.permute(2, 0, 1)
            mask_future = mask_future.permute(2, 0, 1)

            # 4. Stack Inputs (Channel 0: Scan, Channel 1: Mask)
            input_tensor = torch.stack([scan, mask_current], dim=0)

            # 5. Format Target (Channel 0: Future Mask)
            target_tensor = mask_future.unsqueeze(0)

            return input_tensor, target_tensor
            
        except Exception as e:
            # Fallback for corrupt files during training (prevents crash)
            print(f"⚠️ Error loading sample {idx}: {e}. Returning zeros.")
            return torch.zeros(2, 64, 128, 128), torch.zeros(1, 64, 128, 128)

if __name__ == "__main__":
    # Test
    ds = TumorGrowthDataset()
    if len(ds) > 0:
        x, y = ds[0]
        print(f"Sample loaded. Input: {x.shape}, Target: {y.shape}")