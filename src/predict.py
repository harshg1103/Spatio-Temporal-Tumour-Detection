import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from model import UNet3D

# --- CONFIGURATION ---
MODEL_PATH = "tumor_model.pth"
DATA_DIR = r"E:\EDI\tumor_growth_project\data\processed"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    """Re-initializes the U-Net and loads trained weights."""
    print(f"Loading model from {MODEL_PATH}...")
    model = UNet3D(n_channels=2, n_classes=1).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval() # Set to evaluation mode (freezes weights)
        print("✅ Model loaded successfully.")
        return model
    except FileNotFoundError:
        print("❌ Model file not found. Run 'train.py' first.")
        exit()

def get_patient_sample(patient_id, tp_idx=0):
    """
    Loads processed .npy files for a specific patient.
    Returns: input_tensor, ground_truth_mask, raw_scan (for plotting)
    """
    p_dir = os.path.join(DATA_DIR, patient_id)
    if not os.path.exists(p_dir):
        raise FileNotFoundError(f"Patient {patient_id} not found in processed data.")
    
    timepoints = sorted([d for d in os.listdir(p_dir) if d.startswith("Timepoint_")], 
                       key=lambda x: int(x.split('_')[-1]))
    
    if len(timepoints) < 2:
        raise ValueError("Patient needs at least 2 timepoints for prediction.")

    # Pair: T1 (Input) -> T2 (Target)
    tp_in = timepoints[tp_idx]
    tp_out = timepoints[tp_idx + 1]

    print(f"Predicting: {tp_in} -> {tp_out}")

    # Load NPY
    scan = np.load(os.path.join(p_dir, tp_in, "scan.npy"))
    mask_curr = np.load(os.path.join(p_dir, tp_in, "mask.npy"))
    mask_next = np.load(os.path.join(p_dir, tp_out, "mask.npy"))

    # Prepare Tensor (Add Batch Dim: 1, Channels, D, H, W)
    t_scan = torch.from_numpy(scan).float().unsqueeze(0)
    t_mask = torch.from_numpy(mask_curr).float().unsqueeze(0)
    
    # Input Channel 0: Scan, Channel 1: Mask
    input_tensor = torch.cat([t_scan, t_mask], dim=0).unsqueeze(0) # Shape: (1, 2, D, H, W)
    
    return input_tensor, mask_next, scan

def calculate_dice(pred, target):
    """Calculates accuracy (0.0 to 1.0)"""
    smooth = 1e-5
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def visualize_prediction(scan, input_mask, true_mask, pred_mask, patient_id, score):
    """
    Plots a 4-panel comparison of the middle slice.
    """
    # Pick middle slice of the depth
    d_idx = scan.shape[2] // 2 
    
    scan_slice = np.rot90(scan[:, :, d_idx])
    in_slice = np.rot90(input_mask[:, :, d_idx])
    true_slice = np.rot90(true_mask[:, :, d_idx])
    pred_slice = np.rot90(pred_mask[:, :, d_idx])

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    
    # 1. Input MRI
    ax[0].imshow(scan_slice, cmap="gray")
    ax[0].set_title("Input MRI (T1)")
    ax[0].axis('off')

    # 2. Input Tumor
    ax[1].imshow(scan_slice, cmap="gray", alpha=0.6)
    ax[1].imshow(np.ma.masked_where(in_slice==0, in_slice), cmap="Blues", alpha=0.7)
    ax[1].set_title("Starting Tumor (Input)")
    ax[1].axis('off')

    # 3. Ground Truth Future
    ax[2].imshow(scan_slice, cmap="gray", alpha=0.6)
    ax[2].imshow(np.ma.masked_where(true_slice==0, true_slice), cmap="Greens", alpha=0.7)
    ax[2].set_title("Real Future Tumor (Target)")
    ax[2].axis('off')

    # 4. AI Prediction
    ax[3].imshow(scan_slice, cmap="gray", alpha=0.6)
    ax[3].imshow(np.ma.masked_where(pred_slice==0, pred_slice), cmap="Reds", alpha=0.7)
    ax[3].set_title(f"AI Prediction (Dice: {score:.3f})")
    ax[3].axis('off')

    plt.suptitle(f"Tumor Growth Prediction: {patient_id}", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 1. Load Model
    model = load_model()

    # 2. Select Patient
    patients = sorted([d for d in os.listdir(DATA_DIR) if d.startswith("PatientID_")])
    print("\nAvailable Patients:")
    for i, p in enumerate(patients[:10]): # Show first 10
        print(f" [{i}] {p}")
    
    try:
        choice = int(input("Select patient index: "))
        patient_id = patients[choice]
        
        # 3. Get Data
        input_tensor, true_mask, raw_scan = get_patient_sample(patient_id)
        input_tensor = input_tensor.to(DEVICE)

        # 4. Run Inference
        print("Running Neural Network...")
        with torch.no_grad():
            output = model(input_tensor)
            
            # Apply Sigmoid (Logits -> Probability 0..1)
            probs = torch.sigmoid(output)
            
            # Apply Threshold (Probability -> Binary Mask 0 or 1)
            pred_mask = (probs > 0.5).float().cpu().numpy()[0, 0] # Remove Batch/Channel dims

        # 5. Calculate Score
        # Ensure true_mask is simple array
        if len(true_mask.shape) > 3: true_mask = true_mask[0] 
        
        dice_score = calculate_dice(pred_mask, true_mask)
        print(f"\n✅ Prediction Complete.")
        print(f"Dice Similarity Coefficient: {dice_score:.4f}")
        
        # 6. Visualize
        # For visualization, we need the Input Mask from the tensor
        input_mask_display = input_tensor.cpu().numpy()[0, 1] 
        
        visualize_prediction(raw_scan, input_mask_display, true_mask, pred_mask, patient_id, dice_score)

    except Exception as e:
        print(f"Error: {e}")