import torch
import numpy as np
import pandas as pd # You might need to pip install pandas
import os
from tqdm import tqdm # Progress bar (pip install tqdm)
from model import UNet3D

# --- CONFIGURATION ---
MODEL_PATH = "tumor_model.pth"
DATA_DIR = r"E:\EDI\tumor_growth_project\data\processed"
OUTPUT_CSV = "evaluation_results.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    print(f"Loading model from {MODEL_PATH}...")
    model = UNet3D(n_channels=2, n_classes=1).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

def calculate_dice(pred, target):
    smooth = 1e-5
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def evaluate_all():
    # 1. Setup
    model = load_model()
    patients = sorted([d for d in os.listdir(DATA_DIR) if d.startswith("PatientID_")])
    
    results = []
    print(f"\nStarting Batch Evaluation on {len(patients)} patients...")

    # 2. Loop through all patients
    for patient_id in tqdm(patients):
        p_dir = os.path.join(DATA_DIR, patient_id)
        
        # Get timepoints
        timepoints = sorted([d for d in os.listdir(p_dir) if d.startswith("Timepoint_")], 
                           key=lambda x: int(x.split('_')[-1]))
        
        if len(timepoints) < 2:
            continue

        # Evaluate every pair (T1->T2, T2->T3, etc.)
        for i in range(len(timepoints) - 1):
            tp_in = timepoints[i]
            tp_out = timepoints[i+1]
            
            try:
                # Load Data
                scan = np.load(os.path.join(p_dir, tp_in, "scan.npy"))
                mask_curr = np.load(os.path.join(p_dir, tp_in, "mask.npy"))
                mask_true = np.load(os.path.join(p_dir, tp_out, "mask.npy"))

                # Prepare Input
                t_scan = torch.from_numpy(scan).float().unsqueeze(0)
                t_mask = torch.from_numpy(mask_curr).float().unsqueeze(0)
                input_tensor = torch.cat([t_scan, t_mask], dim=0).unsqueeze(0).to(DEVICE)

                # Predict
                with torch.no_grad():
                    output = model(input_tensor)
                    probs = torch.sigmoid(output)
                    pred_mask = (probs > 0.5).float().cpu().numpy()[0, 0]

                # Score
                dice = calculate_dice(pred_mask, mask_true)
                
                results.append({
                    "Patient": patient_id,
                    "Pair": f"{tp_in} -> {tp_out}",
                    "Dice_Score": dice
                })
                
            except Exception as e:
                print(f"Error processing {patient_id}: {e}")

    # 3. Save Results
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    
    # 4. Final Report
    avg_score = df["Dice_Score"].mean()
    print("\n" + "="*30)
    print("       FINAL RESULTS       ")
    print("="*30)
    print(f"Total Pairs Tested: {len(df)}")
    print(f"Average Dice Score: {avg_score:.4f}")
    print(f"Best Result:        {df['Dice_Score'].max():.4f}")
    print(f"Worst Result:       {df['Dice_Score'].min():.4f}")
    print("="*30)
    print(f"Detailed report saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    evaluate_all()