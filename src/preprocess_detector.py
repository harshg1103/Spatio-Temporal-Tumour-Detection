import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
import shutil

# --- CONFIGURATION ---
TARGET_SHAPE = (64, 64, 32)
BASE_DIR = r"E:\EDI\tumor_growth_project\data"
RAW_TUMOR_DIR = os.path.join(BASE_DIR, "raw", "MU-Glioma-Post")
RAW_HEALTHY_DIR = os.path.join(BASE_DIR, "raw", "Healthy")
SAVE_DIR = os.path.join(BASE_DIR, "processed_detector")

def resize_volume(img):
    current_shape = img.shape
    zoom_factors = [t / c for t, c in zip(TARGET_SHAPE, current_shape)]
    return zoom(img, zoom_factors, order=1)

def normalize(img):
    min_val, max_val = np.min(img), np.max(img)
    if max_val - min_val == 0: return img
    return (img - min_val) / (max_val - min_val)

def process_detector_data():
    if os.path.exists(SAVE_DIR): shutil.rmtree(SAVE_DIR)
    os.makedirs(SAVE_DIR)
    
    samples = [] 

    # --- 1. Process Tumor Data ---
    print("Processing Tumor Class...")
    if os.path.exists(RAW_TUMOR_DIR):
        patients = [p for p in os.listdir(RAW_TUMOR_DIR) if p.startswith("PatientID_")]
        for p in patients:
            p_dir = os.path.join(RAW_TUMOR_DIR, p, "Timepoint_1")
            if not os.path.exists(p_dir): continue
            
            scan_files = [f for f in os.listdir(p_dir) if "brain_t1c" in f or "t1" in f]
            if scan_files:
                try:
                    img_path = os.path.join(p_dir, scan_files[0])
                    img = nib.load(img_path).get_fdata()
                    img = normalize(resize_volume(img))
                    
                    fname = f"Tumor_{p}.npy"
                    np.save(os.path.join(SAVE_DIR, fname), img)
                    samples.append((fname, 1))
                    print(f"  + Added Tumor: {p}")
                except Exception as e:
                    print(f"  Error {p}: {e}")

    # --- 2. Process Healthy Data (Recursive Search) ---
    print("\nProcessing Healthy Class...")
    if os.path.exists(RAW_HEALTHY_DIR):
        healthy_scans = []
        
        # Walk through all subfolders to find .nii files
        for root, dirs, files in os.walk(RAW_HEALTHY_DIR):
            for file in files:
                if file.endswith('.nii') or file.endswith('.nii.gz'):
                    healthy_scans.append(os.path.join(root, file))
        
        print(f"  > Found {len(healthy_scans)} files in subfolders.")
        
        # Limit to 60 samples
        healthy_scans = healthy_scans[:60]
        
        for img_path in healthy_scans:
            try:
                # Get filename for saving
                filename = os.path.basename(img_path)
                safe_name = filename.replace(".nii.gz", "").replace(".nii", "")
                
                img = nib.load(img_path).get_fdata()
                img = normalize(resize_volume(img))
                
                fname = f"Healthy_{safe_name}.npy"
                np.save(os.path.join(SAVE_DIR, fname), img)
                samples.append((fname, 0))
                print(f"  - Added Healthy: {safe_name}")
            except Exception as e:
                print(f"  Error processing {img_path}: {e}")
    else:
        print(f"❌ Healthy folder missing!")

    np.save(os.path.join(SAVE_DIR, "labels.npy"), samples)
    print(f"\n--- Data Prep Complete. Total: {len(samples)} ---")

if __name__ == "__main__":
    process_detector_data()