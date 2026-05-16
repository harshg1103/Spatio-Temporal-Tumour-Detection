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
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")

def resize_volume(img):
    """Resize to target shape, handling 4D hidden dimensions."""
    if img.ndim == 4: img = img.squeeze()
    current_shape = img.shape
    zoom_factors = [t / c for t, c in zip(TARGET_SHAPE, current_shape)]
    return zoom(img, zoom_factors, order=1)

def normalize(img):
    min_val, max_val = np.min(img), np.max(img)
    if max_val - min_val == 0: return img
    return (img - min_val) / (max_val - min_val)

def process_dataset():
    # Note: We do NOT delete the folder here to allow incremental updates
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # --- 1. PROCESS TUMOR PATIENTS ---
    print("Processing Tumor Patients (MU-Glioma)...")
    if os.path.exists(RAW_TUMOR_DIR):
        patients = sorted([p for p in os.listdir(RAW_TUMOR_DIR) if p.startswith("PatientID_")])
        for patient_id in patients:
            patient_raw_path = os.path.join(RAW_TUMOR_DIR, patient_id)
            patient_save_path = os.path.join(PROCESSED_DIR, patient_id)
            
            timepoints = sorted([d for d in os.listdir(patient_raw_path) if d.startswith("Timepoint_")])
            for tp in timepoints:
                tp_raw_path = os.path.join(patient_raw_path, tp)
                tp_save_path = os.path.join(patient_save_path, tp)
                os.makedirs(tp_save_path, exist_ok=True)

                scan_files = [f for f in os.listdir(tp_raw_path) if "brain_t1c" in f or "t1" in f]
                mask_files = [f for f in os.listdir(tp_raw_path) if "tumorMask" in f or "mask" in f]

                if not scan_files or not mask_files: continue

                try:
                    # Process Scan
                    scan = nib.load(os.path.join(tp_raw_path, scan_files[0])).get_fdata()
                    scan = normalize(resize_volume(scan))
                    np.save(os.path.join(tp_save_path, "scan.npy"), scan)

                    # Process Mask
                    mask = nib.load(os.path.join(tp_raw_path, mask_files[0])).get_fdata()
                    mask = resize_volume(mask) # No normalize for mask (it's 0 or 1)
                    mask = (mask > 0.5).astype(float) # Ensure binary
                    np.save(os.path.join(tp_save_path, "mask.npy"), mask)
                    
                    print(f"  ✔ Tumor: {patient_id} - {tp}")
                except Exception as e:
                    print(f"  ❌ Error {patient_id}: {e}")

    # --- 2. PROCESS HEALTHY PATIENTS ---
    print("\nProcessing Healthy Patients (IXI)...")
    if os.path.exists(RAW_HEALTHY_DIR):
        healthy_count = 0
        for root, dirs, files in os.walk(RAW_HEALTHY_DIR):
            for file in files:
                if file.endswith('.nii') or file.endswith('.nii.gz'):
                    if healthy_count >= 15: break # Limit to 15 healthy patients for display
                    
                    try:
                        safe_name = f"Healthy_{file.replace('.nii.gz', '').replace('.nii', '')}"
                        patient_save_path = os.path.join(PROCESSED_DIR, safe_name, "Timepoint_1")
                        os.makedirs(patient_save_path, exist_ok=True)

                        # Load & Process Scan
                        img_path = os.path.join(root, file)
                        if os.path.getsize(img_path) == 0: continue
                        
                        img = nib.load(img_path).get_fdata()
                        img = normalize(resize_volume(img))
                        
                        # Save Scan
                        np.save(os.path.join(patient_save_path, "scan.npy"), img)
                        
                        # Create & Save Dummy Mask (All Zeros)
                        dummy_mask = np.zeros(TARGET_SHAPE)
                        np.save(os.path.join(patient_save_path, "mask.npy"), dummy_mask)
                        
                        print(f"  ✔ Added: {safe_name}")
                        healthy_count += 1
                    except Exception as e:
                        print(f"  ❌ Error {file}: {e}")
    else:
        print("Healthy raw folder not found.")

    print("\n--- Processing Complete ---")

if __name__ == "__main__":
    process_dataset()