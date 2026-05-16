
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def compute_tumor_volume(mask_path, voxel_volume_mm3=1.0):
    """
    Compute tumor volume from a binary mask (.nii.gz).
    """
    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata()

    # Count voxels labeled as tumor (value > 0)
    tumor_voxels = np.sum(mask_data > 0)

    # Get voxel spacing from header
    voxel_spacing = mask_img.header.get_zooms()
    voxel_volume_mm3 = np.prod(voxel_spacing)

    # Volume in mm³ → cm³
    tumor_volume_cm3 = (tumor_voxels * voxel_volume_mm3) / 1000.0
    return tumor_volume_cm3


if __name__ == "__main__":
    # Example: two timepoints for the same patient
    mask_t1 = r"E:\EDI\tumor_growth_project\data\raw\MU-Glioma-Post\PatientID_0003\Timepoint_1\PatientID_0003_Timepoint_1_tumorMask.nii.gz"
    mask_t2 = r"E:\EDI\tumor_growth_project\data\raw\MU-Glioma-Post\PatientID_0003\Timepoint_2\PatientID_0003_Timepoint_2_tumorMask.nii.gz"

    vol_t1 = compute_tumor_volume(mask_t1)
    vol_t2 = compute_tumor_volume(mask_t2)

    print(f"Timepoint 1 volume: {vol_t1:.2f} cm³")
    print(f"Timepoint 2 volume: {vol_t2:.2f} cm³")

    # % Growth
    growth_pct = ((vol_t2 - vol_t1) / vol_t1) * 100 if vol_t1 > 0 else float("inf")
    print(f"Growth rate: {growth_pct:.2f}%")

# Plot a bar chart for comparison
plt.bar(["Timepoint 1", "Timepoint 2"], [vol_t1, vol_t2], color=["lightblue", "lightgreen"])
plt.ylabel("Tumor Volume (cm³)")
plt.title("Tumor Growth Across Timepoints")
plt.show()