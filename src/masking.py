import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

# Paths (update these to match your dataset)
mri_path = r"E:\EDI\tumor_growth_project\data\raw\MU-Glioma-Post\PatientID_0003\Timepoint_1\PatientID_0003_Timepoint_1_brain_t1c.nii.gz"
mask_path = r"E:\EDI\tumor_growth_project\data\raw\MU-Glioma-Post\PatientID_0003\Timepoint_1\PatientID_0003_Timepoint_1_tumormask.nii.gz"

# Load images
mri_img = nib.load(mri_path).get_fdata()
mask_img = nib.load(mask_path).get_fdata()

# Pick a middle slice
slice_index = mri_img.shape[2] // 2
mri_slice = mri_img[:, :, slice_index]
mask_slice = mask_img[:, :, slice_index]

# Plot MRI + overlayed tumor mask
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(mri_slice.T, cmap="gray", origin="lower")
plt.title("MRI Slice")

plt.subplot(1, 2, 2)
plt.imshow(mri_slice.T, cmap="gray", origin="lower")
plt.imshow(mask_slice.T, cmap="Reds", alpha=0.4, origin="lower")  # overlay
plt.title("MRI + Tumor Mask Overlay")

plt.show()
