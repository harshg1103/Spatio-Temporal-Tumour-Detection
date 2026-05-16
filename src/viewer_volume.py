import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# --- File path (update if needed) ---
scan_path = r"E:\EDI\tumor_growth_project\data\raw\MU-Glioma-Post\PatientID_0039\Timepoint_1\PatientID_0039_Timepoint_1_tumorMask.nii.gz"

# --- Load data ---
scan = nib.load(scan_path).get_fdata()

# --- Initial slice ---
slice_idx = scan.shape[2] // 2  # middle slice

fig, ax = plt.subplots(figsize=(6, 6))
plt.subplots_adjust(bottom=0.25)

scan_img = ax.imshow(scan[:, :, slice_idx], cmap="gray")
ax.set_title("MRI Scan (T1c)")
ax.axis("off")

# --- Slider setup ---
ax_slider = plt.axes([0.2, 0.1, 0.6, 0.05])
slider = Slider(ax_slider, 'Slice', 0, scan.shape[2]-1, valinit=slice_idx, valfmt='%0.0f')

def update(val):
    idx = int(slider.val)
    scan_img.set_data(scan[:, :, idx])
    fig.canvas.draw_idle()

slider.on_changed(update)
plt.show()