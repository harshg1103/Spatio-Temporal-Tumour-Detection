import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider



def compute_tumor_volume(mask_path):
    """
    Compute tumor volume (in cm³) from a binary mask (.nii.gz).
    """
    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata()

    tumor_voxels = np.sum(mask_data > 0)
    voxel_spacing = mask_img.header.get_zooms()
    voxel_volume_mm3 = np.prod(voxel_spacing)

    tumor_volume_cm3 = (tumor_voxels * voxel_volume_mm3) / 1000.0
    return tumor_volume_cm3


def view_scan_with_slider(scan_path, mask_path=None, title_suffix=""):
    scan = nib.load(scan_path).get_fdata()
    mask = nib.load(mask_path).get_fdata() if mask_path and os.path.exists(mask_path) else None

    slice_idx = scan.shape[2] // 2
    fig, ax = plt.subplots(figsize=(6, 6))
    plt.subplots_adjust(bottom=0.25)

    scan_img = ax.imshow(scan[:, :, slice_idx], cmap="gray")
    if mask is not None:
        mask_overlay = np.ma.masked_where(mask[:, :, slice_idx] == 0, mask[:, :, slice_idx])
        mask_img = ax.imshow(mask_overlay, cmap="autumn", alpha=0.4)
    else:
        mask_img = None

    ax.set_title(f"MRI Scan {title_suffix}")
    ax.axis("off")

    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.05])
    slider = Slider(ax_slider, "Slice", 0, scan.shape[2] - 1, valinit=slice_idx, valfmt="%0.0f")

    def update(val):
        idx = int(slider.val)
        scan_img.set_data(scan[:, :, idx])
        if mask_img is not None:
            mask_overlay = np.ma.masked_where(mask[:, :, idx] == 0, mask[:, :, idx])
            mask_img.set_data(mask_overlay)
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


if __name__ == "__main__":
    base_dir = r"E:\EDI\tumor_growth_project\data\raw\MU-Glioma-Post"

    # --- Patient selection ---
    patients = sorted([d for d in os.listdir(base_dir) if d.startswith("PatientID_")])
    print("Available patients:")
    for i, p in enumerate(patients, 1):
        print(f" [{i}] {p}")
    choice = input("Choose a patient index: ").strip()
    try:
        patient = patients[int(choice) - 1]
    except:
        print("❌ Invalid choice. Exiting.")
        exit()

    patient_dir = os.path.join(base_dir, patient)
    timepoints = sorted([d for d in os.listdir(patient_dir) if d.startswith("Timepoint_")])

    tp_labels, details, vols = [], [], []
    for tp in timepoints:
        tp_dir = os.path.join(patient_dir, tp)
        scan_file = [f for f in os.listdir(tp_dir) if "brain_t1c" in f]
        mask_file = [f for f in os.listdir(tp_dir) if "tumorMask" in f]

        scan_full = os.path.join(tp_dir, scan_file[0]) if scan_file else None
        mask_full = os.path.join(tp_dir, mask_file[0]) if mask_file else None

        vol = compute_tumor_volume(mask_full) if mask_full else None
        tp_labels.append(tp)
        details.append((scan_full, mask_full, vol))
        vols.append(vol)

    # --- Show growth chart ---
    if len(vols) > 1:
        plt.plot(tp_labels, vols, marker="o", color="blue", linestyle="-")
        plt.xlabel("Timepoints")
        plt.ylabel("Tumor Volume (cm³)")
        plt.title(f"Tumor Growth for {patient}")
        plt.grid(True)
        plt.show()

        growth_pct = ((vols[-1] - vols[0]) / vols[0]) * 100 if vols[0] > 0 else float("inf")
        print(f"Overall Growth Rate from {tp_labels[0]} → {tp_labels[-1]}: {growth_pct:.2f}%")
    else:
        print("⚠️ Not enough timepoints to compute growth.")

    # --- Ask to open viewer ---
    view_choice = input("Open interactive viewer for a timepoint? (y/n): ").strip().lower()
    if view_choice == "y":
        print("Available timepoints:")
        for i, (tp, det) in enumerate(zip(tp_labels, details), 1):
            scan_p, mask_p, _ = det
            status = []
            status.append("scan✓" if scan_p and os.path.exists(scan_p) else "scan✗")
            status.append("mask✓" if mask_p and os.path.exists(mask_p) else "mask✗")
            print(f" [{i}] {tp} ({', '.join(status)})")

        sel = input("Choose timepoint index to view: ").strip()
        try:
            idx = int(sel) - 1
            scan_p, mask_p, _ = details[idx]
            if not scan_p or not os.path.exists(scan_p):
                print("[Error] Scan not found for selected timepoint.")
            else:
                view_scan_with_slider(scan_p, mask_p, title_suffix=f"{patient} {tp_labels[idx]}")
        except Exception as e:
            print("Invalid selection:", e)
