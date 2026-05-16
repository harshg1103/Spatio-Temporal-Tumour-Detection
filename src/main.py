import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

class TumorTracker:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.patients = sorted([d for d in os.listdir(base_dir) if d.startswith("PatientID_")])
    
    def select_patient(self):
        """Interactive console menu to select a patient."""
        print("\n--- Patient Selection ---")
        for i, p in enumerate(self.patients, 1):
            print(f" [{i}] {p}")
        
        try:
            choice = int(input("Choose a patient index: ").strip()) - 1
            if 0 <= choice < len(self.patients):
                return self.patients[choice]
            else:
                print("❌ Invalid range.")
                return None
        except ValueError:
            print("❌ Invalid input.")
            return None

    def get_patient_data(self, patient_id):
        """Scans directories to find timepoints, scans, and masks."""
        patient_dir = os.path.join(self.base_dir, patient_id)
        timepoints = sorted([d for d in os.listdir(patient_dir) if d.startswith("Timepoint_")])
        
        data = []
        for tp in timepoints:
            tp_dir = os.path.join(patient_dir, tp)
            # Find files (assuming standard naming convention from your example)
            scan_files = [f for f in os.listdir(tp_dir) if "brain_t1c" in f or "t1" in f]
            mask_files = [f for f in os.listdir(tp_dir) if "tumorMask" in f or "mask" in f]
            
            scan_path = os.path.join(tp_dir, scan_files[0]) if scan_files else None
            mask_path = os.path.join(tp_dir, mask_files[0]) if mask_files else None
            
            vol = self.compute_volume(mask_path) if mask_path else 0.0
            
            data.append({
                "timepoint": tp,
                "scan_path": scan_path,
                "mask_path": mask_path,
                "volume": vol
            })
        return data

    def compute_volume(self, mask_path):
        """Calculates volume in cm³."""
        if not mask_path or not os.path.exists(mask_path):
            return 0.0
        try:
            img = nib.load(mask_path)
            data = img.get_fdata()
            header = img.header
            voxel_vol_mm3 = np.prod(header.get_zooms())
            tumor_voxels = np.sum(data > 0)
            return (tumor_voxels * voxel_vol_mm3) / 1000.0
        except Exception as e:
            print(f"Error reading mask {mask_path}: {e}")
            return 0.0

    def plot_growth_curve(self, patient_id, data):
        """Plots the volume growth over timepoints."""
        tps = [d['timepoint'] for d in data]
        vols = [d['volume'] for d in data]

        if not vols or sum(vols) == 0:
            print("⚠️ No volume data to plot.")
            return

        plt.figure(figsize=(8, 5))
        plt.plot(tps, vols, marker='o', linestyle='-', color='firebrick', linewidth=2)
        plt.fill_between(tps, vols, color='firebrick', alpha=0.1)
        plt.title(f"Tumor Growth Evolution: {patient_id}")
        plt.ylabel("Tumor Volume (cm³)")
        plt.xlabel("Timeline")
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Calculate Growth Rate
        if len(vols) > 1 and vols[0] > 0:
            growth = ((vols[-1] - vols[0]) / vols[0]) * 100
            plt.figtext(0.15, 0.8, f"Total Growth: {growth:.2f}%", fontsize=10, 
                        bbox=dict(facecolor='white', alpha=0.8))
        
        plt.show()

    def view_multi_slice(self, scan_path, mask_path=None):
        """
        Advanced Viewer: Shows Axial, Sagittal, and Coronal views simultaneously.
        This is crucial for medical analysis validation.
        """
        if not scan_path or not os.path.exists(scan_path):
            print("❌ Scan file missing.")
            return

        img = nib.load(scan_path)
        data = img.get_fdata()
        mask = nib.load(mask_path).get_fdata() if mask_path and os.path.exists(mask_path) else None

        # Normalize data for display
        data = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)

        # Initial slices (middle of the brain)
        x_idx, y_idx, z_idx = [s // 2 for s in data.shape]

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        plt.subplots_adjust(bottom=0.25)

        # View 1: Axial (Transverse)
        im1 = ax[0].imshow(np.rot90(data[:, :, z_idx]), cmap="gray")
        if mask is not None:
            m1 = ax[0].imshow(np.rot90(np.ma.masked_where(mask[:, :, z_idx] == 0, mask[:, :, z_idx])), cmap="Reds", alpha=0.5)
        ax[0].set_title("Axial View")

        # View 2: Sagittal (Side)
        im2 = ax[1].imshow(np.rot90(data[x_idx, :, :]), cmap="gray")
        if mask is not None:
            m2 = ax[1].imshow(np.rot90(np.ma.masked_where(mask[x_idx, :, :] == 0, mask[x_idx, :, :])), cmap="Reds", alpha=0.5)
        ax[1].set_title("Sagittal View")

        # View 3: Coronal (Front)
        im3 = ax[2].imshow(np.rot90(data[:, y_idx, :]), cmap="gray")
        if mask is not None:
            m3 = ax[2].imshow(np.rot90(np.ma.masked_where(mask[:, y_idx, :] == 0, mask[:, y_idx, :])), cmap="Reds", alpha=0.5)
        ax[2].set_title("Coronal View")

        # Slider Setup
        ax_sl = plt.axes([0.2, 0.1, 0.6, 0.05])
        slider = Slider(ax_sl, 'Axial Slice', 0, data.shape[2]-1, valinit=z_idx, valfmt='%0.0f')

        def update(val):
            idx = int(slider.val)
            # Update Axial
            im1.set_data(np.rot90(data[:, :, idx]))
            if mask is not None:
                m1.set_data(np.rot90(np.ma.masked_where(mask[:, :, idx] == 0, mask[:, :, idx])))
            fig.canvas.draw_idle()

        slider.on_changed(update)
        plt.show()

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Update this path to your actual raw data path
    BASE_PATH = r"E:\EDI\tumor_growth_project\data\raw\MU-Glioma-Post"
    
    app = TumorTracker(BASE_PATH)
    
    selected_patient = app.select_patient()
    
    if selected_patient:
        print(f"\nProcessing {selected_patient}...")
        p_data = app.get_patient_data(selected_patient)
        
        # 1. Show Growth Statistics
        print(f"{'Timepoint':<15} | {'Volume (cm³)':<15} | {'Status'}")
        print("-" * 45)
        for d in p_data:
            status = "Has Data" if d['scan_path'] else "Missing"
            print(f"{d['timepoint']:<15} | {d['volume']:<15.2f} | {status}")
        
        # 2. Plot Chart
        app.plot_growth_curve(selected_patient, p_data)
        
        # 3. Interactive Visualization
        view_choice = input("\nView MRI scans interactively? (y/n): ").lower()
        if view_choice == 'y':
            print("Select Timepoint index to view (e.g., 1 for first timepoint):")
            try:
                idx = int(input("> ")) - 1
                if 0 <= idx < len(p_data):
                    target = p_data[idx]
                    print(f"Opening viewer for {target['timepoint']}...")
                    app.view_multi_slice(target['scan_path'], target['mask_path'])
                else:
                    print("Invalid index.")
            except ValueError:
                print("Invalid input.")