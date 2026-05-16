import streamlit as st
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import torch
from scipy.ndimage import gaussian_filter
from model import UNet3D
from detector import TumorDetector
from gradcam import GradCAM

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
BASE_PATH = r"E:\EDI\tumor_growth_project\data\raw\MU-Glioma-Post"
PROCESSED_PATH = r"E:\EDI\tumor_growth_project\data\processed"
MODEL_PATH = "tumor_model.pth"
DETECTOR_PATH = "detector_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="Neuro-Oncology AI", layout="wide", page_icon="🧠")

# Custom CSS for a professional look
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; }
    h1, h2, h3 { color: #f0f2f6; font-family: 'Segoe UI', sans-serif; }
    .metric-container { background-color: #262730; padding: 15px; border-radius: 8px; border: 1px solid #41444e; }
    .stButton>button { 
        width: 100%; border-radius: 8px; height: 50px; 
        background: linear-gradient(90deg, #ff4b4b 0%, #ff6b6b 100%); 
        color: white; font-weight: bold; border: none;
    }
    .stButton>button:hover { box-shadow: 0 4px 10px rgba(255, 75, 75, 0.4); }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. HELPER FUNCTIONS & LOADERS
# ==========================================
@st.cache_data
def get_patient_list():
    """Scans the PROCESSED folder for both Tumor and Healthy patients."""
    if os.path.exists(PROCESSED_PATH):
        # We check for folders that contain Timepoint subfolders
        return sorted([d for d in os.listdir(PROCESSED_PATH) if os.path.isdir(os.path.join(PROCESSED_PATH, d))])
    return []

@st.cache_resource
def load_models():
    """Loads both AI models (Detector and Predictor) into GPU/CPU memory."""
    models = {}
    
    # Load Detector (Stage 1)
    if os.path.exists(DETECTOR_PATH):
        try:
            det = TumorDetector().to(DEVICE)
            det.load_state_dict(torch.load(DETECTOR_PATH, map_location=DEVICE))
            det.eval()
            models['detector'] = det
        except Exception as e:
            st.error(f"Error loading Detector: {e}")
    
    # Load Predictor (Stage 2)
    if os.path.exists(MODEL_PATH):
        try:
            pred = UNet3D(n_channels=2, n_classes=1).to(DEVICE)
            pred.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            pred.eval()
            models['predictor'] = pred
        except Exception as e:
            st.error(f"Error loading Predictor: {e}")
            
    return models

def compute_volume(mask_path):
    """Calculates tumor volume in cm³ from a binary mask."""
    if not os.path.exists(mask_path): return 0.0
    try:
        mask = np.load(mask_path)
        # Assuming isotropic 1mm³ resizing in preprocess steps (128x128x64 or 64x64x32)
        # For normalized/resized data we estimate using voxel count.
        voxel_vol_mm3 = 1.0 
        return (np.sum(mask > 0) * voxel_vol_mm3) / 1000.0
    except:
        return 0.0

def load_scan_normalized(path):
    """Loads .npy scan and normalizes it to 0-1 range."""
    img = np.load(path)
    return (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)

# ==========================================
# 3. SIDEBAR & NAVIGATION
# ==========================================
models = load_models()
patients = get_patient_list()

st.sidebar.title("🩺 Neuro-Oncology AI")
st.sidebar.markdown(f"**System Status:** {'🟢 Online' if models else '🔴 Offline'}")

if patients:
    selected_patient = st.sidebar.selectbox("Select Patient Record", patients)
    # Path is now primarily the PROCESSED folder
    p_dir = os.path.join(PROCESSED_PATH, selected_patient)
    timepoints = sorted([d for d in os.listdir(p_dir) if d.startswith("Timepoint_")], 
                       key=lambda x: int(x.split('_')[-1]))
else:
    selected_patient = None
    timepoints = []
    st.sidebar.warning("No processed data found. Run `preprocess.py`.")

st.sidebar.markdown("---")
mode = st.sidebar.radio("Operation Mode", ["📊 Analysis Dashboard", "🧠 AI Diagnosis & Prediction"])

# ==========================================
# 4. MODE: ANALYSIS DASHBOARD
# ==========================================
if mode == "📊 Analysis Dashboard" and selected_patient:
    st.title(f"Patient Analysis: {selected_patient}")
    
    # Distinction: Is this a Healthy control or a Tumor patient?
    is_healthy = "Healthy" in selected_patient
    
    if is_healthy:
        # --- HEALTHY PATIENT VIEW ---
        st.success("✅ **Patient identified as Healthy Reference (IXI Dataset).**")
        st.info("This record is used as a negative control for training the Tumor Detector.")
        
        col1, col2 = st.columns(2)
        col1.metric("Tumor Volume", "0.00 cm³", delta="Normal")
        col1.metric("Risk Factor", "Low", delta="-100%")
        
        # Healthy Viewer
        if timepoints:
            scan_path = os.path.join(p_dir, timepoints[0], "scan.npy")
            if os.path.exists(scan_path):
                scan = load_scan_normalized(scan_path)
                slice_idx = col2.slider("Axial Slice View", 0, scan.shape[2]-1, scan.shape[2]//2)
                
                fig, ax = plt.subplots(figsize=(4,4))
                ax.imshow(np.rot90(scan[:, :, slice_idx]), cmap="gray")
                ax.axis('off')
                fig.patch.set_facecolor('#0e1117')
                col2.pyplot(fig)
                
    else:
        # --- TUMOR PATIENT VIEW ---
        growth_data = []
        tp_labels = []
        
        # Calculate volumes
        for tp in timepoints:
            mask_path = os.path.join(p_dir, tp, "mask.npy")
            vol = compute_volume(mask_path)
            growth_data.append(vol)
            tp_labels.append(tp)

        # Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Longitudinal Scans", len(timepoints))
        c2.metric("Latest Volume", f"{growth_data[-1]:.2f} cm³" if growth_data else "0.00 cm³")
        
        growth_pct = 0.0
        if len(growth_data) > 1 and growth_data[0] > 0:
            growth_pct = ((growth_data[-1] - growth_data[0]) / growth_data[0]) * 100
        c3.metric("Total Growth", f"{growth_pct:.1f}%", delta_color="inverse")

        # Growth Chart
        st.subheader("Tumor Growth Trajectory")
        if len(growth_data) > 0:
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(tp_labels, growth_data, color='#FF4B4B', marker='o', linewidth=2)
            ax.fill_between(tp_labels, growth_data, color='#FF4B4B', alpha=0.1)
            ax.set_ylabel("Volume (cm³)")
            ax.grid(True, linestyle='--', alpha=0.2)
            ax.set_facecolor('#0e1117')
            fig.patch.set_facecolor('#0e1117')
            ax.tick_params(colors='white')
            ax.yaxis.label.set_color('white')
            ax.xaxis.label.set_color('white')
            for spine in ax.spines.values(): spine.set_edgecolor('#41444e')
            st.pyplot(fig)

        # --- 3D SLICER (UPGRADED) ---
        st.subheader("MRI Slice Inspection (3D Slicer)")
        if timepoints:
            selected_tp = st.selectbox("Select Timepoint", timepoints)
            
            scan_path = os.path.join(p_dir, selected_tp, "scan.npy")
            mask_path = os.path.join(p_dir, selected_tp, "mask.npy")
            
            if os.path.exists(scan_path) and os.path.exists(mask_path):
                scan = load_scan_normalized(scan_path)
                mask = np.load(mask_path)
                
                # --- Multi-View Logic ---
                view = st.radio("Select View Plane", ["Axial (Top)", "Coronal (Front)", "Sagittal (Side)"], horizontal=True)
                
                # Slicing logic based on dimensions (64, 64, 32)
                # Axial: Z-axis (2), Coronal: Y-axis (1), Sagittal: X-axis (0)
                if view == "Axial (Top)":
                    max_slice = scan.shape[2] - 1
                    slice_idx = st.slider("Slice Index", 0, max_slice, max_slice//2)
                    img_slice = scan[:, :, slice_idx]
                    mask_slice = mask[:, :, slice_idx]
                elif view == "Coronal (Front)":
                    max_slice = scan.shape[1] - 1
                    slice_idx = st.slider("Slice Index", 0, max_slice, max_slice//2)
                    img_slice = scan[:, slice_idx, :]
                    mask_slice = mask[:, slice_idx, :]
                else: # Sagittal
                    max_slice = scan.shape[0] - 1
                    slice_idx = st.slider("Slice Index", 0, max_slice, max_slice//2)
                    img_slice = scan[slice_idx, :, :]
                    mask_slice = mask[slice_idx, :, :]

                # Rotation Fixer
                # rot = st.slider("Adjust Rotation", 0, 270, 90, 90)
                img_slice = np.rot90(img_slice, k=1)
                mask_slice = np.rot90(mask_slice, k=1)

                # Plot
                fig_s, ax_s = plt.subplots(figsize=(6,6))
                # aspect='auto' prevents squashed images for non-axial views
                ax_s.imshow(img_slice, cmap="gray", aspect='auto') 
                
                # Overlay mask
                m = np.ma.masked_where(mask_slice == 0, mask_slice)
                ax_s.imshow(m, cmap="autumn", alpha=0.6, aspect='auto')
                
                ax_s.axis('off')
                fig_s.patch.set_facecolor('#0e1117')
                st.pyplot(fig_s)

# ==========================================
# 5. MODE: AI DIAGNOSIS & PREDICTION
# ==========================================
elif mode == "🧠 AI Diagnosis & Prediction" and selected_patient:
    st.title("🤖 AI Diagnostic Pipeline")
    
    if len(timepoints) == 0:
        st.error("No data available for diagnosis.")
    else:
        # Input Selection
        tp_idx = st.selectbox("Select Input Scan (T1)", timepoints)
        
        try:
            current_idx = timepoints.index(tp_idx)
            if current_idx + 1 < len(timepoints):
                tp_next_idx = timepoints[current_idx + 1]
                has_target = True
            else:
                tp_next_idx = None
                has_target = False
        except ValueError:
            has_target = False

        if has_target:
            st.info(f"Targeting Growth: {tp_idx} ➔ {tp_next_idx}")
        else:
            st.warning("No future timepoint available for accuracy validation. Running in Prediction-Only mode.")

        # Debug Option
        use_dummy = st.checkbox("🛠️ Test with Empty Brain (Simulate Healthy)")

        # RUN BUTTON
        if st.button("🚀 RUN DIAGNOSIS"):
            # Load Data
            scan_path = os.path.join(p_dir, tp_idx, "scan.npy")
            mask_curr_path = os.path.join(p_dir, tp_idx, "mask.npy")
            
            if not os.path.exists(scan_path):
                st.error("Scan file missing.")
                st.stop()
                
            scan = load_scan_normalized(scan_path)
            mask_curr = np.load(mask_curr_path) if os.path.exists(mask_curr_path) else np.zeros_like(scan)
            
            # --- STAGE 1: DETECTION ---
            st.markdown("### 🔍 Stage 1: Tumor Detection")
            
            if 'detector' not in models:
                st.error("Detector Model not found. Please train it first.")
            else:
                # Prepare Input
                if use_dummy:
                    st.warning("⚠️ TESTING MODE: Feeding empty brain volume...")
                    det_input = torch.zeros(1, 1, scan.shape[0], scan.shape[1], scan.shape[2]).to(DEVICE)
                else:
                    det_input = torch.from_numpy(scan).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
                
                # Grad-CAM Setup
                target_layer = models['detector'].features[12] # Last Conv Layer
                cam_extractor = GradCAM(models['detector'], target_layer)
                
                with st.spinner("Analyzing brain structure & generating heatmap..."):
                    prob = models['detector'](det_input).item()
                    heatmap = cam_extractor.generate_cam(det_input)
                
                # Display Detection Results
                d_col1, d_col2 = st.columns([1, 4])
                d_col1.metric("Probability", f"{prob*100:.2f}%")
                
                if prob < 0.5:
                    d_col2.success("✅ **Result: Normal.** No tumor detected.")
                    st.info("Pipeline terminated. (Growth Prediction skipped)")
                else:
                    d_col2.error("⚠️ **Result: Tumor Detected.** Initiating Stage 2.")
                    
                    # --- 1. VISUALIZE GRAD-CAM (SMOOTH & MASKED) ---
                    st.subheader("👁️ Explainable AI (Grad-CAM)")
                    
                    # Find best slice
                    if use_dummy:
                        max_slice = scan.shape[2]//2
                    else:
                        # Find max activation slice
                        max_slice = np.unravel_index(np.argmax(heatmap), heatmap.shape)[2]
                    
                    # Get Slices
                    scan_slice = np.rot90(scan[:, :, max_slice])
                    hm_slice = np.rot90(heatmap[:, :, max_slice])
                    
                    # Apply Gaussian Smoothing to Heatmap
                    hm_smooth = gaussian_filter(hm_slice, sigma=2)  # Smoother
                    
                    # Robust Normalization (Scale to 0-1 range for plotting)
                    if np.max(hm_smooth) > 0:
                        hm_smooth = (hm_smooth - np.min(hm_smooth)) / (np.max(hm_smooth) - np.min(hm_smooth) + 1e-8)
                    
                    # NO MASKING HERE - Just show the raw heatmap overlay
                    # This ensures colors appear even if signal is weak or background is noisy

                    gc1, gc2 = st.columns(2)
                    with gc1:
                        st.write(f"**MRI Scan (Slice {max_slice})**")
                        f, ax = plt.subplots()
                        ax.imshow(scan_slice, cmap='gray')
                        ax.axis('off')
                        f.patch.set_facecolor('#0e1117')
                        st.pyplot(f)
                    with gc2:
                        st.write("**AI Attention Heatmap**")
                        f, ax = plt.subplots()
                        ax.imshow(scan_slice, cmap='gray', alpha=0.5)
                        # Display heatmap without cutting it off
                        ax.imshow(hm_smooth, cmap='jet', alpha=0.6, vmin=0, vmax=1)
                        ax.axis('off')
                        f.patch.set_facecolor('#0e1117')
                        st.pyplot(f)
                    
                    # --- 2. STAGE 2: SEGMENTATION ---
                    st.markdown("---")
                    st.markdown("### 🚀 Stage 2: Future Tumor Segmentation")
                    
                    if 'predictor' not in models:
                        st.error("Predictor Model not found.")
                    else:
                        with st.spinner("Forecasting Spatio-Temporal Evolution..."):
                            t_scan = torch.from_numpy(scan).float().unsqueeze(0)
                            t_mask = torch.from_numpy(mask_curr).float().unsqueeze(0)
                            input_tensor = torch.cat([t_scan, t_mask], dim=0).unsqueeze(0).to(DEVICE)
                            
                            with torch.no_grad():
                                output = models['predictor'](input_tensor)
                                pred_mask = (torch.sigmoid(output) > 0.5).float().cpu().numpy()[0, 0]
                        
                        # HELPER FUNCTION FOR ROBUST PLOTTING
                        def plot_overlay(bg, mask, color_cmap, title):
                            f, ax = plt.subplots()
                            ax.imshow(bg, cmap='gray')
                            
                            # Check if mask is empty
                            if np.sum(mask) == 0:
                                ax.text(5, 10, "Empty / No Tumor", color='white', fontsize=8, bbox=dict(facecolor='black', alpha=0.5))
                            else:
                                # Strict Binary & Range Forcing
                                mask_binary = mask > 0.5
                                masked_layer = np.ma.masked_where(~mask_binary, mask_binary)
                                # vmin=0, vmax=1 ensures the color is MAX BRIGHTNESS
                                ax.imshow(masked_layer, cmap=color_cmap, alpha=0.8, vmin=0, vmax=1)
                            
                            ax.axis('off')
                            ax.set_title(title, color='white', fontsize=10)
                            f.patch.set_facecolor('#0e1117')
                            return f

                        cols = st.columns(3 if has_target else 2)
                        
                        # Input
                        with cols[0]:
                            slice_curr = np.rot90(mask_curr[:,:,max_slice])
                            fig = plot_overlay(scan_slice, slice_curr, 'Blues', "Input (Current)")
                            st.pyplot(fig)
                        
                        # Ground Truth
                        if has_target:
                            mask_next_path = os.path.join(p_dir, tp_next_idx, "mask.npy")
                            if os.path.exists(mask_next_path):
                                mask_true = np.load(mask_next_path)
                                smooth = 1e-5
                                intersection = (pred_mask * mask_true).sum()
                                dice = (2. * intersection + smooth) / (pred_mask.sum() + mask_true.sum() + smooth)
                                st.success(f"Segmentation Complete. Accuracy (Dice Score): **{dice:.4f}**")
                                
                                with cols[1]:
                                    slice_true = np.rot90(mask_true[:,:,max_slice])
                                    fig = plot_overlay(scan_slice, slice_true, 'Greens', "Ground Truth (Future)")
                                    st.pyplot(fig)
                                
                                with cols[2]:
                                    slice_pred = np.rot90(pred_mask[:,:,max_slice])
                                    fig = plot_overlay(scan_slice, slice_pred, 'Reds', "AI Prediction")
                                    st.pyplot(fig)
                            else:
                                st.warning("Target mask missing.")
                        else:
                            with cols[1]:
                                slice_pred = np.rot90(pred_mask[:,:,max_slice])
                                fig = plot_overlay(scan_slice, slice_pred, 'Reds', "AI Prediction")
                                st.pyplot(fig)