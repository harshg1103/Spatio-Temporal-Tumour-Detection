import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import zoom, gaussian_filter

class GradCAM:
    """
    Advanced Grad-CAM for 3D Medical Imaging.
    Includes automated smoothing and thresholding for explainability.
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hooks to capture internal states
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_tensor, target_class=None):
        # 1. Enable Gradients (Crucial for Inference Mode)
        prev_grad_state = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        input_tensor.requires_grad_(True)
        
        # 2. Forward Pass
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # 3. Backward Pass (Capture Gradients)
        score = output[0][0] # Target the tumor class score
        score.backward(retain_graph=True)
        
        # 4. Restore State
        torch.set_grad_enabled(prev_grad_state)
        
        # 5. Get Captured Tensors
        grads = self.gradients.cpu().data.numpy()[0] # (Channels, D, H, W)
        fmap = self.activations.cpu().data.numpy()[0] # (Channels, D, H, W)
        
        # 6. Global Average Pooling (Weights)
        weights = np.mean(grads, axis=(1, 2, 3)) 
        
        # 7. Weighted Combination
        cam = np.zeros(fmap.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * fmap[i]
            
        # 8. ReLU (Focus on Positive Impact only)
        cam = np.maximum(cam, 0)
        
        # 9. Resize to Input Size (e.g., 64x64x32)
        input_shape = input_tensor.shape[2:] 
        zoom_factors = [i / c for i, c in zip(input_shape, cam.shape)]
        cam = zoom(cam, zoom_factors, order=1)
        
        # 10. Post-Processing: Smooth & Normalize
        # Apply Gaussian Blur for "Glow" effect
        cam = gaussian_filter(cam, sigma=1.0)
        
        # Normalize to 0-1
        if np.max(cam) > 0:
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
        
        return cam