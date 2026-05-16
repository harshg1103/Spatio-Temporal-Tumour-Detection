import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import zoom

class GradCAM:
    """
    Robust Grad-CAM engine for 3D Medical Imaging.
    Forces gradient calculation even in inference mode.
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
        # 1. CRITICAL: Enable Gradients (Fixes "No Heatmap" issue)
        prev_grad_state = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        input_tensor.requires_grad_(True)
        
        # 2. Forward Pass
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # 3. Backward Pass (Capture Gradients)
        # We target the 'Tumor' class score (index 0 since binary)
        score = output[0][0]
        score.backward(retain_graph=True)
        
        # 4. Restore previous state
        torch.set_grad_enabled(prev_grad_state)
        
        # 5. Check if we captured data
        if self.gradients is None or self.activations is None:
            return np.zeros(input_tensor.shape[2:])

        # 6. Process Gradients
        grads = self.gradients.cpu().data.numpy()[0]
        fmap = self.activations.cpu().data.numpy()[0]
        
        # Global Average Pooling
        weights = np.mean(grads, axis=(1, 2, 3)) 
        
        # Weighted Combination
        cam = np.zeros(fmap.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * fmap[i]
            
        # ReLU (Keep only positive influence)
        cam = np.maximum(cam, 0)
        
        # Resize to match input MRI (e.g., 64x64x32)
        input_shape = input_tensor.shape[2:] 
        zoom_factors = [i / c for i, c in zip(input_shape, cam.shape)]
        cam = zoom(cam, zoom_factors, order=1)
        
        # Normalize strictly to 0-1
        if np.max(cam) > 0:
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
        
        return cam