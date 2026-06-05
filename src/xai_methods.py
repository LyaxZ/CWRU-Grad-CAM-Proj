import torch
import numpy as np
from scipy.interpolate import interp1d

class OcclusionSensitivity1D:
    def __init__(self, model, window_size=64, stride=16):
        self.model = model
        self.window_size = window_size
        self.stride = stride

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        signal_len = input_tensor.shape[2]
        
        with torch.no_grad():
            probs = torch.softmax(self.model(input_tensor), dim=1)
            if target_class is None:
                target_class = probs.argmax(dim=1).item()
            original_prob = probs[0, target_class].item()
        
        sensitivity = np.zeros(signal_len)
        count = np.zeros(signal_len)
        
        for start in range(0, signal_len - self.window_size + 1, self.stride):
            occluded = input_tensor.clone()
            occluded[0, 0, start:start+self.window_size] = 0.0
            
            with torch.no_grad():
                prob = torch.softmax(self.model(occluded), dim=1)[0, target_class].item()
            
            drop = original_prob - prob
            sensitivity[start:start+self.window_size] += drop
            count[start:start+self.window_size] += 1
            
        count[count == 0] = 1
        sensitivity /= count
        
        if sensitivity.max() - sensitivity.min() > 1e-8:
            sensitivity = (sensitivity - sensitivity.min()) / (sensitivity.max() - sensitivity.min())
        else:
            sensitivity = np.zeros_like(sensitivity)
        return sensitivity

def upsample_cam(cam, target_length):
    if len(cam) == target_length:
        return cam
    x_cam = np.linspace(0, 1, len(cam))
    x_target = np.linspace(0, 1, target_length)
    return interp1d(x_cam, cam, kind='linear', fill_value='extrapolate')(x_target)
