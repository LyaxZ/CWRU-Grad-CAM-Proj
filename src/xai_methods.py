import torch.nn as nn
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import torch


class OcclusionSensitivity1D:
    def __init__(self, model, window_size=32, stride=8):
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
            mean_val = torch.tensor(np.mean(input_tensor[0, 0, :].cpu().numpy())).to(input_tensor.device)
            occluded[0, 0, start:start + self.window_size] = mean_val
            with torch.no_grad():
                prob = torch.softmax(self.model(occluded), dim=1)[0, target_class].item()
            drop = original_prob - prob
            sensitivity[start:start + self.window_size] += drop
            count[start:start + self.window_size] += 1

        count[count == 0] = 1
        sensitivity /= count
        sensitivity = np.maximum(sensitivity, 0)
        if sensitivity.max() > 0:
            sensitivity = sensitivity / sensitivity.max()
        return sensitivity


def compute_rf_center(model, target_layer):
    offset, stride = 0.0, 1.0
    for module in model.features:
        if module is target_layer:
            break
        if isinstance(module, nn.Conv1d):
            offset += (module.kernel_size[0] - 1) / 2.0 * stride - module.padding[0] * stride
            stride = stride * module.stride[0]
        elif isinstance(module, nn.MaxPool1d):
            offset += (module.kernel_size - 1) / 2.0 * stride
            stride = stride * module.stride

    return offset, stride

def upsample_cam(cam, target_length, rf_offset=0.0, rf_stride=1.0, smooth_sigma=2):

    n = len(cam)
    if n == target_length:
        return cam

    cam_positions = rf_offset + np.arange(n) * rf_stride
    x_target = np.arange(target_length, dtype=float)

    upsampled = np.interp(x_target, cam_positions, cam, left=0.0, right=0.0)

    if smooth_sigma > 0:
        upsampled = gaussian_filter1d(upsampled, sigma=smooth_sigma)
        if upsampled.max() > 0:
            upsampled = upsampled / upsampled.max()

    return upsampled
