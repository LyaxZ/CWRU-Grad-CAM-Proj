import torch
import numpy as np
from scipy.ndimage import gaussian_filter1d

class GradCAM1D:
    def __init__(self, model, target_layer):
        self.model = model
        self.feature_maps = None
        self.gradients = None
        self.hooks = []
        self.hooks.append(target_layer.register_forward_hook(self._save_fm))
        self.hooks.append(target_layer.register_full_backward_hook(self._save_grad))

    def _save_fm(self, module, input, output):
        self.feature_maps = output.detach()

    def _save_grad(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].clone().detach()

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        input_tensor = input_tensor.clone().detach().requires_grad_(True)
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # 标准 Grad-CAM
        weights = self.gradients.mean(dim=2)
        cam = (weights.unsqueeze(2) * self.feature_maps).sum(dim=1)
        cam = torch.relu(cam).squeeze(0).cpu().numpy()

        cam = gaussian_filter1d(cam, sigma=1.0)

        if cam.max() > 0:
            vmax = np.percentile(cam, 99)
            cam = np.clip(cam, 0, vmax)
            cam = cam / (cam.max() + 1e-10)

        print(f"CAM range: {cam.min():.4f} ~ {cam.max():.4f}")
        if np.all(cam == 0):
            print("CAM is all zero! Check gradients or noise strength.")
        return cam

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
