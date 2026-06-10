import torch
import numpy as np

class GradCAM1D:
    def __init__(self, model, target_layer):
        self.model = model
        self.feature_maps = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_fm)
        target_layer.register_full_backward_hook(self._save_grad)

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
        
        weights = self.gradients.abs().mean(dim=2)
        weights = weights / (weights.max() + 1e-10)
        cam = (weights.unsqueeze(2) * self.feature_maps).sum(dim=1)
        cam = torch.relu(cam).squeeze(0).cpu().numpy()
        
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-10)
        
        print(f"CAM range: {cam.min():.4f} ~ {cam.max():.4f}")
        if np.all(cam == 0):
            print("CAM is all zero! Check gradients or noise strength.")
            
        return cam