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
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        weights = self.gradients.mean(dim=2)
        cam = (weights.unsqueeze(2) * self.feature_maps).sum(dim=1)
        cam = torch.relu(cam).squeeze(0).cpu().numpy()
        
        if cam.max() - cam.min() > 1e-8:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)
        return cam