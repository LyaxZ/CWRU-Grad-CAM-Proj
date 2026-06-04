"""
gradcam.py - 1D Grad-CAM 实现
"""
import torch
import numpy as np


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None

        target_layer.register_forward_hook(self.save_feature_map)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_feature_map(self, module, input, output):
        self.feature_maps = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        # 【修改】同样加了 .detach()
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=None):
        """
        生成Grad-CAM热力图
        Args:
            input_tensor: shape [1, 1, signal_length]
            target_class: 目标类别，None则取预测类别
        Returns:
            cam: 归一化到[0,1]的一维numpy数组
        """
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # 权重：梯度在时间维上全局平均池化
        weights = self.gradients.mean(dim=2)  # [B, C]

        # 加权求和
        cam = (weights.unsqueeze(2) * self.feature_maps).sum(dim=1)  # [B, L]
        cam = torch.relu(cam)

        # 归一化到 [0, 1]
        cam = cam.squeeze(0).cpu().numpy()  # [L]

        # 全零保护
        #       cam.max()-cam.min()=0，除以0会产生NaN
        if cam.max() - cam.min() > 1e-8:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam
