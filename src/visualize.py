import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from dataloader import get_dataloaders
from model import CWRU_1D_CNN
from gradcam import GradCAM1D
from xai_methods import OcclusionSensitivity1D, upsample_cam, compute_rf_center

# 配置参数
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAMPLE_LEN = 1024
NOISE_SNR_DB = [-5, -10, -15]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, '..', 'results')
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')

# 加载数据和模型
print("加载数据和模型...")
_, _, test_loader, num_classes, class_names = get_dataloaders(
    data_dir=DATA_DIR, results_dir=RESULTS_DIR, batch_size=1)

model = CWRU_1D_CNN(num_classes).to(DEVICE)
model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, 'best_model.pth'), map_location=DEVICE))
model.eval()

target_layer = model.features[2]
grad_cam = GradCAM1D(model, target_layer)

# ✅ 计算感受野参数
rf_offset, rf_stride = compute_rf_center(model, target_layer)
print(f"Receptive field: offset={rf_offset:.1f}, stride={rf_stride:.1f}")

occlusion = OcclusionSensitivity1D(model)

# 收集测试数据
all_inputs, all_labels, all_preds, all_probs = [], [], [], []
with torch.no_grad():
    for X, y in test_loader:
        X_dev, y_dev = X.to(DEVICE), y.to(DEVICE)
        out = model(X_dev)
        prob = torch.softmax(out, dim=1)
        _, pred = torch.max(out, 1)
        all_inputs.append(X)
        all_labels.append(y.item())
        all_preds.append(pred.item())
        all_probs.append(prob.cpu().numpy()[0])

all_labels, all_preds, all_probs = np.array(all_labels), np.array(all_preds), np.array(all_probs)
correct_idx = np.where(all_preds == all_labels)[0]
wrong_idx = np.where(all_preds != all_labels)[0]

def add_noise(signal, snr_db):
    power = np.mean(signal**2) / (10 ** (snr_db / 10))
    return signal + np.random.randn(len(signal)) * np.sqrt(power)

def plot_signal_cam(ax, signal, cam, title='', show_legend=True):
    ax.plot(signal, color='dodgerblue', linewidth=0.5, label='Signal')
    ax2 = ax.twinx()
    ax2.fill_between(np.arange(len(signal)), 0, cam, color='tomato', alpha=0.3, label='Grad-CAM')
    ax2.set_ylim(0, 1.2)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    if show_legend:
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=7)

# Fig 1: 混淆矩阵
print("[1/5] 混淆矩阵")
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap=plt.cm.Blues)
ax.set_xticks(np.arange(num_classes))
ax.set_xticklabels(class_names, rotation=45)
ax.set_yticks(np.arange(num_classes))
ax.set_yticklabels(class_names)
ax.set_ylabel('True')
ax.set_xlabel('Predicted')
for i in range(num_classes):
    for j in range(num_classes):
        ax.text(j, i, cm[i, j], ha="center", color="white" if cm[i, j] > cm.max()/2 else "black")
plt.colorbar(im)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig1_confusion_matrix.png'), dpi=300)
plt.close()

# Fig 2: 正确分类的CAM可视化
print("[2/5] 正确分类的CAM可视化")
fig, axes = plt.subplots(num_classes, 2, figsize=(14, 3*num_classes))
for cls in range(num_classes):
    cls_idx = [i for i in correct_idx if all_labels[i] == cls]
    if not cls_idx:
        continue
    top2 = np.argsort(all_probs[cls_idx, cls])[-2:][::-1]
    for col, rank in enumerate(top2):
        idx = cls_idx[rank]
        sig = all_inputs[idx].squeeze().numpy()
        cam = grad_cam.generate(all_inputs[idx].to(DEVICE), cls)
        show_legend = (cls == 0 and col == 0)
        plot_signal_cam(axes[cls, col], sig, upsample_cam(cam, SAMPLE_LEN, rf_offset, rf_stride), 
                        f'{class_names[cls]} (conf={all_probs[idx, cls]:.3f})', show_legend=show_legend)
plt.suptitle('Grad-CAM on Correctly Classified Samples', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig2_correct_cam.png'), dpi=300, bbox_inches='tight')
plt.close()

# Fig 3: 噪声鲁棒性分析
print("[3/5] 噪声鲁棒性分析")
np.random.seed(42)
cls_idx = [i for i in correct_idx if all_labels[i] == 2]
if not cls_idx:
    cls_idx = [correct_idx[0]]
base_idx = cls_idx[np.argmax(all_probs[cls_idx, all_labels[cls_idx]])]
base_sig = all_inputs[base_idx].squeeze().numpy()
base_cls = all_labels[base_idx]

fig, axes = plt.subplots(1+len(NOISE_SNR_DB), 2, figsize=(14, 3*(1+len(NOISE_SNR_DB))))
cam_clean = upsample_cam(grad_cam.generate(all_inputs[base_idx].to(DEVICE), base_cls), SAMPLE_LEN, rf_offset, rf_stride)
occ_clean = occlusion.generate(all_inputs[base_idx].to(DEVICE), base_cls)
plot_signal_cam(axes[0, 0], base_sig, cam_clean, 'Clean - Grad-CAM', show_legend=True)
plot_signal_cam(axes[0, 1], base_sig, occ_clean, 'Clean - Occlusion', show_legend=False)

for i, snr in enumerate(NOISE_SNR_DB):
    noisy_sig = add_noise(base_sig, snr)
    noisy_tensor = torch.FloatTensor(noisy_sig).unsqueeze(0).unsqueeze(0)
    cam_n = upsample_cam(grad_cam.generate(noisy_tensor.to(DEVICE), base_cls), SAMPLE_LEN, rf_offset, rf_stride)
    occ_n = occlusion.generate(noisy_tensor.to(DEVICE), base_cls)
    plot_signal_cam(axes[i+1, 0], noisy_sig, cam_n, f'SNR={snr}dB - Grad-CAM', show_legend=False)
    plot_signal_cam(axes[i+1, 1], noisy_sig, occ_n, f'SNR={snr}dB - Occlusion', show_legend=False)
plt.suptitle('Noise Robustness Analysis', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig3_noise_robustness.png'), dpi=300, bbox_inches='tight')
plt.close()

# Fig 4: Grad-CAM与Occlusion对比
print("[4/5] Grad-CAM与Occlusion对比")
fig, axes = plt.subplots(num_classes, 2, figsize=(12, 3 * num_classes))

if num_classes == 1:
    axes = axes.reshape(1, -1)

for cls in range(num_classes):
    cls_idx = [i for i in correct_idx if all_labels[i] == cls]
    if not cls_idx:
        continue
    idx = cls_idx[np.argmax(all_probs[cls_idx, cls])]
    sig = all_inputs[idx].squeeze().numpy()
    cam_gc = upsample_cam(grad_cam.generate(all_inputs[idx].to(DEVICE), cls), SAMPLE_LEN, rf_offset, rf_stride)
    cam_occ = occlusion.generate(all_inputs[idx].to(DEVICE), cls)

    plot_signal_cam(axes[cls, 0], sig, cam_gc, f'{class_names[cls]} - Grad-CAM', show_legend=(cls == 0))
    plot_signal_cam(axes[cls, 1], sig, cam_occ, f'{class_names[cls]} - Occlusion', show_legend=False)

    if cls == 0:
        axes[cls, 0].set_title('Grad-CAM', fontweight='bold', fontsize=11)
        axes[cls, 1].set_title('Occlusion', fontweight='bold', fontsize=11)

plt.suptitle('Grad-CAM vs Occlusion', y=1.01, fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig4_xai_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()


# Fig 5: 噪声诱导误差分析
print("[5/5] 噪声诱导误差分析")
np.random.seed(42)
wrong_idx_noisy = []
noisy_inputs_list = []
noisy_preds = []

for idx in range(len(all_inputs)):
    true_cls = all_labels[idx]
    clean_sig = all_inputs[idx].squeeze().numpy()
    noisy_sig = add_noise(clean_sig, snr_db=-15)
    noisy_tensor = torch.FloatTensor(noisy_sig).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        out = model(noisy_tensor.to(DEVICE))
        pred_cls = out.argmax(dim=1).item()
    if pred_cls != true_cls:
        wrong_idx_noisy.append(idx)
        noisy_inputs_list.append(noisy_tensor)
        noisy_preds.append(pred_cls)

if len(wrong_idx_noisy) > 0:
    n_show = min(4, len(wrong_idx_noisy))
    fig, axes = plt.subplots(n_show, 1, figsize=(14, 3*n_show))
    if n_show == 1:
        axes = [axes]
    for i in range(n_show):
        idx = wrong_idx_noisy[i]
        true_cls, pred_cls = all_labels[idx], noisy_preds[i]
        noisy_sig = noisy_inputs_list[i].squeeze().numpy()

        cam_pred = upsample_cam(grad_cam.generate(noisy_inputs_list[i].to(DEVICE), pred_cls), SAMPLE_LEN, rf_offset, rf_stride)
        cam_true = upsample_cam(grad_cam.generate(noisy_inputs_list[i].to(DEVICE), true_cls), SAMPLE_LEN, rf_offset, rf_stride)
        
        axes[i].plot(noisy_sig, color='dodgerblue', linewidth=0.5, label='Noisy Signal')
        ax2 = axes[i].twinx()
        ax2.fill_between(np.arange(SAMPLE_LEN), 0, cam_pred, color='red', alpha=0.4, label=f'Predicted: {class_names[pred_cls]}')
        ax2.fill_between(np.arange(SAMPLE_LEN), 0, cam_true, color='blue', alpha=0.3, label=f'True: {class_names[true_cls]}')
        ax2.set_ylim(0, 1.2)
        ax2.legend(loc='upper right', fontsize=7)
        axes[i].set_title(f'Noise-Induced Error (-15dB): True={class_names[true_cls]}, Pred={class_names[pred_cls]}', fontsize=9)
    plt.suptitle('Diagnosis of Noise-Induced Misclassification', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'fig5_misclassification.png'), dpi=300, bbox_inches='tight')
    plt.close()
else:
    print("模型鲁棒性极强！在-15dB噪声下未发现误分类样本。")

print("\n所有可视化完成。")