"""
visualize.py - 完整的可视化分析脚本
生成5张图：
  图1: 混淆矩阵
  图2: 正确分类样本的Grad-CAM
  图3: 误分类样本的诊断热力图
  图4: 不同类别关注模式对比
  图5: 噪声鲁棒性分析
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无GUI环境也能保存图片
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.metrics import confusion_matrix

from model import CWRU_1D_CNN
from gradcam import GradCAM
from dataloader import get_dataloaders

# ==================== 全局配置 ====================
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = "./data"
MODEL_PATH = os.path.join(RESULTS_DIR, 'best_model.pth')
NUM_CLASSES = 4
SAMPLE_LEN = 1024
USE_CLASSES = 4

# 噪声鲁棒性实验的信噪比等级
NOISE_SNR_DB = [10, 5, 0]

# ==================== 工具函数 ====================
def add_gaussian_noise(signal, snr_db):
    """
    对信号添加指定信噪比的高斯白噪声
    snr_db: 信噪比，单位dB。值越小噪声越大
    """
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.randn(len(signal)) * np.sqrt(noise_power)
    return signal + noise


def generate_cam_for_sample(model, grad_cam, signal_tensor, target_class, device):
    """
    为单个样本生成Grad-CAM热力图
    返回: (上采样后的cam, 目标类别置信度, 全类别概率向量)
    """
    signal_tensor = signal_tensor.to(device)
    output = model(signal_tensor)
    probs = torch.softmax(output, dim=1)
    confidence = probs[0, target_class].item()

    cam = grad_cam.generate(signal_tensor, target_class=target_class)

    # 上采样到原始信号长度
    x_cam = np.linspace(0, 1, len(cam))
    x_sig = np.linspace(0, 1, SAMPLE_LEN)
    cam_up = interp1d(x_cam, cam, kind='linear', fill_value='extrapolate')(x_sig)

    return cam_up, confidence, probs[0].detach().cpu().numpy()


def plot_signal_with_cam(ax, signal, cam_up, title=''):
    """
    在给定的axes上绘制信号+热力图叠加
    双纵轴：左侧=振动幅值，右侧=Grad-CAM强度
    """
    ax.plot(signal, color='dodgerblue', linewidth=0.6, label='Vibration signal')
    ax.set_ylabel('Amplitude', fontsize=9)

    ax2 = ax.twinx()
    ax2.fill_between(np.arange(len(signal)), 0, cam_up,
                     color='tomato', alpha=0.35, label='Grad-CAM')
    ax2.set_ylim(0, 1.2)
    ax2.set_ylabel('Grad-CAM', fontsize=9)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=7)
    ax.set_title(title, fontsize=10)


# ==================== 加载模型和数据 ====================
print("加载数据和模型...")
_, _, test_loader, num_classes, class_names = get_dataloaders(
    data_dir=DATA_DIR,
    batch_size=1,
    use_classes=USE_CLASSES,
    sample_len=SAMPLE_LEN,
    overlap=512
)

model = CWRU_1D_CNN(num_classes).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

target_layer = model.features[8]  # 最后一个卷积层
grad_cam = GradCAM(model, target_layer)


# ==================== 收集所有测试集预测结果 ====================
print("收集测试集预测...")
all_inputs = []   # 保存每个样本的tensor
all_labels = []   # 真实标签
all_preds = []    # 预测标签
all_probs = []    # 预测概率

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)
        all_inputs.append(inputs.cpu())
        all_labels.append(labels.item())
        all_preds.append(preds.item())
        all_probs.append(probs.cpu().numpy()[0])

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)


# ========================================================
# 图1：混淆矩阵
# ========================================================
print("\n[1/5] 生成混淆矩阵...")
cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.set_title('Confusion Matrix', fontsize=13)
plt.colorbar(im, ax=ax)
tick_marks = np.arange(num_classes)
ax.set_xticks(tick_marks)
ax.set_xticklabels(class_names, rotation=45, fontsize=9)
ax.set_yticks(tick_marks)
ax.set_yticklabels(class_names, fontsize=9)
ax.set_ylabel('True Label', fontsize=11)
ax.set_xlabel('Predicted Label', fontsize=11)

# 在格子中写入数字
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig1_confusion_matrix.png'), dpi=300)
print("  -> fig1_confusion_matrix.png")
plt.close()


# ========================================================
# 图2：正确分类样本的Grad-CAM分析（每类2个样本）
# ========================================================
print("\n[2/5] 生成正确分类热力图...")
correct_indices = np.where(all_preds == all_labels)[0]

fig, axes = plt.subplots(num_classes, 2, figsize=(16, 3 * num_classes))
if num_classes == 1:
    axes = axes.reshape(1, -1)

for class_idx in range(num_classes):
    # 该类别中所有正确分类的样本索引
    class_correct = [i for i in correct_indices if all_labels[i] == class_idx]
    if not class_correct:
        for col in range(2):
            axes[class_idx, col].text(0.5, 0.5, 'No samples',
                                       transform=axes[class_idx, col].transAxes)
        continue

    # 按该类别的置信度排序，选最高的2个
    class_confidences = all_probs[class_correct, class_idx]
    top2_rank = np.argsort(class_confidences)[-2:][::-1]

    for col, rank in enumerate(top2_rank):
        sample_idx = class_correct[rank]
        signal = all_inputs[sample_idx].squeeze().numpy()  # [1024]
        signal_tensor = all_inputs[sample_idx]              # [1, 1, 1024]

        cam_up, confidence, prob_vec = generate_cam_for_sample(
            model, grad_cam, signal_tensor, class_idx, DEVICE
        )

        title = f'{class_names[class_idx]} (conf={confidence:.3f})'
        plot_signal_with_cam(axes[class_idx, col], signal, cam_up, title)

plt.suptitle('Grad-CAM on Correctly Classified Samples', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig2_correct_classification_cam.png'), dpi=300, bbox_inches='tight')
print("  -> fig2_correct_classification_cam.png")
plt.close()


# ========================================================
# 图3：误分类样本的Grad-CAM诊断分析
# 对比：模型"以为"的类别特征(红色) vs 真实类别特征(绿色)
# ========================================================
print("\n[3/5] 生成误分类诊断热力图...")
wrong_indices = np.where(all_preds != all_labels)[0]

if len(wrong_indices) > 0:
    # 按误分类的置信度排序——选最"自信"的错误最有分析价值
    wrong_confidences = all_probs[wrong_indices, all_preds[wrong_indices]]
    sorted_wrong = wrong_indices[np.argsort(wrong_confidences)[::-1]]

    n_show = min(4, len(sorted_wrong))
    fig, axes = plt.subplots(n_show, 1, figsize=(14, 3 * n_show))
    if n_show == 1:
        axes = [axes]

    for i, idx in enumerate(sorted_wrong[:n_show]):
        true_label = all_labels[idx]
        pred_label = all_preds[idx]
        signal = all_inputs[idx].squeeze().numpy()
        signal_tensor = all_inputs[idx]

        # 按预测类别生成热力图（看模型"以为"的特征在哪）
        cam_up_pred, conf_pred, _ = generate_cam_for_sample(
            model, grad_cam, signal_tensor, pred_label, DEVICE
        )
        # 按真实类别生成热力图（看真实特征应该在哪）
        cam_up_true, conf_true, _ = generate_cam_for_sample(
            model, grad_cam, signal_tensor, true_label, DEVICE
        )

        # 叠加绘制：红色=预测类别的CAM，绿色=真实类别的CAM
        ax = axes[i]
        ax.plot(signal, color='dodgerblue', linewidth=0.6)
        ax_twin = ax.twinx()
        ax_twin.fill_between(np.arange(len(signal)), 0, cam_up_pred,
                             color='tomato', alpha=0.35,
                             label=f'CAM for pred: {class_names[pred_label]}')
        ax_twin.fill_between(np.arange(len(signal)), 0, cam_up_true,
                             color='limegreen', alpha=0.25,
                             label=f'CAM for true: {class_names[true_label]}')
        ax_twin.set_ylim(0, 1.5)
        ax_twin.legend(loc='upper right', fontsize=7)
        ax.set_title(
            f'Misclassified: True={class_names[true_label]}, '
            f'Pred={class_names[pred_label]} '
            f'(pred_conf={conf_pred:.3f}, true_conf={conf_true:.3f})',
            fontsize=9
        )
        ax.set_ylabel('Amplitude', fontsize=8)

    plt.suptitle(
        'Grad-CAM on Misclassified Samples '
        '(Red=Predicted class, Green=True class)',
        fontsize=12, y=1.01
    )
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'fig3_misclassification_cam.png'), dpi=300, bbox_inches='tight')
    print("  -> fig3_misclassification_cam.png")
    plt.close()
else:
    print("  没有误分类样本！")
    print("  建议：减少训练epoch到10-15，或减少训练数据量，")
    print("        让模型准确率降到93%-97%，产生更多可分析的误分类案例。")


# ========================================================
# 图4：不同类别关注模式对比（每类1个最佳样本，纵向并排）
# ========================================================
print("\n[4/5] 生成多类对比图...")
fig, axes = plt.subplots(num_classes, 1, figsize=(14, 2.5 * num_classes))
if num_classes == 1:
    axes = [axes]

for class_idx in range(num_classes):
    ax = axes[class_idx]
    class_correct = [i for i in correct_indices if all_labels[i] == class_idx]
    if not class_correct:
        ax.text(0.5, 0.5, 'No samples', transform=ax.transAxes)
        continue

    # 选置信度最高的样本
    best_idx = class_correct[np.argmax(all_probs[class_correct, class_idx])]
    signal = all_inputs[best_idx].squeeze().numpy()
    signal_tensor = all_inputs[best_idx]

    cam_up, confidence, _ = generate_cam_for_sample(
        model, grad_cam, signal_tensor, class_idx, DEVICE
    )

    plot_signal_with_cam(
        ax, signal, cam_up,
        title=f'{class_names[class_idx]} (best conf={confidence:.3f})'
    )

plt.suptitle('Attention Pattern Comparison Across Fault Types', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig4_class_comparison.png'), dpi=300, bbox_inches='tight')
print("  -> fig4_class_comparison.png")
plt.close()


# ========================================================
# 图5：噪声鲁棒性分析
# 对同一信号添加不同强度噪声，观察热力图变化
# ========================================================
print("\n[5/5] 生成噪声鲁棒性分析...")
np.random.seed(42)

if len(correct_indices) > 0:
    # 选第一个类别的最高置信度样本作为baseline
    class0_correct = [i for i in correct_indices if all_labels[i] == 0]
    if not class0_correct:
        class0_correct = [correct_indices[0]]
    baseline_idx = class0_correct[np.argmax(
        all_probs[class0_correct, all_labels[class0_correct]]
    )]
    baseline_signal = all_inputs[baseline_idx].squeeze().numpy()
    baseline_label = all_labels[baseline_idx]

    n_noise_levels = len(NOISE_SNR_DB)
    fig, axes = plt.subplots(1 + n_noise_levels, 1,
                              figsize=(14, 3 * (1 + n_noise_levels)))

    # 原始无噪声
    signal_tensor = all_inputs[baseline_idx]
    cam_up, confidence, prob_vec = generate_cam_for_sample(
        model, grad_cam, signal_tensor, baseline_label, DEVICE
    )
    plot_signal_with_cam(
        axes[0], baseline_signal, cam_up,
        title=f'Original (no noise) - {class_names[baseline_label]}, '
              f'conf={confidence:.3f}'
    )

    # 不同SNR
    for i, snr in enumerate(NOISE_SNR_DB):
        noisy_signal = add_gaussian_noise(baseline_signal, snr)
        noisy_tensor = torch.FloatTensor(noisy_signal).unsqueeze(0).unsqueeze(0)

        cam_up, confidence, prob_vec = generate_cam_for_sample(
            model, grad_cam, noisy_tensor, baseline_label, DEVICE
        )
        pred_class = np.argmax(prob_vec)
        pred_name = class_names[pred_class]

        plot_signal_with_cam(
            axes[i + 1], noisy_signal, cam_up,
            title=f'SNR={snr}dB | Pred={pred_name}, '
                  f'True class conf={confidence:.3f}'
        )

    plt.suptitle(
        'Noise Robustness Analysis: Grad-CAM under Gaussian Noise',
        fontsize=13, y=1.01
    )
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'fig5_noise_robustness.png'), dpi=300, bbox_inches='tight')
    print("  -> fig5_noise_robustness.png")
    plt.close()

    # 额外：输出噪声下整体准确率统计表
    print("\n  噪声鲁棒性统计:")
    print(f"  {'SNR(dB)':<10} {'Accuracy':<10}")
    for snr in NOISE_SNR_DB + [None]:
        correct_count = 0
        total_count = 0
        # 抽样200个测试样本
        sample_range = min(200, len(all_inputs))
        for j in range(sample_range):
            sig = all_inputs[j].squeeze().numpy()
            true_lbl = all_labels[j]
            if snr is not None:
                sig = add_gaussian_noise(sig, snr)
            sig_tensor = torch.FloatTensor(sig).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                out = model(sig_tensor.to(DEVICE))
                pred = out.argmax(dim=1).item()
            correct_count += (pred == true_lbl)
            total_count += 1
        acc = correct_count / total_count
        label = f"{snr}" if snr is not None else "Clean"
        print(f"  {label:<10} {acc:<10.4f}")

else:
    print("  没有正确分类样本，跳过噪声实验。")

print("\n========== 全部可视化完成 ==========")
