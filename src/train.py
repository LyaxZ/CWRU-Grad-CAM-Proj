"""
train.py - 训练1D-CNN并输出完整评估指标
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from sklearn.metrics import confusion_matrix, classification_report

from dataloader import get_dataloaders
from model import CWRU_1D_CNN

# ==================== 超参数 ====================
BATCH_SIZE = 64
LR = 0.001
EPOCHS = 50 
PATIENCE = 7  
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== 加载数据 ====================
DATA_DIR = "./data"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

train_loader, val_loader, test_loader, num_classes, class_names = get_dataloaders(
    data_dir=DATA_DIR,
    batch_size=BATCH_SIZE,
    use_classes=4,
    sample_len=1024,
    overlap=512
)

# ==================== 模型、损失、优化器 ====================
model = CWRU_1D_CNN(num_classes).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3
)

# ==================== 训练循环（含早停） ====================
best_acc = 0.0
best_model_wts = copy.deepcopy(model.state_dict())
patience_counter = 0

for epoch in range(EPOCHS):
    # --- 训练 ---
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)

    train_loss /= len(train_loader.dataset)

    # --- 验证 ---
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total

    scheduler.step(val_acc)

    current_lr = optimizer.param_groups[0]['lr']
    print(f'Epoch {epoch+1:2d} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.6f}')

    if val_acc > best_acc:
        best_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, os.path.join(RESULTS_DIR, 'best_model.pth'))
        patience_counter = 0  # 重置计数器
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f'早停触发！最佳验证准确率: {best_acc:.4f}')
            break

print(f'\n训练完成，最佳验证准确率: {best_acc:.4f}')

# ==================== 测试集详细评估 ====================
model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, 'best_model.pth'), map_location=DEVICE))
model.eval()

all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)  # 【新增】
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())  # 【新增】

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)

test_acc = (all_preds == all_labels).mean()
print(f'\n测试集准确率: {test_acc:.4f}')

cm = confusion_matrix(all_labels, all_preds)
print(f'\n混淆矩阵:\n{cm}')

report = classification_report(all_labels, all_preds,
                               target_names=class_names,
                               digits=4)
print(f'\n分类报告:\n{report}')

np.savez(os.path.join(RESULTS_DIR, 'test_results.npz'),
         preds=all_preds,
         labels=all_labels,
         probs=all_probs,
         allow_pickle=True)
print('\n测试结果已保存至 test_results.npz')
