import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

from dataloader import get_dataloaders
from model import CWRU_1D_CNN
from baselines import SmallKernelCNN, DeepCNN

# 配置参数
BATCH_SIZE = 64
LR = 0.001
EPOCHS = 30
PATIENCE = 7
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, '..', 'results')
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
os.makedirs(RESULTS_DIR, exist_ok=True)

# 加载数据
print("\nLoading data...")
train_loader, val_loader, test_loader, num_classes, class_names = get_dataloaders(
    data_dir=DATA_DIR, results_dir=RESULTS_DIR, batch_size=BATCH_SIZE)

def train_model(model, name):
    print(f"\nTraining {name}...")
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss()
    
    best_acc, best_wts, patience_cnt = 0.0, copy.deepcopy(model.state_dict()), 0
    history = {'loss': [], 'acc': []}
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                _, preds = torch.max(model(X), 1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        val_acc = correct / total
        scheduler.step(val_acc)
        
        history['loss'].append(train_loss)
        history['acc'].append(val_acc)
        
        if val_acc > best_acc:
            best_acc, best_wts, patience_cnt = val_acc, copy.deepcopy(model.state_dict()), 0
            if name == '1D-CNN':
                torch.save(best_wts, os.path.join(RESULTS_DIR, 'best_model.pth'))
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                break
                
    print(f"{name} Best Val Acc: {best_acc:.4f}")
    model.load_state_dict(best_wts)
    return model, history

def evaluate_model(model, name):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            out = model(X)
            probs = torch.softmax(out, dim=1)
            _, preds = torch.max(out, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    preds, labels, probs = np.array(all_preds), np.array(all_labels), np.array(all_probs)
    acc = (preds == labels).mean()
    params = sum(p.numel() for p in model.parameters())
    
    if name == '1D-CNN':
        np.savez(os.path.join(RESULTS_DIR, 'test_results.npz'), preds=preds, labels=labels, probs=probs)
        print("\n1D-CNN Classification Report:\n", classification_report(labels, preds, target_names=class_names, digits=4))
    return acc, params

models_to_train = {
    '1D-CNN': CWRU_1D_CNN(num_classes),
    'Small-Kernel': SmallKernelCNN(num_classes),
    'Deep-CNN': DeepCNN(num_classes)
}

results, histories = {}, {}

for name, model in models_to_train.items():
    trained_model, hist = train_model(model, name)
    acc, params = evaluate_model(trained_model, name)
    results[name] = {'acc': acc, 'params': params}
    histories[name] = hist

print(f"\n{'Model':<15} {'Test Acc':<10} {'Params':<10}")
print("-" * 35)
for name, res in results.items():
    print(f"{name:<15} {res['acc']:<10.4f} {res['params']:<10,}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
for name, hist in histories.items():
    ax1.plot(hist['loss'], label=name)
    ax2.plot(hist['acc'], label=name)

ax1.set_title('Training Loss')
ax1.set_xlabel('Epoch')
ax1.legend()
ax2.set_title('Validation Accuracy')
ax2.set_xlabel('Epoch')
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'fig_training_curves.png'), dpi=300)
plt.close()
print("\nTraining curves saved.")