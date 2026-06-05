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

# ==================== Config ====================
BATCH_SIZE = 64
LR = 0.001
EPOCHS = 50
PATIENCE = 7
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, '..', 'results')
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==================== Data ====================
print("Loading data...")
train_loader, val_loader, test_loader, num_classes, class_names = get_dataloaders(
    data_dir=DATA_DIR, results_dir=RESULTS_DIR, batch_size=BATCH_SIZE)

# ==================== Train Function ====================
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
            optimizer