import os
import glob
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

SAMPLE_LEN = 1024
OVERLAP = 512
RANDOM_SEED = 42

LABEL_MAP_4 = {"Normal": 0, "IR007": 1, "OR007": 2, "B007": 3}

def extract_signal(filepath):
    mat = sio.loadmat(filepath)
    for k in mat.keys():
        if 'DE' in k:
            return mat[k].flatten()
    raise ValueError(f"Cannot extract DE signal from {filepath}")

def segment_signal(signal, label, sample_len, overlap):
    step = sample_len - overlap
    sample_num = max(0, (len(signal) - sample_len) // step + 1)
    
    segments, labels = [], []
    for i in range(sample_num):
        start = i * step
        segments.append(signal[start:start+sample_len])
        labels.append(label)
    
    return np.array(segments), np.array(labels)

def build_dataset(data_dir, use_classes=4, sample_len=1024, overlap=512, save_file=None):
    label_map = LABEL_MAP_4
    X_all, y_all = [], []
    
    for fpath in glob.glob(os.path.join(data_dir, "*.mat")):
        fname = os.path.basename(fpath).replace('.mat', '')
        
        label = next((label_map[k] for k in label_map if k in fname), None)
        if label is None:
            continue
        
        raw = extract_signal(fpath)
        segs, labs = segment_signal(raw, label, sample_len, overlap)
        
        X_all.append(segs)
        y_all.append(labs)
    
    X, y = np.concatenate(X_all), np.concatenate(y_all)
    if save_file:
        np.savez_compressed(save_file, X=X, y=y)
    return X, y

class CWRUDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X).unsqueeze(1)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataloaders(data_dir, results_dir, batch_size=64, use_classes=4,
                    sample_len=1024, overlap=512, random_seed=42,
                    train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    preprocessed_file = os.path.join(results_dir, f"cwru_data_{use_classes}class.npz")
    
    if os.path.exists(preprocessed_file):
        data = np.load(preprocessed_file)
        X, y = data['X'], data['y']
    else:
        X, y = build_dataset(data_dir, use_classes, sample_len, overlap, preprocessed_file)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=random_seed, stratify=y)
    
    val_ratio_adj = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio_adj, 
        random_state=random_seed, stratify=y_train_val)

    train_loader = DataLoader(CWRUDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(CWRUDataset(X_val, y_val), batch_size=batch_size)
    test_loader = DataLoader(CWRUDataset(X_test, y_test), batch_size=batch_size)
    
    class_names = ["Normal", "InnerRace", "OuterRace", "Ball"]
    
    return train_loader, val_loader, test_loader, use_classes, class_names