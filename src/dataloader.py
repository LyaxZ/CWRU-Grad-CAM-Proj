import os
import glob
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader

SAMPLE_LEN = 1024
OVERLAP = 512
RANDOM_SEED = 42

LABEL_MAP_10 = {
    "Normal": 0, 
    "IR007": 1, "IR014": 2, "IR021": 3,  # 内圈故障
    "OR007": 4, "OR014": 5, "OR021": 6,  # 外圈故障
    "B007": 7, "B014": 8, "B021": 9      # 滚动体故障
}

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

def build_dataset(data_dir, use_classes=10, sample_len=1024, overlap=512, save_file=None):

    label_map = LABEL_MAP_10
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
        
    if not X_all:
        raise FileNotFoundError(f"No valid .mat files found in {data_dir} for {use_classes}-class task.")
        
    X, y = np.concatenate(X_all), np.concatenate(y_all)
    if save_file:
        np.savez_compressed(save_file, X=X, y=y)
    return X, y

class CWRUDataset(Dataset):
    def __init__(self, X, y):
        X_normalized = np.zeros_like(X)
        for i in range(len(X)):
            mean = np.mean(X[i])
            std = np.std(X[i])
            X_normalized[i] = (X[i] - mean) / (std + 1e-8)
            
        self.X = torch.FloatTensor(X_normalized).unsqueeze(1)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataloaders(data_dir, results_dir, batch_size=64, use_classes=10, sample_len=1024, overlap=512, random_seed=42, train_ratio=0.3, val_ratio=0.3, test_ratio=0.4):

    preprocessed_file = os.path.join(results_dir, f"cwru_data_{use_classes}class_seq_split.npz")
    
    if os.path.exists(preprocessed_file):
        data = np.load(preprocessed_file)
        X, y = data['X'], data['y']
    else:
        os.makedirs(results_dir, exist_ok=True)
        X, y = build_dataset(data_dir, use_classes, sample_len, overlap, preprocessed_file)

    train_idx, val_idx, test_idx = [], [], []
    
    for label in range(use_classes):
        idx = np.where(y == label)[0]
        n_total = len(idx)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        np.random.seed(random_seed)
        np.random.shuffle(idx)
        
        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train:n_train+n_val])
        test_idx.extend(idx[n_train+n_val:])

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"Dataset sizes -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    train_loader = DataLoader(CWRUDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(CWRUDataset(X_val, y_val), batch_size=batch_size)
    test_loader = DataLoader(CWRUDataset(X_test, y_test), batch_size=batch_size)

    class_names = ["Normal", "IR007", "IR014", "IR021", "OR007", "OR014", "OR021", "B007", "B014", "B021"]
        
    return train_loader, val_loader, test_loader, use_classes, class_names