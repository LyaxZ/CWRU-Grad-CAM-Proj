"""
dataloader.py - CWRU轴承数据集加载、预处理与DataLoader构建
支持4类（0.007英寸）与10类（全部故障直径）两种模式。
"""
import os
import glob
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ----------------- 配置参数 -----------------
SAMPLE_LEN = 1024       # 每个样本的窗口长度
OVERLAP = 512           # 重叠点数（滑动步长 = SAMPLE_LEN - OVERLAP）
RANDOM_SEED = 42
BATCH_SIZE = 64
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# 类别标签映射
LABEL_MAP_4 = {
    "Normal": 0, "IR007": 1, "OR007": 2, "B007": 3
}

LABEL_MAP_10 = {
    "Normal": 0, "IR007": 1, "IR014": 2, "IR021": 3,
    "OR007": 4, "OR014": 5, "OR021": 6,
    "B007": 7, "B014": 8, "B021": 9
}


def extract_signal(filepath):
    """从.mat文件中提取驱动端(DE)振动信号"""
    mat = sio.loadmat(filepath)
    keys = [k for k in mat.keys() if not k.startswith('__')]
    signal = None
    for key in keys:
        if 'DE' in key:
            signal = mat[key].flatten()
            break
    if signal is None:
        for key in keys:
            arr = mat[key]
            if isinstance(arr, np.ndarray) and arr.dtype == 'float64':
                signal = arr.flatten()
                break
    if signal is None:
        raise ValueError(f"无法从 {filepath} 中提取有效振动信号。")
    return signal


def segment_signal(signal, label, sample_len, overlap):
    """滑动窗口切分信号"""
    step = sample_len - overlap
    if step <= 0:
        raise ValueError("overlap 必须小于 sample_len")
    n_samples = max(0, (len(signal) - sample_len) // step + 1)
    if n_samples == 0:
        padded = np.zeros(sample_len)
        padded[:len(signal)] = signal
        return np.array([padded]), np.array([label])
    segments = []
    labels = []
    for i in range(n_samples):
        start = i * step
        end = start + sample_len
        segments.append(signal[start:end])
        labels.append(label)
    return np.array(segments), np.array(labels)


def build_dataset(data_dir, use_classes=4, sample_len=1024, overlap=512, save_file=None):
    """遍历data_dir下的所有.mat文件，生成全体样本和标签"""
    label_map = LABEL_MAP_4 if use_classes == 4 else LABEL_MAP_10

    file_list = glob.glob(os.path.join(data_dir, "*.mat"))
    if not file_list:
        raise FileNotFoundError(f"在 {data_dir} 下未找到任何 .mat 文件")

    X_all = []
    y_all = []
    for fpath in file_list:
        fname = os.path.basename(fpath).replace('.mat', '')
        label = None
        for key in label_map:
            if key in fname:
                label = label_map[key]
                break
        if label is None:
            print(f"跳过未匹配的文件: {fname}")
            continue
        try:
            raw = extract_signal(fpath)
        except Exception as e:
            print(f"读取文件失败 {fname}: {e}")
            continue
        segs, labs = segment_signal(raw, label, sample_len, overlap)
        X_all.append(segs)
        y_all.append(labs)

    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    print(f"数据集构建完成，共 {len(X)} 个样本，形状 {X.shape}，类别分布：")
    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  类别 {u}: {c} 个样本")

    if save_file is not None:
        np.savez_compressed(save_file, X=X, y=y)
        print(f"预处理数据已保存至 {save_file}")
    return X, y


class CWRUDataset(Dataset):
    """PyTorch Dataset，将一维信号转换为 (1, len) 的张量"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X).unsqueeze(1)  # (N, 1, L)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloaders(data_dir="./data", batch_size=64, use_classes=4,
                    sample_len=1024, overlap=512, random_seed=42,
                    train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
                    preprocessed_file=None):
    """
    主函数：返回 train_loader, val_loader, test_loader 以及类别信息。
    """
    if preprocessed_file is None:
        preprocessed_file = os.path.join(RESULTS_DIR, f"cwru_data_{use_classes}class.npz")

    if os.path.exists(preprocessed_file):
        print(f"加载预处理文件 {preprocessed_file} ...")
        data = np.load(preprocessed_file)
        X, y = data['X'], data['y']
        print(f"已加载 {len(X)} 个样本。")
    else:
        X, y = build_dataset(data_dir, use_classes=use_classes,
                             sample_len=sample_len, overlap=overlap,
                             save_file=preprocessed_file)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=random_seed, stratify=y)

    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_ratio_adjusted,
        random_state=random_seed,
        stratify=y_train_val)

    print(f"训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")

    train_dataset = CWRUDataset(X_train, y_train)
    val_dataset = CWRUDataset(X_val, y_val)
    test_dataset = CWRUDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    class_names_4 = ["Normal", "InnerRace", "OuterRace", "Ball"]
    class_names_10 = [
        "Normal", "IR007", "IR014", "IR021",
        "OR007", "OR014", "OR021",
        "B007", "B014", "B021"
    ]
    class_names = class_names_4 if use_classes == 4 else class_names_10
    num_classes = len(class_names)

    return train_loader, val_loader, test_loader, num_classes, class_names


if __name__ == "__main__":
    train_loader, val_loader, test_loader, num_classes, class_names = get_dataloaders(
        data_dir="./data",
        batch_size=64,
        use_classes=4,
        sample_len=1024,
        overlap=512
    )
    print(f"类别数: {num_classes}, 类别名: {class_names}")
    for x, y in train_loader:
        print(f"一个batch的输入形状: {x.shape}, 标签形状: {y.shape}")
        break
