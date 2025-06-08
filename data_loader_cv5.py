import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold


class GaitDataset(Dataset):
    """步态数据集类
    
    用于创建PyTorch数据集，处理步态数据
    
    Args:
        data (np.ndarray): 输入数据，形状为 (n_samples, window_size, n_features)
        labels (np.ndarray): 标签数据，形状为 (n_samples,)
    """
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 直接返回标量标签，不需要额外的维度
        return torch.FloatTensor(self.data[idx]), torch.LongTensor([self.labels[idx]]).squeeze(0)

def extract_subject_id(filename):
    """从文件名中提取被试ID
    
    Args:
        filename (str): 文件名
        
    Returns:
        str: 被试ID
    """
    return filename[:6]

def load_and_classify_data(data_dir):
    """加载并分类数据文件
    
    Args:
        data_dir (str): 数据目录路径
        
    Returns:
        dict: 包含分类后文件路径的字典
    """
    data_dict = {
        'ctrl': {'files': [], 'label': 0},  # 对照组
        'pd': {'files': [], 'label': 1}     # 患者组
    }
    
    for filename in os.listdir(data_dir):
        if not (filename.startswith(("Ga", "Ju", "Si")) and filename.endswith(".txt")):
            continue
            
        file_path = os.path.join(data_dir, filename)
        if "Co" in filename:
            data_dict['ctrl']['files'].append(file_path)
        elif "Pt" in filename:
            data_dict['pd']['files'].append(file_path)
    
    # 随机打乱文件列表
    np.random.seed(42)
    np.random.shuffle(data_dict['ctrl']['files'])
    np.random.shuffle(data_dict['pd']['files'])
    
    return data_dict

def create_windows(data, window_size, stride):
    """使用滑动窗口方法创建步态周期数据
    
    Args:
        data (np.ndarray): 原始数据
        window_size (int): 窗口大小
        stride (int): 步长
        
    Returns:
        np.ndarray: 窗口化后的数据
    """
    windows = []
    for i in range(0, len(data) - window_size + 1, stride):
        window = data[i:i + window_size]
        windows.append(window)
    return np.array(windows)

def load_subject_data(file_path, window_size, stride):
    """加载单个被试的数据并创建窗口
    
    Args:
        file_path (str): 数据文件路径
        window_size (int): 窗口大小
        stride (int): 步长
        
    Returns:
        np.ndarray: 窗口化后的数据，如果加载失败则返回None
        返回的形状 -----> [window_nums, window_size, 18]
    """
    try:
        data = pd.read_csv(file_path, sep='\t', header=None)
        # 只取传感器数据（第2-19列）
        sensor_data = data.iloc[:, 1:19].values
        
        # 处理异常值
        if not np.isfinite(sensor_data).all():
            non_finite_count = np.sum(~np.isfinite(sensor_data))
            non_finite_ratio = non_finite_count / sensor_data.size
            
            if non_finite_ratio > 0.1:  # 如果异常值超过10%
                print(f"非有限值比例过高，跳过该文件")
                return None
            else:
                # 使用前后值的均值替换异常值
                for col in range(sensor_data.shape[1]):
                    mask = ~np.isfinite(sensor_data[:, col])
                    if np.any(mask):
                        # 获取前后有效值的索引
                        valid_indices = np.where(np.isfinite(sensor_data[:, col]))[0]
                        if len(valid_indices) > 0:
                            # 使用最近的有效值替换
                            for idx in np.where(mask)[0]:
                                nearest_idx = valid_indices[np.argmin(np.abs(valid_indices - idx))]
                                sensor_data[idx, col] = sensor_data[nearest_idx, col]
                        else:
                            # 如果没有有效值，使用0替换
                            sensor_data[mask, col] = 0
        
        # 创建窗口数据
        windows = create_windows(sensor_data, window_size, stride)
        return windows
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {str(e)}")
        return None

def organize_data(data_dict, window_size, stride):
    """按被试ID组织数据
    
    Args:
        data_dict (dict): 分类后的数据字典
        window_size (int): 窗口大小
        stride (int): 步长
        
    Returns:
        tuple: (pd_data, pd_labels, hc_data, hc_labels)
    """
    pd_data = {}    # 存储PD患者数据
    pd_labels = {}  # 存储PD患者标签
    hc_data = {}    # 存储HC对照组数据
    hc_labels = {}  # 存储HC对照组标签
    
    # 处理对照组数据
    for file_path in data_dict['ctrl']['files']:
        subject_id = extract_subject_id(os.path.basename(file_path))
        windows = load_subject_data(file_path, window_size, stride)
        if windows is None:
            continue
            
        if subject_id not in hc_data:
            hc_data[subject_id] = []
            hc_labels[subject_id] = []
            
        hc_data[subject_id].extend(windows)
        hc_labels[subject_id].extend([data_dict['ctrl']['label']] * len(windows))
    
    # 处理患者组数据
    for file_path in data_dict['pd']['files']:
        subject_id = extract_subject_id(os.path.basename(file_path))
        windows = load_subject_data(file_path, window_size, stride)
        if windows is None:
            continue
            
        if subject_id not in pd_data:
            pd_data[subject_id] = []
            pd_labels[subject_id] = []
            
        pd_data[subject_id].extend(windows)
        pd_labels[subject_id].extend([data_dict['pd']['label']] * len(windows))
    
    # 转换为numpy数组
    for subject_id in hc_data:
        hc_data[subject_id] = np.array(hc_data[subject_id])
        hc_labels[subject_id] = np.array(hc_labels[subject_id])
    
    for subject_id in pd_data:
        pd_data[subject_id] = np.array(pd_data[subject_id])
        pd_labels[subject_id] = np.array(pd_labels[subject_id])
    
    return pd_data, pd_labels, hc_data, hc_labels

def collect_subject_data(subject_ids, pd_data, pd_labels, hc_data, hc_labels):
    """收集指定被试ID的数据
    
    Args:
        subject_ids (list): 被试ID列表
        pd_data (dict): PD患者数据
        pd_labels (dict): PD患者标签
        hc_data (dict): HC对照组数据
        hc_labels (dict): HC对照组标签
        
    Returns:
        tuple: (X, y)
    """
    X, y = [], []
    
    for subject_id in subject_ids:
        if subject_id in pd_data:
            X.extend(pd_data[subject_id])
            y.extend(pd_labels[subject_id])
        elif subject_id in hc_data:
            X.extend(hc_data[subject_id])
            y.extend(hc_labels[subject_id])
    
    return np.array(X), np.array(y)

def create_kfold_loaders(pd_data, pd_labels, hc_data, hc_labels, n_splits=5, batch_size=32, normalize=True):
    """创建基于患者ID的K折交叉验证数据加载器
    
    Args:
        pd_data (dict): PD患者数据
        pd_labels (dict): PD患者标签
        hc_data (dict): HC对照组数据
        hc_labels (dict): HC对照组标签
        n_splits (int): 交叉验证折数
        batch_size (int): 批次大小
        normalize (bool): 是否标准化数据
        
    Returns:
        list: 包含每折(train_loader, val_loader)的列表
    """
    # 获取所有被试ID
    pd_ids = list(pd_data.keys())
    hc_ids = list(hc_data.keys())
    
    # 创建KFold对象
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # 存储每折的数据加载器
    fold_loaders = []
    
    # 对PD和HC分别进行K折划分
    pd_folds = list(kf.split(pd_ids))
    hc_folds = list(kf.split(hc_ids))
    
    for fold_idx in range(n_splits):
        # 获取当前折的训练集和验证集被试ID
        pd_train_idx, pd_val_idx = pd_folds[fold_idx]
        hc_train_idx, hc_val_idx = hc_folds[fold_idx]
        
        train_pd_ids = [pd_ids[i] for i in pd_train_idx]
        val_pd_ids = [pd_ids[i] for i in pd_val_idx]
        train_hc_ids = [hc_ids[i] for i in hc_train_idx]
        val_hc_ids = [hc_ids[i] for i in hc_val_idx]
        
        # 收集训练集和验证集数据
        X_train, y_train = collect_subject_data(train_pd_ids + train_hc_ids, pd_data, pd_labels, hc_data, hc_labels)
        X_val, y_val = collect_subject_data(val_pd_ids + val_hc_ids, pd_data, pd_labels, hc_data, hc_labels)
        
        # 数据标准化
        if normalize:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
            X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        
        # 创建数据加载器
        train_dataset = GaitDataset(X_train, y_train)
        val_dataset = GaitDataset(X_val, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        fold_loaders.append((train_loader, val_loader))
        
        # 打印当前折的数据分布
        print(f"\n第{fold_idx + 1}折数据分布:")
        print(f"训练集: {len(X_train)}个样本, 验证集: {len(X_val)}个样本")
        print(f"训练集PD患者数: {len(train_pd_ids)}, HC对照组数: {len(train_hc_ids)}")
        print(f"验证集PD患者数: {len(val_pd_ids)}, HC对照组数: {len(val_hc_ids)}")
    
    return fold_loaders

def get_data_loaders(data_dir, batch_size=32, n_splits=5, window_size=100, stride=50, normalize=True):
    """获取数据加载器
    
    Args:
        data_dir (str): 数据目录路径
        batch_size (int): 批次大小
        n_splits (int): 交叉验证折数
        window_size (int): 窗口大小
        stride (int): 步长
        normalize (bool): 是否标准化数据
        
    Returns:
        list: 包含每折(train_loader, val_loader)的列表
    """
    # 加载并分类数据
    data_dict = load_and_classify_data(data_dir)
    
    # 组织数据
    pd_data, pd_labels, hc_data, hc_labels = organize_data(data_dict, window_size, stride)
    # pd_data = {
    # 'GaPt01': array([[...], [...], [...], ...]),  # 第一个PD患者的数据
    # 'GaPt02': array([[...], [...], [...], ...]),  # 第二个PD患者的数据
    # ...
    # }
    
    # 创建K折交叉验证数据加载器
    fold_loaders = create_kfold_loaders(pd_data, pd_labels, hc_data, hc_labels, 
                                      n_splits=n_splits, batch_size=batch_size, 
                                      normalize=normalize)
    
    return fold_loaders
