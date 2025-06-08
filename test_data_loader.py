import os
import numpy as np
import pandas as pd
from data_loader_cv5 import (
    extract_subject_id,
    load_and_classify_data,
    create_windows,
    load_subject_data,
    organize_data,
    collect_subject_data,
    create_kfold_loaders,
    get_data_loaders
)

def test_extract_subject_id():
    """测试从文件名中提取被试ID"""
    test_files = [
        "GaPt01_1.txt",
        "GaCo01_1.txt",
        "JuPt02_2.txt",
        "SiCo03_1.txt"
    ]
    print("\n=== 测试 extract_subject_id ===")
    for file in test_files:
        subject_id = extract_subject_id(file)
        print(f"文件名: {file} -> 被试ID: {subject_id}")

def test_load_and_classify_data(data_dir):
    """测试数据加载和分类"""
    print("\n=== 测试 load_and_classify_data ===")
    data_dict = load_and_classify_data(data_dir)
    
    print("\n对照组文件:")
    for file in data_dict['ctrl']['files'][:3]:  # 只显示前3个文件
        print(f"- {os.path.basename(file)}")
    print(f"对照组文件总数: {len(data_dict['ctrl']['files'])}")
    
    print("\n患者组文件:")
    for file in data_dict['pd']['files'][:3]:  # 只显示前3个文件
        print(f"- {os.path.basename(file)}")
    print(f"患者组文件总数: {len(data_dict['pd']['files'])}")
    
    return data_dict

def test_create_windows():
    """测试窗口创建"""
    print("\n=== 测试 create_windows ===")
    # 创建测试数据
    test_data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    window_size = 3
    stride = 2
    
    print(f"原始数据:\n{test_data}")
    print(f"\n参数设置:")
    print(f"窗口大小 (window_size): {window_size}")
    print(f"步长 (stride): {stride}")
    
    windows = create_windows(test_data, window_size, stride)
    print(f"\n原始数据形状: {test_data.shape}")
    print(f"窗口化后数据形状: {windows.shape}")
    print("\n窗口化后的数据:")
    for i, window in enumerate(windows):
        print(f"\n窗口 {i+1} (起始索引: {i*stride}):")
        print(window)
        print(f"窗口范围: 行 {i*stride} 到 {i*stride + window_size - 1}")

def test_load_subject_data(data_dir):
    """测试单个被试数据加载"""
    print("\n=== 测试 load_subject_data ===")
    # 获取第一个文件进行测试
    files = os.listdir(data_dir)
    test_file = next(f for f in files if f.endswith('.txt'))
    file_path = os.path.join(data_dir, test_file)
    
    window_size = 100
    stride = 50
    
    windows = load_subject_data(file_path, window_size, stride)
    if windows is not None:
        print(f"文件: {test_file}")
        print(f"窗口化后数据形状: {windows.shape}")
        print(f"第一个窗口数据示例:")
        print(windows[0][:5, :5])  # 只显示前5行5列
    else:
        print(f"无法加载文件: {test_file}")

def test_organize_data(data_dict):
    """测试数据组织"""
    print("\n=== 测试 organize_data ===")
    window_size = 100
    stride = 50
    
    pd_data, pd_labels, hc_data, hc_labels = organize_data(data_dict, window_size, stride)
    
    print("\nPD患者数据:")
    for subject_id in list(pd_data.keys())[:3]:  # 只显示前3个被试
        print(f"\n被试 {subject_id}:")
        print(f"数据形状: {pd_data[subject_id].shape}")
        print(f"标签形状: {pd_labels[subject_id].shape}")
        print(f"标签值: {np.unique(pd_labels[subject_id])}")
    
    print("\nHC对照组数据:")
    for subject_id in list(hc_data.keys())[:3]:  # 只显示前3个被试
        print(f"\n被试 {subject_id}:")
        print(f"数据形状: {hc_data[subject_id].shape}")
        print(f"标签形状: {hc_labels[subject_id].shape}")
        print(f"标签值: {np.unique(hc_labels[subject_id])}")
    
    return pd_data, pd_labels, hc_data, hc_labels

def test_collect_subject_data(pd_data, pd_labels, hc_data, hc_labels):
    """测试被试数据收集"""
    print("\n=== 测试 collect_subject_data ===")
    # 选择前两个PD患者和HC对照组进行测试
    test_pd_ids = list(pd_data.keys())[:2]
    test_hc_ids = list(hc_data.keys())[:2]
    
    X, y = collect_subject_data(test_pd_ids + test_hc_ids, pd_data, pd_labels, hc_data, hc_labels)
    
    print(f"收集的数据形状: {X.shape}")
    print(f"收集的标签形状: {y.shape}")
    print(f"标签值分布: {np.unique(y, return_counts=True)}")

def test_create_kfold_loaders(pd_data, pd_labels, hc_data, hc_labels):
    """测试K折交叉验证数据加载器创建"""
    print("\n=== 测试 create_kfold_loaders ===")
    n_splits = 5
    batch_size = 32
    
    fold_loaders = create_kfold_loaders(
        pd_data, pd_labels, hc_data, hc_labels,
        n_splits=n_splits,
        batch_size=batch_size,
        normalize=True
    )
    
    print(f"创建的折数: {len(fold_loaders)}")
    
    # 测试第一折的数据
    train_loader, val_loader = fold_loaders[0]
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    
    print("\n第一折数据示例:")
    print(f"训练集批次数据形状: {train_batch[0].shape}")
    print(f"训练集批次标签形状: {train_batch[1].shape}")
    print(f"验证集批次数据形状: {val_batch[0].shape}")
    print(f"验证集批次标签形状: {val_batch[1].shape}")

def main():
    """主函数"""
    data_dir = "data"  # 请根据实际数据目录修改
    
    # 运行所有测试
    test_extract_subject_id()
    data_dict = test_load_and_classify_data(data_dir)
    test_create_windows()
    test_load_subject_data(data_dir)
    pd_data, pd_labels, hc_data, hc_labels = test_organize_data(data_dict)
    test_collect_subject_data(pd_data, pd_labels, hc_data, hc_labels)
    test_create_kfold_loaders(pd_data, pd_labels, hc_data, hc_labels)

if __name__ == "__main__":
    main() 