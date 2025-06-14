import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from data_laoder import separate_fold
from model import CNN_Three_Model
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random

def format_time(seconds):
    """将秒数转换为时分秒格式"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=7, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def augment_data(X, y, noise_level=0.01):
    """数据增强：添加高斯噪声
    
    Args:
        X: 输入数据
        y: 标签
        noise_level: 噪声水平
    
    Returns:
        增强后的数据
    """
    # 随机选择50%的样本进行增强
    mask = np.random.random(len(X)) < 0.5
    X_aug = X.copy()
    X_aug[mask] = X[mask] + np.random.normal(0, noise_level, X[mask].shape)
    return X_aug, y

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                num_epochs, device, early_stopping):
    """训练模型"""
    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rates': []
    }
    
    # 创建epoch进度条
    epoch_pbar = tqdm(range(num_epochs), desc='Training Progress', position=0)
    start_time = time.time()
    
    # 打印表头
    tqdm.write("\n" + "="*80)
    tqdm.write(f"{'Epoch':^6} | {'Train Loss':^12} | {'Train Acc':^12} | {'Val Loss':^12} | {'Val Acc':^12} | {'LR':^12}")
    tqdm.write("-"*80)
    
    for epoch in epoch_pbar:
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        batch_acc = 0.0
        
        # 创建训练批次进度条
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', 
                         leave=False, position=1)
        
        for inputs, labels in train_pbar:
            # 数据增强
            if random.random() < 0.5:  # 50%的概率进行数据增强
                inputs = inputs + torch.randn_like(inputs) * 0.01
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # L2正则化
            l2_lambda = 0.01
            l2_reg = torch.tensor(0.).to(device)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += l2_lambda * l2_reg
            
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            batch_acc = (preds == labels).float().mean().item()
            
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{batch_acc:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        train_loss = train_loss / len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        batch_val_acc = 0.0
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]', 
                       leave=False, position=1)
        
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                batch_val_acc = (preds == labels).float().mean().item()
                
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{batch_val_acc:.4f}'
                })
        
        val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        
        # 更新学习率
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 更新历史记录
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rates'].append(current_lr)
        
        # 计算预计剩余时间
        elapsed_time = time.time() - start_time
        avg_epoch_time = elapsed_time / (epoch + 1)
        remaining_time = avg_epoch_time * (num_epochs - epoch - 1)
        
        # 更新epoch进度条
        epoch_pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'train_acc': f'{train_acc:.4f}',
            'val_loss': f'{val_loss:.4f}',
            'val_acc': f'{val_acc:.4f}',
            'lr': f'{current_lr:.6f}',
            'ETA': format_time(remaining_time)
        })
        
        # 在终端打印当前epoch的结果
        tqdm.write(f"{epoch+1:^6d} | {train_loss:^12.4f} | {train_acc:^12.4f} | {val_loss:^12.4f} | {val_acc:^12.4f} | {current_lr:^12.6f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
            }, 'best_model.pth')
            tqdm.write(f'[Best Model] Validation accuracy improved to {best_val_acc:.4f}')
        
        # 早停检查
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            tqdm.write('\nEarly stopping triggered')
            break
    
    # 打印训练结束信息
    tqdm.write("="*80)
    tqdm.write(f"Training completed! Best validation accuracy: {best_val_acc:.4f}")
    tqdm.write("="*80 + "\n")
    
    epoch_pbar.close()
    return history, model

def plot_training_history(history):
    """绘制训练历史曲线"""
    plt.figure(figsize=(15, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制准确率曲线
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # 绘制学习率曲线
    plt.subplot(1, 3, 3)
    plt.plot(history['learning_rates'])
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 加载数据
    print('Loading data...')
    X_data = np.load("Xdata.npy")
    y_data = np.load("ydata.npy")
    data_person = np.load("data_person.npy")
    print(f'Data loaded: X_data shape: {X_data.shape}, y_data shape: {y_data.shape}')
    
    # 设置超参数
    batch_size = 64  # 增加批次大小
    num_epochs = 100  # 增加训练轮数
    learning_rate = 0.001
    num_folds = 5
    
    # 创建保存结果的目录
    os.makedirs('results', exist_ok=True)
    
    # 进行5折交叉验证
    all_fold_results = []
    
    # 创建折数进度条
    fold_pbar = tqdm(range(num_folds), desc='Cross Validation Progress', position=0)
    
    for fold in fold_pbar:
        fold_pbar.set_description(f'Fold {fold + 1}/{num_folds}')
        
        # 划分训练集和验证集
        X_train, y_train, X_val, y_val = separate_fold(X_data, y_data, data_person, fold, num_folds)
        
        # 转换为PyTorch张量
        X_train = torch.FloatTensor(X_train)
        y_train = torch.LongTensor(y_train.ravel())
        X_val = torch.FloatTensor(X_val)
        y_val = torch.LongTensor(y_val.ravel())
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # 初始化模型
        model = CNN_Three_Model(dropout_rate=0.5).to(device)
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        
        # 学习率调度器
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, 
                                    verbose=True, min_lr=1e-6)
        
        # 早停
        early_stopping = EarlyStopping(patience=10, verbose=True)
        
        # 训练模型
        history, best_model = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            num_epochs, device, early_stopping
        )
        
        # 绘制训练历史
        plot_training_history(history)
        
        # 在验证集上评估模型
        best_model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Evaluating', leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = best_model(inputs)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # 计算评估指标
        fold_results = {
            'accuracy': accuracy_score(val_labels, val_preds),
            'precision': precision_score(val_labels, val_preds, average='weighted'),
            'recall': recall_score(val_labels, val_preds, average='weighted'),
            'f1': f1_score(val_labels, val_preds, average='weighted')
        }
        
        all_fold_results.append(fold_results)
        
        # 更新折数进度条
        fold_pbar.set_postfix({
            'accuracy': f'{fold_results["accuracy"]:.4f}',
            'f1': f'{fold_results["f1"]:.4f}'
        })
        
        # 保存当前折的结果
        tqdm.write(f'\nFold {fold + 1} Results:')
        for metric, value in fold_results.items():
            tqdm.write(f'{metric.capitalize()}: {value:.4f}')
    
    # 清除折数进度条
    fold_pbar.close()
    
    # 计算并打印平均结果
    mean_results = {
        metric: np.mean([fold[metric] for fold in all_fold_results])
        for metric in all_fold_results[0].keys()
    }
    
    print('\nAverage Results across all folds:')
    for metric, value in mean_results.items():
        print(f'{metric.capitalize()}: {value:.4f}')
    
    # 保存平均结果
    np.save('results/mean_results.npy', mean_results)

if __name__ == '__main__':
    main()
