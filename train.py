import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import matplotlib
import os
import numpy as np
from data_loader_cv5 import *
from CNN_GRU_KAN import CNN_GRU_KAN_Model
from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, patience=10):
    """训练模型
    
    Args:
        model: 模型实例
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        num_epochs: 训练轮数
        device: 训练设备
        patience: 早停耐心值
        
    Returns:
        dict: 包含训练历史的字典
    """
    best_val_acc = 0.0
    early_stopping_counter = 0
    best_epoch = 0

    # 记录训练过程
    history = {
        'train_losses': [], 'train_accs': [], 'train_precisions': [], 'train_recalls': [], 'train_f1s': [],
        'val_losses': [], 'val_accs': [], 'val_precisions': [], 'val_recalls': [], 'val_f1s': []
    }

    epoch_iterator = trange(num_epochs, desc="Epochs")

    for epoch in epoch_iterator:
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_preds, train_labels = [], []

        train_iterator = tqdm(train_loader, desc=f'训练 (Epoch {epoch + 1}/{num_epochs})', leave=False)
        
        for inputs, labels in train_iterator:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

            train_iterator.set_postfix({"loss": f"{loss.item():.4f}"})

        # 计算训练指标
        train_metrics = calculate_metrics(train_labels, train_preds)
        avg_train_loss = train_loss / len(train_loader)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []

        val_iterator = tqdm(val_loader, desc=f"验证 (Epoch {epoch + 1}/{num_epochs})", leave=False)
        
        with torch.no_grad():
            for inputs, labels in val_iterator:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

                val_iterator.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # 计算验证指标
        val_metrics = calculate_metrics(val_labels, val_preds)
        avg_val_loss = val_loss / len(val_loader)

        # 更新历史记录
        update_history(history, train_metrics, val_metrics, avg_train_loss, avg_val_loss)

        # 更新进度条
        epoch_iterator.set_postfix({
            "Train Loss": f"{avg_train_loss:.4f}",
            "Train Acc": f"{train_metrics['acc']:.4f}",
            "Val Loss": f"{avg_val_loss:.4f}",
            "Val Acc": f"{val_metrics['acc']:.4f}"
        })

        # 打印当前epoch的结果
        print(f"\nEpoch {epoch + 1}/{num_epochs}:")
        print(f"训练 - 损失: {avg_train_loss:.4f}, 准确率: {train_metrics['acc']:.4f}")
        print(f"验证 - 损失: {avg_val_loss:.4f}, 准确率: {val_metrics['acc']:.4f}")

        # 早停检查
        if val_metrics['acc'] > best_val_acc:
            best_val_acc = val_metrics['acc']
            best_epoch = epoch + 1
            early_stopping_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"新的最佳模型已保存！验证准确率: {val_metrics['acc']:.4f}")
        else:
            early_stopping_counter += 1
            print(f"验证准确率未提升, 早停计数: {early_stopping_counter}/{patience}")
            
            if early_stopping_counter >= patience:
                print(f"早停触发! 连续{patience}个epoch验证准确率未提升。")
                print(f"最佳模型来自epoch {best_epoch}，验证准确率: {best_val_acc:.4f}")
                break

    history['best_val_acc'] = best_val_acc
    history['best_epoch'] = best_epoch
    history['early_stopped'] = early_stopping_counter >= patience
    
    return history

def calculate_metrics(labels, preds):
    """计算评估指标"""
    return {
        'acc': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds),
        'recall': recall_score(labels, preds),
        'f1': f1_score(labels, preds)
    }

def update_history(history, train_metrics, val_metrics, train_loss, val_loss):
    """更新训练历史记录"""
    history['train_losses'].append(train_loss)
    history['train_accs'].append(train_metrics['acc'])
    history['train_precisions'].append(train_metrics['precision'])
    history['train_recalls'].append(train_metrics['recall'])
    history['train_f1s'].append(train_metrics['f1'])
    
    history['val_losses'].append(val_loss)
    history['val_accs'].append(val_metrics['acc'])
    history['val_precisions'].append(val_metrics['precision'])
    history['val_recalls'].append(val_metrics['recall'])
    history['val_f1s'].append(val_metrics['f1'])

def visualize_fold_results(history, fold_idx, save_dir):
    """可视化单折训练结果"""
    matplotlib.rcParams['font.family'] = 'SimHei'
    matplotlib.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(15, 10))
    
    # 1. 损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(history['train_losses'], label='训练损失', color='blue', alpha=0.7)
    plt.plot(history['val_losses'], label='验证损失', color='red', alpha=0.7)
    plt.title(f'第{fold_idx + 1}折 - 训练和验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 2. 准确率曲线
    plt.subplot(2, 2, 2)
    plt.plot(history['train_accs'], label='训练准确率', color='blue', alpha=0.7)
    plt.plot(history['val_accs'], label='验证准确率', color='red', alpha=0.7)
    plt.title(f'第{fold_idx + 1}折 - 训练和验证准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 3. 精确率和召回率曲线
    plt.subplot(2, 2, 3)
    plt.plot(history['train_precisions'], label='训练精确率', color='blue', alpha=0.7)
    plt.plot(history['val_precisions'], label='验证精确率', color='red', alpha=0.7)
    plt.plot(history['train_recalls'], label='训练召回率', color='green', alpha=0.7)
    plt.plot(history['val_recalls'], label='验证召回率', color='orange', alpha=0.7)
    plt.title(f'第{fold_idx + 1}折 - 训练和验证精确率/召回率')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 4. F1分数曲线
    plt.subplot(2, 2, 4)
    plt.plot(history['train_f1s'], label='训练F1分数', color='blue', alpha=0.7)
    plt.plot(history['val_f1s'], label='验证F1分数', color='red', alpha=0.7)
    plt.title(f'第{fold_idx + 1}折 - 训练和验证F1分数')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'fold_{fold_idx + 1}_results.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_cv_results(all_fold_results, save_dir):
    """可视化五折交叉验证结果"""
    matplotlib.rcParams['font.family'] = 'SimHei'
    matplotlib.rcParams['axes.unicode_minus'] = False

    # 1. 创建五折对比图
    plt.figure(figsize=(15, 10))

    # 1.1 所有折的验证准确率对比
    plt.subplot(2, 2, 1)
    for fold_result in all_fold_results:
        fold = fold_result['fold']
        val_accs = fold_result['history']['val_accs']
        plt.plot(val_accs, label=f'第{fold}折', alpha=0.7)
    plt.title('五折交叉验证 - 验证准确率对比')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 1.2 所有折的验证损失对比
    plt.subplot(2, 2, 2)
    for fold_result in all_fold_results:
        fold = fold_result['fold']
        val_losses = fold_result['history']['val_losses']
        plt.plot(val_losses, label=f'第{fold}折', alpha=0.7)
    plt.title('五折交叉验证 - 验证损失对比')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 1.3 所有折的最佳验证准确率对比
    plt.subplot(2, 2, 3)
    folds = [result['fold'] for result in all_fold_results]
    best_accs = [result['best_val_acc'] for result in all_fold_results]
    plt.bar(folds, best_accs, alpha=0.7)
    plt.title('五折交叉验证 - 最佳验证准确率')
    plt.xlabel('Fold')
    plt.ylabel('Best Validation Accuracy')
    plt.xticks(folds)
    plt.grid(True, linestyle='--', alpha=0.7)

    # 1.4 所有折的最佳验证准确率分布
    plt.subplot(2, 2, 4)
    plt.boxplot(best_accs)
    plt.title('五折交叉验证 - 最佳验证准确率分布')
    plt.ylabel('Best Validation Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'cv_results.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 创建五折平均值图
    plt.figure(figsize=(15, 10))

    # 计算每个epoch的平均值和标准差
    n_epochs = len(all_fold_results[0]['history']['val_accs'])
    avg_val_accs = []
    std_val_accs = []
    avg_val_losses = []
    std_val_losses = []
    avg_train_accs = []
    std_train_accs = []
    avg_train_losses = []
    std_train_losses = []

    for epoch in range(n_epochs):
        # 验证准确率
        epoch_val_accs = [fold['history']['val_accs'][epoch] for fold in all_fold_results]
        avg_val_accs.append(np.mean(epoch_val_accs))
        std_val_accs.append(np.std(epoch_val_accs))
        
        # 验证损失
        epoch_val_losses = [fold['history']['val_losses'][epoch] for fold in all_fold_results]
        avg_val_losses.append(np.mean(epoch_val_losses))
        std_val_losses.append(np.std(epoch_val_losses))
        
        # 训练准确率
        epoch_train_accs = [fold['history']['train_accs'][epoch] for fold in all_fold_results]
        avg_train_accs.append(np.mean(epoch_train_accs))
        std_train_accs.append(np.std(epoch_train_accs))
        
        # 训练损失
        epoch_train_losses = [fold['history']['train_losses'][epoch] for fold in all_fold_results]
        avg_train_losses.append(np.mean(epoch_train_losses))
        std_train_losses.append(np.std(epoch_train_losses))

    # 2.1 平均训练和验证准确率
    plt.subplot(2, 2, 1)
    epochs = range(1, n_epochs + 1)
    plt.plot(epochs, avg_train_accs, label='训练准确率', color='blue', alpha=0.7)
    plt.plot(epochs, avg_val_accs, label='验证准确率', color='red', alpha=0.7)
    plt.fill_between(epochs, 
                    np.array(avg_train_accs) - np.array(std_train_accs),
                    np.array(avg_train_accs) + np.array(std_train_accs),
                    color='blue', alpha=0.2)
    plt.fill_between(epochs,
                    np.array(avg_val_accs) - np.array(std_val_accs),
                    np.array(avg_val_accs) + np.array(std_val_accs),
                    color='red', alpha=0.2)
    plt.title('五折交叉验证 - 平均训练和验证准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 2.2 平均训练和验证损失
    plt.subplot(2, 2, 2)
    plt.plot(epochs, avg_train_losses, label='训练损失', color='blue', alpha=0.7)
    plt.plot(epochs, avg_val_losses, label='验证损失', color='red', alpha=0.7)
    plt.fill_between(epochs,
                    np.array(avg_train_losses) - np.array(std_train_losses),
                    np.array(avg_train_losses) + np.array(std_train_losses),
                    color='blue', alpha=0.2)
    plt.fill_between(epochs,
                    np.array(avg_val_losses) - np.array(std_val_losses),
                    np.array(avg_val_losses) + np.array(std_val_losses),
                    color='red', alpha=0.2)
    plt.title('五折交叉验证 - 平均训练和验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 2.3 最终性能指标对比
    plt.subplot(2, 2, 3)
    metrics = ['准确率', '精确率', '召回率', 'F1分数']
    avg_metrics = [
        np.mean([fold['history']['val_accs'][-1] for fold in all_fold_results]),
        np.mean([fold['history']['val_precisions'][-1] for fold in all_fold_results]),
        np.mean([fold['history']['val_recalls'][-1] for fold in all_fold_results]),
        np.mean([fold['history']['val_f1s'][-1] for fold in all_fold_results])
    ]
    std_metrics = [
        np.std([fold['history']['val_accs'][-1] for fold in all_fold_results]),
        np.std([fold['history']['val_precisions'][-1] for fold in all_fold_results]),
        np.std([fold['history']['val_recalls'][-1] for fold in all_fold_results]),
        np.std([fold['history']['val_f1s'][-1] for fold in all_fold_results])
    ]
    
    bars = plt.bar(metrics, avg_metrics, yerr=std_metrics, capsize=5, alpha=0.7)
    plt.title('五折交叉验证 - 最终性能指标')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    
    # 在柱状图上添加具体数值
    for bar, avg, std in zip(bars, avg_metrics, std_metrics):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{avg:.4f}\n±{std:.4f}',
                ha='center', va='bottom')

    # 2.4 最佳验证准确率统计
    plt.subplot(2, 2, 4)
    best_accs = [result['best_val_acc'] for result in all_fold_results]
    plt.boxplot(best_accs, labels=['最佳验证准确率'])
    plt.title('五折交叉验证 - 最佳验证准确率统计')
    plt.ylabel('Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加具体数值
    plt.text(1, np.mean(best_accs), 
            f'平均值: {np.mean(best_accs):.4f}\n标准差: {np.std(best_accs):.4f}',
            ha='center', va='center')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'cv_average_results.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 设置随机种子
    torch.manual_seed(35)
    np.random.seed(35)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建保存目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_dir, '../results')
    os.makedirs(save_dir, exist_ok=True)

    # 加载数据
    data_dir = 'data'
    print('数据加载中...')
    fold_loaders = get_data_loaders(
        data_dir=data_dir,
        batch_size=32,
        n_splits=5,
        window_size=100,
        stride=50,
        normalize=True
    )
    
    print(f"数据已加载 - 创建了{len(fold_loaders)}折交叉验证数据加载器")

    # 存储所有折的结果
    all_fold_results = []
    
    # 对每一折进行训练
    for fold, (train_loader, val_loader) in enumerate(fold_loaders):
        print(f"\n开始训练第{fold + 1}折...")
        print(f"训练集: {len(train_loader.dataset)}个样本, 验证集: {len(val_loader.dataset)}个样本")
        
        # 创建模型
        model = CNN_GRU_KAN_Model(
            input_channels=18,
            seq_len=100,
            num_classes=2
        ).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # 训练参数
        num_epochs = 30
        patience = 5
        
        # 训练当前折
        history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, patience)
        
        # 保存当前折的结果
        fold_result = {
            'fold': fold + 1,
            'history': history,
            'best_val_acc': history['best_val_acc'],
            'best_epoch': history['best_epoch']
        }
        all_fold_results.append(fold_result)
        breakpoint()
        # 保存当前折的最佳模型
        torch.save(model.state_dict(), f'best_model_fold_{fold + 1}.pth')
        
        # 保存当前折的详细结果
        fold_save_dir = os.path.join(save_dir, f'fold_{fold + 1}')
        os.makedirs(fold_save_dir, exist_ok=True)
        np.save(os.path.join(fold_save_dir, 'fold_history.npy'), fold_result)
        
        print(f"\n第{fold + 1}折训练完成")
        print(f"最佳验证准确率: {history['best_val_acc']:.4f}")
        print(f"最佳模型来自epoch {history['best_epoch']}")
        
        # 可视化当前折的训练结果
        visualize_fold_results(history, fold, save_dir)
    
    # 计算并打印所有折的平均结果
    avg_val_acc = np.mean([result['best_val_acc'] for result in all_fold_results])
    std_val_acc = np.std([result['best_val_acc'] for result in all_fold_results])
    print(f"\n五折交叉验证结果:")
    print(f"平均验证准确率: {avg_val_acc:.4f} ± {std_val_acc:.4f}")
    
    # 可视化所有折的结果
    visualize_cv_results(all_fold_results, save_dir)
    
    # 保存所有结果
    results = {
        'all_fold_results': all_fold_results,
        'avg_val_acc': avg_val_acc,
        'std_val_acc': std_val_acc,
        'final_metrics': {
            'accuracy': np.mean([fold['history']['val_accs'][-1] for fold in all_fold_results]),
            'precision': np.mean([fold['history']['val_precisions'][-1] for fold in all_fold_results]),
            'recall': np.mean([fold['history']['val_recalls'][-1] for fold in all_fold_results]),
            'f1': np.mean([fold['history']['val_f1s'][-1] for fold in all_fold_results])
        },
        'final_metrics_std': {
            'accuracy': np.std([fold['history']['val_accs'][-1] for fold in all_fold_results]),
            'precision': np.std([fold['history']['val_precisions'][-1] for fold in all_fold_results]),
            'recall': np.std([fold['history']['val_recalls'][-1] for fold in all_fold_results]),
            'f1': np.std([fold['history']['val_f1s'][-1] for fold in all_fold_results])
        }
    }
    
    np.save(os.path.join(save_dir, 'cv_results.npy'), results)
    print(f"\n所有结果已保存到：{os.path.join(save_dir, 'cv_results.npy')}")

if __name__ == '__main__':
    main()


