import torch
import torch.nn as nn
import torch.nn.functional as F
from kan import KAN

class SELayer(nn.Module):
    """改进的Squeeze-and-Excitation注意力模块，添加dropout"""
    def __init__(self, channel, reduction=16, dropout_rate=0.1):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),  # 添加dropout
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class CNNBranch(nn.Module):
    """改进的CNN分支：增加正则化，减少过拟合"""
    def __init__(self, input_channels=18, dropout_rate=0.3):
        super(CNNBranch, self).__init__()
        self.dropout_rate = dropout_rate
        
        # 第一层卷积块
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout1d(dropout_rate * 0.5)  # 较轻的dropout
        
        # 第二层卷积块  
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout1d(dropout_rate * 0.7)
        
        # 第三层卷积块
        self.conv3 = nn.Conv1d(64, 96, kernel_size=3, padding=1)  # 减少通道数
        self.bn3 = nn.BatchNorm1d(96)
        self.dropout3 = nn.Dropout1d(dropout_rate)
        
        self.pool = nn.MaxPool1d(2)
        self.se = SELayer(96, reduction=16, dropout_rate=dropout_rate)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_channels)
        x = x.transpose(1, 2)  # (batch_size, input_channels, seq_len)
        
        # 第一层
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        if self.training:  # 只在训练时应用dropout
            x = self.dropout1(x)
        x = self.pool(x)
        
        # 第二层
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        if self.training:
            x = self.dropout2(x)
        x = self.pool(x)
        
        # 第三层
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        if self.training:
            x = self.dropout3(x)
        x = self.se(x)
        
        return x

class GRUBranch(nn.Module):
    """改进的GRU分支：添加dropout和层归一化"""
    def __init__(self, input_channels=18, hidden_size=96, dropout_rate=0.3):  # 减小hidden_size
        super(GRUBranch, self).__init__()
        
        # 输入投影层
        self.input_proj = nn.Linear(input_channels, hidden_size)
        self.input_ln = nn.LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(dropout_rate * 0.5)
        
        # GRU层
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate  # GRU内置dropout
        )
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),  # 添加层归一化
            nn.Tanh(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_channels)
        
        # 输入投影
        x = self.input_proj(x)
        x = self.input_ln(x)
        x = self.input_dropout(x)
        
        # GRU处理
        gru_out, _ = self.gru(x)  # (batch_size, seq_len, hidden_size*2)
        
        # 注意力机制
        attention_weights = self.attention(gru_out)  # (batch_size, seq_len, 1)
        context = torch.sum(attention_weights * gru_out, dim=1)  # (batch_size, hidden_size*2)
        
        # 输出投影
        output = self.output_proj(context)
        
        return output

class MultiBranchModel(nn.Module):
    """改进的多分支模型：增强正则化，防止过拟合"""
    def __init__(self, input_channels=18, seq_len=100, num_classes=2, dropout_rate=0.4):
        super(MultiBranchModel, self).__init__()
        
        # 分支模块
        self.cnn_branch = CNNBranch(input_channels, dropout_rate)
        self.gru_branch = GRUBranch(input_channels, hidden_size=96, dropout_rate=dropout_rate)
        
        # 计算融合后的特征维度
        cnn_output_size = 96 * (seq_len // 4)  # 96通道，经过两次池化
        gru_output_size = 96  # GRU分支输出96维
        total_features = cnn_output_size + gru_output_size
        
        # 自适应池化层，进一步减少特征维度
        self.adaptive_pool = nn.AdaptiveAvgPool1d(8)  # 将CNN特征池化到固定大小
        adaptive_cnn_size = 96 * 8
        
        # 特征融合层 - 多层渐进式降维
        self.fusion = nn.Sequential(
            # 第一层融合
            nn.Linear(adaptive_cnn_size + gru_output_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # 第二层融合
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.8),
            
            # 第三层融合
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.6)
        )
        
        # 使用更保守的KAN配置
        self.kan_classifier = KAN(
            layers_hidden=[64, 32, num_classes],  # 更小的网络
            grid_size=3,  # 减小grid_size
            spline_order=2,  # 减小spline_order  
            scale_noise=0.05,  # 减小噪声
            scale_base=0.8,
            scale_spline=0.8,
            base_activation=torch.nn.SiLU,
            grid_eps=0.01,
            grid_range=[-0.8, 0.8],  # 缩小范围
        )
        
        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier/He初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # CNN分支
        cnn_out = self.cnn_branch(x)
        cnn_out = self.adaptive_pool(cnn_out)  # 自适应池化
        cnn_out = cnn_out.view(cnn_out.size(0), -1)  # 展平
        
        # GRU分支
        gru_out = self.gru_branch(x)
        
        # 特征融合
        combined = torch.cat([cnn_out, gru_out], dim=1)
        fused = self.fusion(combined)
        
        # KAN分类
        out = self.kan_classifier(fused)
        
        return out

def CNN_GRU_KAN_Model(input_channels=18, seq_len=100, num_classes=2, dropout_rate=0.4):
    """获取改进的模型实例"""
    model = MultiBranchModel(
        input_channels=input_channels,
        seq_len=seq_len,
        num_classes=num_classes,
        dropout_rate=dropout_rate
    )
    return model