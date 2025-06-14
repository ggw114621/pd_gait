import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """位置编码层"""
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # 注册为缓冲区（不参与训练）
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1)]

class GaitTransformer(nn.Module):
    """步态数据Transformer模型
    
    用于步态数据分类的Transformer模型，包含：
    - 输入嵌入层
    - 位置编码
    - 多层Transformer编码器
    - 分类头
    
    Args:
        input_channels (int): 输入通道数，默认为18（传感器数量）
        seq_len (int): 输入序列长度，默认为100（窗口大小）
        d_model (int): Transformer模型维度，默认为128
        nhead (int): 注意力头数，默认为8
        num_layers (int): Transformer层数，默认为3
        num_classes (int): 输出类别数，默认为2
        dropout_rate (float): Dropout比率，默认为0.1
    """
    def __init__(self, input_channels=18, seq_len=100, d_model=128, nhead=8, 
                 num_layers=3, num_classes=2, dropout_rate=0.1):
        super(GaitTransformer, self).__init__()
        
        # 输入投影层
        self.input_proj = nn.Linear(input_channels, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, seq_len)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,  # 前馈网络维度
            dropout=dropout_rate,
            activation='gelu',  # 使用GELU激活函数
            batch_first=True  # 使用batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, input_channels]
        Returns:
            输出张量，形状为 [batch_size, num_classes]
        """
        # 输入投影
        x = self.input_proj(x)  # [batch_size, seq_len, d_model]
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # 创建注意力掩码（这里不需要掩码，因为处理的是完整序列）
        # 但为了符合Transformer的接口，创建一个全1掩码
        mask = None
        
        # Transformer编码器
        x = self.transformer_encoder(x, mask)  # [batch_size, seq_len, d_model]
        
        # 使用序列的平均值作为特征
        x = x.mean(dim=1)  # [batch_size, d_model]
        
        # 分类
        x = self.classifier(x)  # [batch_size, num_classes]
        
        return x

# 为了保持兼容性，保留原来的CNN模型类名
CNN_Three_Model = GaitTransformer

