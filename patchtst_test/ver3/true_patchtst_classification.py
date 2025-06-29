import torch
import torch.nn as nn
import math
import numpy as np

class InstanceNormalization:
    """PatchTST의 핵심: 각 시계열 인스턴스별 정규화"""
    def __init__(self):
        self.means = None
        self.stds = None
    
    def fit_transform(self, data):
        # 각 시계열 인스턴스별로 정규화 (axis=1은 시간축)
        self.means = data.mean(axis=1, keepdims=True)
        self.stds = data.std(axis=1, keepdims=True)
        return (data - self.means) / (self.stds + 1e-8)
    
    def transform(self, data):
        return (data - self.means) / (self.stds + 1e-8)
    
    def inverse_transform(self, data):
        return data * self.stds + self.means

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        return x + self.pe[:x.size(1), :].unsqueeze(0)

class TruePatchTSTClassification(nn.Module):
    """진짜 PatchTST 구현 - 논문 기준"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 논문 권장 설정
        self.patch_length = 16  # 논문 권장값
        self.patch_stride = 8   # 논문 권장값
        self.sequence_length = config['sequence_length']
        self.num_channels = config['num_features']
        self.d_model = config['d_model']
        self.num_classes = config['num_classes']
        
        # 패치 수 계산
        self.num_patches = (self.sequence_length - self.patch_length) // self.patch_stride + 1
        
        # Instance Normalization
        self.instance_norm = InstanceNormalization()
        
        # 패치 임베딩 (각 채널별로 독립적)
        self.patch_embedding = nn.Linear(self.patch_length, self.d_model)
        
        # 위치 인코딩
        self.pos_encoding = PositionalEncoding(self.d_model, max_len=self.num_patches)
        
        # Transformer 인코더
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config['num_heads'],
            dim_feedforward=config['ffn_dim'],
            dropout=config['dropout'],
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config['num_layers'])
        
        # 분류 헤드 (Close 채널용)
        self.classification_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            nn.Linear(self.d_model // 2, self.num_classes)
        )
        
        # 가중치 초기화
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def create_patches(self, x):
        """패치 생성 - 논문 방식"""
        # x: (batch_size, seq_len)
        batch_size, seq_len = x.shape
        patches = []
        
        for i in range(0, seq_len - self.patch_length + 1, self.patch_stride):
            patch = x[:, i:i + self.patch_length]  # (batch_size, patch_length)
            patches.append(patch)
        
        if len(patches) == 0:
            # 시퀀스가 너무 짧은 경우 전체를 하나의 패치로
            patches.append(x[:, :self.patch_length])
        
        return torch.stack(patches, dim=1)  # (batch_size, num_patches, patch_length)
    
    def forward(self, x):
        # x: (batch_size, seq_len, num_channels)
        batch_size, seq_len, num_channels = x.shape
        
        # Instance Normalization (각 시계열별로)
        x_normalized = torch.zeros_like(x)
        for i in range(batch_size):
            instance_data = x[i].cpu().numpy()  # (seq_len, num_channels)
            normalized_instance = self.instance_norm.fit_transform(instance_data.T).T  # 채널별 정규화
            x_normalized[i] = torch.FloatTensor(normalized_instance).to(x.device)
        
        # Close 채널만 사용 (분류용)
        close_channel_idx = min(3, num_channels - 1)  # Close는 보통 3번 인덱스
        close_data = x_normalized[:, :, close_channel_idx]  # (batch_size, seq_len)
        
        # 패치 생성
        patches = self.create_patches(close_data)  # (batch_size, num_patches, patch_length)
        
        # 패치 임베딩
        embedded = self.patch_embedding(patches)  # (batch_size, num_patches, d_model)
        
        # 위치 인코딩
        embedded = self.pos_encoding(embedded)
        
        # Transformer
        encoded = self.transformer(embedded)  # (batch_size, num_patches, d_model)
        
        # 글로벌 평균 풀링
        pooled = encoded.mean(dim=1)  # (batch_size, d_model)
        
        # 분류 결과
        logits = self.classification_head(pooled)  # (batch_size, num_classes)
        
        return logits
