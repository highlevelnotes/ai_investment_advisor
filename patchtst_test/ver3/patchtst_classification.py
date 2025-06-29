import torch
import torch.nn as nn
import math
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)

class PatchTSTClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 패치 설정
        self.patch_length = config['patch_length']
        self.patch_stride = config['patch_stride']
        self.num_features = config['num_features']
        self.num_classes = config['num_classes']  # 2 (상승/하락)
        
        # 패치 수 계산
        self.num_patches = (config['sequence_length'] - self.patch_length) // self.patch_stride + 1
        
        # 패치 임베딩
        self.patch_embedding = nn.Sequential(
            nn.Linear(self.patch_length * self.num_features, config['d_model']),
            nn.ReLU(),
            nn.Dropout(config['dropout'])
        )
        
        # 위치 인코딩
        self.pos_encoding = PositionalEncoding(config['d_model'], config['dropout'])
        
        # 클래스 토큰 (분류용)
        self.cls_token = nn.Parameter(torch.randn(1, 1, config['d_model']))
        
        # Transformer 인코더
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['d_model'],
            nhead=config['num_heads'],
            dim_feedforward=config['ffn_dim'],
            dropout=config['dropout'],
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config['num_layers'])
        
        # 분류 헤드
        self.classification_head = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model'] // 2),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['d_model'] // 2, self.num_classes)
        )
        
        # 가중치 초기화
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def create_patches(self, x):
        """입력을 패치로 분할"""
        batch_size, seq_len, num_features = x.shape
        patches = []
        
        for i in range(0, seq_len - self.patch_length + 1, self.patch_stride):
            patch = x[:, i:i + self.patch_length, :].reshape(batch_size, -1)
            patches.append(patch)
        
        return torch.stack(patches, dim=1)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # 패치 생성
        patches = self.create_patches(x)  # (batch_size, num_patches, patch_length * num_features)
        
        # 패치 임베딩
        embedded = self.patch_embedding(patches)  # (batch_size, num_patches, d_model)
        
        # 클래스 토큰 추가
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, d_model)
        embedded = torch.cat([cls_tokens, embedded], dim=1)  # (batch_size, num_patches+1, d_model)
        
        # 위치 인코딩
        embedded = self.pos_encoding(embedded)
        
        # Transformer
        encoded = self.transformer(embedded)  # (batch_size, num_patches+1, d_model)
        
        # 클래스 토큰의 출력 사용
        cls_output = encoded[:, 0, :]  # (batch_size, d_model)
        
        # 분류 결과
        logits = self.classification_head(cls_output)  # (batch_size, num_classes)
        
        return logits
