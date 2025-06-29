import torch
import torch.nn as nn
import math
from transformers import PatchTSTConfig, PatchTSTForPrediction

class BitcoinPatchTST(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # PatchTST 설정
        patchtst_config = PatchTSTConfig(
            context_length=config['sequence_length'],
            prediction_length=config['prediction_length'],
            num_input_channels=config['num_features'],
            patch_length=config['patch_length'],
            patch_stride=config['patch_stride'],
            d_model=config['d_model'],
            num_attention_heads=config['num_heads'],
            num_hidden_layers=config['num_layers'],
            ffn_dim=config['ffn_dim'],
            dropout=config['dropout'],
            attention_dropout=config['attention_dropout'],
            pooling_type="mean",
            channel_attention=False,
            scaling="std"
        )
        
        # PatchTST 모델 초기화
        self.patchtst = PatchTSTForPrediction(patchtst_config)
    
    def forward(self, past_values):
        """
        Args:
            past_values: (batch_size, sequence_length, num_features)
        Returns:
            prediction: (batch_size, prediction_length)
        """
        # PatchTST는 (batch_size, num_channels, sequence_length) 형태를 기대
        past_values = past_values.transpose(1, 2)  # (batch_size, num_features, sequence_length)
        
        outputs = self.patchtst(past_values=past_values)
        
        # 예측 결과 추출 (첫 번째 채널만 사용 - Close 가격)
        prediction = outputs.prediction[:, 0, :]  # (batch_size, prediction_length)
        
        return prediction

class SimplePatchTST(nn.Module):
    """간단한 PatchTST 구현 (Hugging Face 의존성 없이)"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 패치 임베딩
        self.patch_embedding = nn.Linear(
            config['patch_length'] * config['num_features'], 
            config['d_model']
        )
        
        # 위치 인코딩
        self.pos_encoding = PositionalEncoding(config['d_model'], config['dropout'])
        
        # Transformer 인코더
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config['d_model'],
            nhead=config['num_heads'],
            dim_feedforward=config['ffn_dim'],
            dropout=config['dropout'],
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config['num_layers'])
        
        # 출력 레이어
        num_patches = (config['sequence_length'] - config['patch_length']) // config['patch_stride'] + 1
        self.output_projection = nn.Linear(config['d_model'] * num_patches, config['prediction_length'])
        
        self.patch_length = config['patch_length']
        self.patch_stride = config['patch_stride']
    
    def create_patches(self, x):
        """입력을 패치로 분할"""
        batch_size, seq_len, num_features = x.shape
        patches = []
        
        for i in range(0, seq_len - self.patch_length + 1, self.patch_stride):
            patch = x[:, i:i + self.patch_length, :].reshape(batch_size, -1)
            patches.append(patch)
        
        return torch.stack(patches, dim=1)  # (batch_size, num_patches, patch_length * num_features)
    
    def forward(self, x):
        # 패치 생성
        patches = self.create_patches(x)  # (batch_size, num_patches, patch_length * num_features)
        
        # 패치 임베딩
        embedded = self.patch_embedding(patches)  # (batch_size, num_patches, d_model)
        
        # 위치 인코딩
        embedded = self.pos_encoding(embedded)
        
        # Transformer
        encoded = self.transformer(embedded)  # (batch_size, num_patches, d_model)
        
        # 출력 생성
        flattened = encoded.reshape(encoded.size(0), -1)  # (batch_size, num_patches * d_model)
        output = self.output_projection(flattened)  # (batch_size, prediction_length)
        
        return output

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
