import torch
import torch.nn as nn
import math
import numpy as np
from transformers import PatchTSTConfig, PatchTSTForPrediction
import warnings
warnings.filterwarnings('ignore')

class BitcoinPatchTST(nn.Module):
    """올바른 PatchTST 사용법"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 올바른 PatchTST 설정
        self.patchtst_config = PatchTSTConfig(
            context_length=32,              # 시퀀스 길이 (올바른 설정)
            prediction_length=1,            # 예측 길이
            num_input_channels=1,           # 채널 수 (Close 가격만)
            patch_length=8,                 # 패치 길이
            patch_stride=4,                 # 패치 스트라이드
            d_model=128,
            num_attention_heads=4,
            num_hidden_layers=3,
            ffn_dim=256,
            dropout=0.1,
            head_dropout=0.1,
            pooling_type="mean",
            channel_attention=False,
            scaling="std"
        )
        
        try:
            self.backbone = PatchTSTForPrediction(self.patchtst_config)
            print(f"✅ PatchTST 올바른 설정:")
            print(f"   context_length: {self.patchtst_config.context_length}")
            print(f"   num_input_channels: {self.patchtst_config.num_input_channels}")
            
        except Exception as e:
            print(f"❌ PatchTST 초기화 실패: {e}")
            raise e
        
        # 분류 헤드 (prediction_length=1, num_channels=1이므로 출력은 (batch_size, 1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, config['num_classes'])
        )
    
    def forward(self, x):
        # x: (batch_size, sequence_length=32, num_features=24)
        batch_size, seq_len, num_features = x.shape
        
        # Close 가격만 추출
        close_prices = x[:, :, 3]  # (batch_size, 32)
        
        # PatchTST 형태로 변환: (batch_size, num_channels, sequence_length)
        close_prices = close_prices.unsqueeze(1)  # (batch_size, 1, 32)
        
        print(f"PatchTST 입력 shape: {close_prices.shape}")
        print(f"예상: (batch_size={batch_size}, channels=1, seq_len=32)")
        
        try:
            # PatchTST 통과
            outputs = self.backbone(past_values=close_prices)
            
            print(f"PatchTST 출력 타입: {type(outputs)}")
            
            if hasattr(outputs, 'prediction'):
                features = outputs.prediction  # (batch_size, prediction_length=1, num_channels=1)
                print(f"Prediction shape: {features.shape}")
                
                # (batch_size, 1) 형태로 변환
                features = features.squeeze(-1)  # 마지막 차원 제거: (batch_size, 1)
                
                if features.dim() == 1:
                    features = features.unsqueeze(1)  # (batch_size, 1)
                
                print(f"분류 입력 shape: {features.shape}")
                
            else:
                print("⚠️ prediction 속성 없음, 폴백 사용")
                features = close_prices.mean(dim=2)  # (batch_size, 1)
            
            # 분류 헤드 통과
            logits = self.classifier(features)
            print(f"최종 출력 shape: {logits.shape}")
            
            return logits
            
        except Exception as e:
            print(f"❌ PatchTST Forward 에러: {e}")
            print(f"입력 shape 상세: {close_prices.shape}")
            print(f"설정 상세: context_length={self.patchtst_config.context_length}")
            
            # 완전 폴백
            features = close_prices.mean(dim=2)  # (batch_size, 1)
            logits = self.classifier(features)
            return logits

class SimplePatchTSTFallback(nn.Module):
    """PatchTST 실패 시 사용할 간단한 대안"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 간단한 CNN + Transformer
        self.conv1d = nn.Conv1d(1, 64, kernel_size=8, stride=4, padding=2)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=64, nhead=4, dim_feedforward=128, 
                dropout=0.1, batch_first=True
            ),
            num_layers=2
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, config['num_classes'])
        )
    
    def forward(self, x):
        # x: (batch_size, 32, num_features)
        close_prices = x[:, :, 3]  # Close 가격
        close_prices = close_prices.unsqueeze(1)  # (batch_size, 1, 32)
        
        # CNN으로 패치 생성
        conv_out = self.conv1d(close_prices)  # (batch_size, 64, seq_len')
        conv_out = conv_out.transpose(1, 2)   # (batch_size, seq_len', 64)
        
        # Transformer
        transformer_out = self.transformer(conv_out)  # (batch_size, seq_len', 64)
        
        # 글로벌 평균 풀링
        pooled = transformer_out.mean(dim=1)  # (batch_size, 64)
        
        # 분류
        logits = self.classifier(pooled)
        return logits

class FixedPatchTSTClassifier(nn.Module):
    """수동 구현된 PatchTST (가장 안전한 대안)"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 패치 설정
        self.patch_length = 8
        self.patch_stride = 4
        self.d_model = 128
        
        # 패치 임베딩
        self.patch_embedding = nn.Linear(self.patch_length, self.d_model)
        
        # 위치 인코딩
        max_patches = (32 - self.patch_length) // self.patch_stride + 1
        self.register_buffer('pos_encoding', 
                           self._create_positional_encoding(max_patches, self.d_model))
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=8,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # 분류 헤드
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, config['num_classes'])
        )
        
        print(f"✅ 수동 PatchTST 초기화 완료")
        print(f"   패치 길이: {self.patch_length}")
        print(f"   패치 스트라이드: {self.patch_stride}")
        print(f"   최대 패치 수: {max_patches}")
    
    def _create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def create_patches(self, x):
        """패치 생성"""
        batch_size, seq_len = x.shape
        patches = []
        
        for i in range(0, seq_len - self.patch_length + 1, self.patch_stride):
            patch = x[:, i:i + self.patch_length]
            patches.append(patch)
        
        if len(patches) == 0:
            # 시퀀스가 너무 짧은 경우
            patch = torch.nn.functional.pad(x, (0, self.patch_length - seq_len))
            patches.append(patch[:, :self.patch_length])
        
        return torch.stack(patches, dim=1)
    
    def forward(self, x):
        # x: (batch_size, 32, num_features)
        batch_size, seq_len, num_features = x.shape
        
        # Close 가격만 사용
        close_prices = x[:, :, 3]  # (batch_size, 32)
        
        print(f"수동 PatchTST 입력: {close_prices.shape}")
        
        # 패치 생성
        patches = self.create_patches(close_prices)  # (batch_size, num_patches, patch_length)
        print(f"패치 shape: {patches.shape}")
        
        # 패치 임베딩
        embedded = self.patch_embedding(patches)  # (batch_size, num_patches, d_model)
        
        # 위치 인코딩
        num_patches = embedded.size(1)
        embedded = embedded + self.pos_encoding[:num_patches, :].unsqueeze(0)
        
        # Transformer
        encoded = self.transformer(embedded)  # (batch_size, num_patches, d_model)
        
        # 글로벌 평균 풀링
        pooled = encoded.mean(dim=1)  # (batch_size, d_model)
        
        # 분류
        logits = self.classifier(pooled)
        
        print(f"수동 PatchTST 출력: {logits.shape}")
        return logits
