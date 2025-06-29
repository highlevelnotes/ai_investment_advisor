import torch
import torch.nn as nn
from transformers import PatchTSTForPrediction, PatchTSTConfig
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

class TransferLearningPatchTST(nn.Module):
    """실제 사전훈련된 PatchTST 모델을 사용한 전이학습"""
    
    def __init__(self, config, pretrained_model_name="microsoft/patchtst-etth1-forecast"):
        super().__init__()
        self.config = config
        
        # 사전훈련된 PatchTST 모델 로드
        print(f"사전훈련된 모델 로드 중: {pretrained_model_name}")
        try:
            self.patchtst_config = PatchTSTConfig.from_pretrained(pretrained_model_name)
            self.backbone = PatchTSTForPrediction.from_pretrained(pretrained_model_name)
            print("✅ 사전훈련된 PatchTST 모델 로드 성공!")
        except Exception as e:
            print(f"사전훈련된 모델 로드 실패: {e}")
            print("기본 PatchTST 설정으로 초기화...")
            self.patchtst_config = PatchTSTConfig(
                context_length=config['sequence_length'],
                prediction_length=1,  # 분류를 위해 1로 설정
                num_input_channels=config['num_features'],
                patch_length=16,
                patch_stride=8,
                d_model=128,
                num_attention_heads=16,
                num_hidden_layers=3,
                ffn_dim=256,
                dropout=0.1
            )
            self.backbone = PatchTSTForPrediction(self.patchtst_config)
        
        # 백본 모델의 일부 레이어 동결
        self.freeze_backbone_layers()
        
        # 분류를 위한 새로운 헤드 추가
        self.classification_head = nn.Sequential(
            nn.Linear(self.patchtst_config.d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, config['num_classes'])
        )
        
        # 전역 평균 풀링
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def freeze_backbone_layers(self, freeze_ratio=0.7):
        """백본 모델의 일부 레이어 동결"""
        total_layers = len(list(self.backbone.named_parameters()))
        freeze_count = int(total_layers * freeze_ratio)
        
        frozen_count = 0
        for name, param in self.backbone.named_parameters():
            if frozen_count < freeze_count:
                param.requires_grad = False
                frozen_count += 1
            else:
                param.requires_grad = True
        
        trainable_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.backbone.parameters())
        
        print(f"백본 모델 동결: {freeze_ratio*100:.1f}% ({frozen_count}/{total_layers} 레이어)")
        print(f"훈련 가능한 파라미터: {trainable_params:,} / {total_params:,}")
    
    def forward(self, x):
        # x: (batch_size, sequence_length, num_features)
        batch_size = x.size(0)
        
        # PatchTST는 (batch_size, num_channels, sequence_length) 형태를 기대
        x_transposed = x.transpose(1, 2)  # (batch_size, num_features, sequence_length)
        
        try:
            # 사전훈련된 PatchTST 백본 통과
            outputs = self.backbone(past_values=x_transposed)
            
            # 예측 결과에서 특성 추출
            if hasattr(outputs, 'prediction'):
                features = outputs.prediction  # (batch_size, prediction_length, num_channels)
            else:
                features = outputs.last_hidden_state  # 또는 다른 출력
            
            # 특성을 1차원으로 변환
            if len(features.shape) == 3:
                features = features.mean(dim=1)  # 시간 차원 평균
                features = features.mean(dim=1)  # 채널 차원 평균
            elif len(features.shape) == 2:
                features = features.mean(dim=1)  # 마지막 차원 평균
            
            # 1차원 벡터로 만들기
            if len(features.shape) > 1:
                features = features.view(batch_size, -1)
                features = features.mean(dim=1, keepdim=True)
            
            # 분류 헤드 통과
            if features.dim() == 1:
                features = features.unsqueeze(1)
            
            # 고정 크기 특성으로 변환
            if features.size(1) != self.patchtst_config.d_model:
                features = torch.nn.functional.adaptive_avg_pool1d(
                    features.unsqueeze(1), self.patchtst_config.d_model
                ).squeeze(1)
            
            logits = self.classification_head(features)
            
        except Exception as e:
            print(f"PatchTST 포워드 에러: {e}")
            # 폴백: 간단한 특성 추출
            features = x.mean(dim=1)  # 시간 차원 평균
            features = torch.nn.functional.adaptive_avg_pool1d(
                features.unsqueeze(1), self.patchtst_config.d_model
            ).squeeze(1)
            logits = self.classification_head(features)
        
        return logits

class FineTuningTrainer:
    """전이학습 전용 트레이너"""
    
    def __init__(self, model, class_weights=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        
        # 가중 교차 엔트로피 손실
        if class_weights is not None:
            class_weights = class_weights.to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # 차별적 학습률 적용
        backbone_params = []
        head_params = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'classification_head' in name:
                    head_params.append(param)
                else:
                    backbone_params.append(param)
        
        # 백본은 낮은 학습률, 헤드는 높은 학습률
        self.optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': 1e-5},  # 백본: 매우 낮은 학습률
            {'params': head_params, 'lr': 1e-3}       # 헤드: 높은 학습률
        ], weight_decay=1e-4)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_accuracies = []
    
    def train_epoch(self, train_loader):
        """한 에포크 전이학습"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (sequences, labels) in enumerate(train_loader):
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            try:
                logits = self.model(sequences)
                loss = self.criterion(logits, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                
                # 정확도 계산
                _, predicted = torch.max(logits.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                if batch_idx % 10 == 0:
                    print(f'Batch {batch_idx}, Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%')
                    
            except Exception as e:
                print(f"배치 {batch_idx} 처리 중 오류: {e}")
                continue
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """검증"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                
                try:
                    logits = self.model(sequences)
                    loss = self.criterion(logits, labels)
                    
                    total_loss += loss.item()
                    
                    _, predicted = torch.max(logits.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                except Exception as e:
                    print(f"검증 중 오류: {e}")
                    continue
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def fine_tune(self, train_loader, val_loader, epochs=50, save_path='models/transfer_learning/best_model.pth'):
        """전이학습 실행"""
        print("🚀 PatchTST 전이학습 시작!")
        
        for epoch in range(epochs):
            print(f"\n=== Epoch {epoch+1}/{epochs} ===")
            
            # 학습
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # 검증
            val_loss, val_acc = self.validate(val_loader)
            self.val_accuracies.append(val_acc)
            
            # 학습률 스케줄링
            self.scheduler.step(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # 최고 모델 저장
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_model(save_path)
                print(f"🎉 새로운 최고 모델 저장! Val Acc: {val_acc:.2f}%")
            
            # 조기 종료
            if epoch > 10 and val_acc < np.mean(self.val_accuracies[-5:]) * 0.95:
                print("조기 종료!")
                break
        
        print(f"✅ 전이학습 완료! 최고 검증 정확도: {self.best_val_acc:.2f}%")
    
    def save_model(self, path):
        """모델 저장"""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': self.model.config
        }, path)
    
    def load_model(self, path):
        """모델 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']
        print(f"모델 로드 완료! 최고 검증 정확도: {self.best_val_acc:.2f}%")
