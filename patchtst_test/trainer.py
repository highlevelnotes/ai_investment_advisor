import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os

class PatchTSTTrainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def train_epoch(self, train_loader):
        """한 에포크 학습"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch_sequences, batch_targets in progress_bar:
            batch_sequences = batch_sequences.to(self.device)
            batch_targets = batch_targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(batch_sequences)
            
            # 타겟 차원 맞추기
            if len(batch_targets.shape) > len(predictions.shape):
                batch_targets = batch_targets.squeeze(-1)
            
            loss = self.criterion(predictions, batch_targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({'Loss': f'{loss.item():.6f}'})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, val_loader):
        """검증"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_sequences, batch_targets in val_loader:
                batch_sequences = batch_sequences.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                predictions = self.model(batch_sequences)
                
                # 타겟 차원 맞추기
                if len(batch_targets.shape) > len(predictions.shape):
                    batch_targets = batch_targets.squeeze(-1)
                
                loss = self.criterion(predictions, batch_targets)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self, train_loader, val_loader, epochs=100, save_path='outputs/models/best_model.pth'):
        """전체 학습 과정"""
        print(f"학습 시작 - Device: {self.device}")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # 학습
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # 검증
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # 학습률 스케줄링
            self.scheduler.step(val_loss)
            
            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # 최고 모델 저장
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model(save_path)
                print(f"새로운 최고 모델 저장! Val Loss: {val_loss:.6f}")
            
            # 조기 종료 (선택사항)
            if epoch > 20 and val_loss > np.mean(self.val_losses[-10:]) * 1.1:
                print("조기 종료!")
                break
        
        print(f"학습 완료! 최고 검증 손실: {self.best_val_loss:.6f}")
    
    def save_model(self, path):
        """모델 저장"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, path)
    
    def load_model(self, path):
        """모델 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        print(f"모델 로드 완료! 최고 검증 손실: {self.best_val_loss:.6f}")
