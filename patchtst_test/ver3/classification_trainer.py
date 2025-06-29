import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import os
import json

class ClassificationTrainer:
    def __init__(self, model, class_weights=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        
        # 가중 교차 엔트로피 손실 (불균형 데이터 처리)
        if class_weights is not None:
            class_weights = class_weights.to(device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=10, verbose=True
        )
        
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_acc = 0.0
    
    def train_epoch(self, train_loader):
        """한 에포크 학습"""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch_sequences, batch_labels in progress_bar:
            batch_sequences = batch_sequences.to(self.device)
            batch_labels = batch_labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(batch_sequences)
            loss = self.criterion(logits, batch_labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 예측 및 실제 레이블 저장
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            
            # 실시간 정확도 계산
            batch_acc = accuracy_score(batch_labels.cpu().numpy(), predictions.cpu().numpy())
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{batch_acc:.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(all_labels, all_predictions)
        
        return avg_loss, train_acc
    
    def validate(self, val_loader):
        """검증"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch_sequences, batch_labels in val_loader:
                batch_sequences = batch_sequences.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                logits = self.model(batch_sequences)
                loss = self.criterion(logits, batch_labels)
                total_loss += loss.item()
                
                # 예측 및 확률
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        
        # 상세 메트릭 계산
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        # ROC AUC (이진 분류)
        if len(np.unique(all_labels)) == 2:
            auc = roc_auc_score(all_labels, np.array(all_probabilities)[:, 1])
        else:
            auc = 0.0
        
        return avg_loss, accuracy, precision, recall, f1, auc
    
    def train(self, train_loader, val_loader, epochs=100, save_path='models/classification/best_model.pth'):
        """전체 학습 과정"""
        print(f"분류 모델 학습 시작 - Device: {self.device}")
        print(f"모델 파라미터 수: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # 학습
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # 검증
            val_loss, val_acc, val_precision, val_recall, val_f1, val_auc = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # 학습률 스케줄링 (정확도 기준)
            self.scheduler.step(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}")
            print(f"Val F1: {val_f1:.4f}, Val AUC: {val_auc:.4f}")
            
            # 최고 모델 저장 (정확도 기준)
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_model(save_path)
                print(f"새로운 최고 모델 저장! Val Acc: {val_acc:.4f}")
            
            # 조기 종료
            if epoch > 20 and val_acc < np.mean(self.val_accuracies[-10:]) * 0.95:
                print("조기 종료!")
                break
        
        print(f"학습 완료! 최고 검증 정확도: {self.best_val_acc:.4f}")
        
        # 학습 히스토리 저장
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_val_acc': self.best_val_acc
        }
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path.replace('.pth', '_history.json'), 'w') as f:
            json.dump(history, f)
    
    def save_model(self, path):
        """모델 저장"""
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
        print(f"모델 로드 완료! 최고 검증 정확도: {self.best_val_acc:.4f}")
