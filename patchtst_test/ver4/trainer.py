import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

class BitcoinTrainer:
    def __init__(self, model, class_weights=None, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        
        # 손실 함수
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # 옵티마이저
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=1e-4, 
            weight_decay=1e-5
        )
        
        # 스케줄러
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=10, verbose=True
        )
        
        # 기록
        self.train_losses = []
        self.val_accuracies = []
        self.best_val_acc = 0.0
        
        print(f"🚀 트레이너 초기화 완료 (Device: {self.device})")
    
    def train_epoch(self, train_loader):
        """한 에포크 학습 - 강화된 에러 처리"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        successful_batches = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (sequences, labels) in enumerate(pbar):
            try:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                
                # 입력 검증
                if sequences.size(0) == 0 or labels.size(0) == 0:
                    print(f"⚠️ 빈 배치 건너뜀: {batch_idx}")
                    continue
                
                # Forward pass
                self.optimizer.zero_grad()
                logits = self.model(sequences)
                loss = self.criterion(logits, labels)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                # 기록
                total_loss += loss.item()
                successful_batches += 1
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # 진행률 업데이트
                if len(all_labels) > 0:
                    current_acc = accuracy_score(all_labels, all_preds)
                    pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{current_acc:.4f}'
                    })
                
            except Exception as e:
                print(f"⚠️ 배치 {batch_idx} 처리 중 오류: {e}")
                continue
        
        if successful_batches == 0:
            print("❌ 성공한 배치가 없습니다!")
            return 0.0, 0.0
        
        avg_loss = total_loss / successful_batches
        accuracy = accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0.0
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """검증 - 강화된 에러 처리"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        successful_batches = 0
        
        with torch.no_grad():
            for batch_idx, (sequences, labels) in enumerate(tqdm(val_loader, desc="Validation")):
                try:
                    sequences = sequences.to(self.device)
                    labels = labels.to(self.device)
                    
                    if sequences.size(0) == 0 or labels.size(0) == 0:
                        continue
                    
                    logits = self.model(sequences)
                    loss = self.criterion(logits, labels)
                    
                    total_loss += loss.item()
                    successful_batches += 1
                    
                    preds = torch.argmax(logits, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                except Exception as e:
                    print(f"⚠️ 검증 배치 {batch_idx} 처리 중 오류: {e}")
                    continue
        
        if successful_batches == 0:
            return 0.0, 0.0, [], []
        
        avg_loss = total_loss / successful_batches
        accuracy = accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0.0
        
        return avg_loss, accuracy, all_preds, all_labels
    
    def train(self, train_loader, val_loader, epochs=100, save_path='models/best_model.pth'):
        """전체 학습 과정 - 강화된 에러 처리"""
        print(f"🎯 학습 시작 (에포크: {epochs})")
        
        for epoch in range(epochs):
            print(f"\n=== Epoch {epoch+1}/{epochs} ===")
            
            try:
                # 학습
                train_loss, train_acc = self.train_epoch(train_loader)
                self.train_losses.append(train_loss)
                
                # 검증
                val_loss, val_acc, val_preds, val_labels = self.validate(val_loader)
                self.val_accuracies.append(val_acc)
                
                # 스케줄러 업데이트
                self.scheduler.step(val_acc)
                
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                
                # 최고 모델 저장
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.save_model(save_path)
                    print(f"🎉 새로운 최고 모델 저장! (Acc: {val_acc:.4f})")
                    
                    # 상세 분류 리포트 (에러 처리)
                    if len(val_labels) > 0 and len(val_preds) > 0:
                        try:
                            print("\n분류 리포트:")
                            print(classification_report(val_labels, val_preds, 
                                                      target_names=['보합', '상승', '하락'],
                                                      zero_division=0))
                        except Exception as e:
                            print(f"분류 리포트 생성 실패: {e}")
                
                # 조기 종료
                if epoch > 20 and len(self.val_accuracies) >= 10:
                    recent_avg = np.mean(self.val_accuracies[-10:])
                    if val_acc < recent_avg * 0.95:
                        print("조기 종료!")
                        break
                        
            except Exception as e:
                print(f"❌ 에포크 {epoch+1} 처리 중 오류: {e}")
                continue
        
        print(f"✅ 학습 완료! 최고 검증 정확도: {self.best_val_acc:.4f}")
    
    def save_model(self, path):
        """모델 저장"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_val_acc': self.best_val_acc,
                'config': self.model.config
            }, path)
        except Exception as e:
            print(f"⚠️ 모델 저장 실패: {e}")
    
    def load_model(self, path):
        """모델 로드"""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.best_val_acc = checkpoint['best_val_acc']
            print(f"✅ 모델 로드 완료! (최고 정확도: {self.best_val_acc:.4f})")
        except Exception as e:
            print(f"⚠️ 모델 로드 실패: {e}")
