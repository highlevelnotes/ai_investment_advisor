import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report, 
                           confusion_matrix, precision_recall_fscore_support)
import os

class ModelEvaluator:
    def __init__(self, model, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
    
    def predict(self, test_loader):
        """테스트 예측"""
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.model(sequences)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_preds), np.array(all_labels), np.array(all_probs)
    
    def calculate_metrics(self, y_true, y_pred, y_prob):
        """평가 지표 계산"""
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        # 클래스별 지표
        precision_per_class, recall_per_class, f1_per_class, _ = \
            precision_recall_fscore_support(y_true, y_pred, average=None)
        
        # 혼동 행렬
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'confusion_matrix': cm
        }
    
    def plot_results(self, y_true, y_pred, y_prob, save_dir='results'):
        """결과 시각화"""
        os.makedirs(save_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 혼동 행렬
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        axes[0, 0].set_xticklabels(['보합', '상승', '하락'])
        axes[0, 0].set_yticklabels(['보합', '상승', '하락'])
        
        # 클래스별 정확도
        class_acc = [cm[i, i] / cm[i].sum() for i in range(len(cm))]
        axes[0, 1].bar(['보합', '상승', '하락'], class_acc)
        axes[0, 1].set_title('Class-wise Accuracy')
        axes[0, 1].set_ylabel('Accuracy')
        
        # 예측 확률 분포
        for i, class_name in enumerate(['보합', '상승', '하락']):
            axes[1, 0].hist(y_prob[y_true == i, i], alpha=0.5, 
                           label=f'{class_name} (실제)', bins=20)
        axes[1, 0].set_title('Prediction Probability Distribution')
        axes[1, 0].set_xlabel('Probability')
        axes[1, 0].legend()
        
        # 시간별 예측 결과 (샘플)
        sample_size = min(200, len(y_true))
        sample_idx = np.random.choice(len(y_true), sample_size, replace=False)
        axes[1, 1].scatter(range(sample_size), y_true[sample_idx], 
                          alpha=0.6, label='실제', s=20)
        axes[1, 1].scatter(range(sample_size), y_pred[sample_idx], 
                          alpha=0.6, label='예측', s=20)
        axes[1, 1].set_title('Sample Predictions')
        axes[1, 1].set_xlabel('Sample Index')
        axes[1, 1].set_ylabel('Class')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/evaluation_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate(self, test_loader, save_dir='results'):
        """전체 평가"""
        print("🔍 모델 평가 시작...")
        
        # 예측
        y_pred, y_true, y_prob = self.predict(test_loader)
        
        # 지표 계산
        metrics = self.calculate_metrics(y_true, y_pred, y_prob)
        
        # 결과 출력
        print("\n📊 평가 결과:")
        print(f"   정확도: {metrics['accuracy']:.4f}")
        print(f"   정밀도: {metrics['precision']:.4f}")
        print(f"   재현율: {metrics['recall']:.4f}")
        print(f"   F1-Score: {metrics['f1']:.4f}")
        
        print("\n클래스별 성능:")
        class_names = ['보합', '상승', '하락']
        for i, name in enumerate(class_names):
            if i < len(metrics['precision_per_class']):
                print(f"   {name}: P={metrics['precision_per_class'][i]:.3f}, "
                      f"R={metrics['recall_per_class'][i]:.3f}, "
                      f"F1={metrics['f1_per_class'][i]:.3f}")
        
        # 상세 분류 리포트
        print("\n📋 상세 분류 리포트:")
        print(classification_report(y_true, y_pred, target_names=class_names))
        
        # 시각화
        self.plot_results(y_true, y_pred, y_prob, save_dir)
        
        # 결과 저장
        os.makedirs(save_dir, exist_ok=True)
        
        results_df = pd.DataFrame({
            'Actual': y_true,
            'Predicted': y_pred,
            'Prob_Hold': y_prob[:, 0],
            'Prob_Up': y_prob[:, 1],
            'Prob_Down': y_prob[:, 2],
            'Correct': y_true == y_pred
        })
        results_df.to_csv(f'{save_dir}/predictions.csv', index=False)
        
        return metrics, results_df
