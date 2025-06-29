import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, 
                           roc_auc_score, confusion_matrix, classification_report,
                           roc_curve, precision_recall_curve)
import os

class ClassificationEvaluator:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
    
    def predict(self, test_sequences):
        """테스트 데이터 예측"""
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for i in range(0, len(test_sequences), 32):
                batch = test_sequences[i:i+32]
                batch_tensor = torch.FloatTensor(batch).to(self.device)
                
                logits = self.model(batch_tensor)
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_probabilities)
    
    def calculate_metrics(self, y_true, y_pred, y_prob):
        """분류 메트릭 계산"""
        # 기본 메트릭
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        # 클래스별 메트릭
        precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )
        
        # ROC AUC
        auc = roc_auc_score(y_true, y_prob[:, 1])
        
        # 혼동 행렬
        cm = confusion_matrix(y_true, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'confusion_matrix': cm
        }
        
        return metrics
    
    def plot_results(self, y_true, y_pred, y_prob, save_path='results/classification/plots'):
        """결과 시각화"""
        os.makedirs(save_path, exist_ok=True)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 혼동 행렬
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        axes[0, 0].set_xticklabels(['하락', '상승'])
        axes[0, 0].set_yticklabels(['하락', '상승'])
        
        # 2. ROC 곡선
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        auc = roc_auc_score(y_true, y_prob[:, 1])
        axes[0, 1].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], 'k--')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 3. Precision-Recall 곡선
        precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
        axes[0, 2].plot(recall, precision)
        axes[0, 2].set_xlabel('Recall')
        axes[0, 2].set_ylabel('Precision')
        axes[0, 2].set_title('Precision-Recall Curve')
        axes[0, 2].grid(True)
        
        # 4. 예측 확률 분포
        axes[1, 0].hist(y_prob[y_true == 0, 1], alpha=0.5, label='하락 실제', bins=30)
        axes[1, 0].hist(y_prob[y_true == 1, 1], alpha=0.5, label='상승 실제', bins=30)
        axes[1, 0].set_xlabel('상승 예측 확률')
        axes[1, 0].set_ylabel('빈도')
        axes[1, 0].set_title('예측 확률 분포')
        axes[1, 0].legend()
        
        # 5. 클래스별 성능
        precision_per_class = precision_recall_fscore_support(y_true, y_pred, average=None)[0]
        recall_per_class = precision_recall_fscore_support(y_true, y_pred, average=None)[1]
        f1_per_class = precision_recall_fscore_support(y_true, y_pred, average=None)[2]
        
        x = ['하락', '상승']
        x_pos = np.arange(len(x))
        
        width = 0.25
        axes[1, 1].bar(x_pos - width, precision_per_class, width, label='Precision')
        axes[1, 1].bar(x_pos, recall_per_class, width, label='Recall')
        axes[1, 1].bar(x_pos + width, f1_per_class, width, label='F1-Score')
        
        axes[1, 1].set_xlabel('클래스')
        axes[1, 1].set_ylabel('점수')
        axes[1, 1].set_title('클래스별 성능')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(x)
        axes[1, 1].legend()
        
        # 6. 시간별 예측 결과 (샘플)
        sample_size = min(200, len(y_true))
        sample_indices = np.random.choice(len(y_true), sample_size, replace=False)
        sample_indices = np.sort(sample_indices)
        
        axes[1, 2].scatter(range(sample_size), y_true[sample_indices], 
                          alpha=0.6, label='실제', s=20)
        axes[1, 2].scatter(range(sample_size), y_pred[sample_indices], 
                          alpha=0.6, label='예측', s=20)
        axes[1, 2].set_xlabel('시간 (샘플)')
        axes[1, 2].set_ylabel('클래스 (0: 하락, 1: 상승)')
        axes[1, 2].set_title('시간별 예측 결과 (샘플)')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/classification_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate_classification(self, test_sequences, test_labels):
        """전체 분류 평가"""
        print("분류 모델 평가 시작...")
        
        # 예측 수행
        predictions, probabilities = self.predict(test_sequences)
        
        # 메트릭 계산
        metrics = self.calculate_metrics(test_labels, predictions, probabilities)
        
        # 결과 출력
        print("\n=== 분류 성능 결과 ===")
        print(f"정확도 (Accuracy): {metrics['accuracy']:.4f}")
        print(f"정밀도 (Precision): {metrics['precision']:.4f}")
        print(f"재현율 (Recall): {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1']:.4f}")
        print(f"ROC AUC: {metrics['auc']:.4f}")
        
        print(f"\n클래스별 성능:")
        print(f"하락 클래스 - Precision: {metrics['precision_per_class'][0]:.4f}, "
              f"Recall: {metrics['recall_per_class'][0]:.4f}, "
              f"F1: {metrics['f1_per_class'][0]:.4f}")
        print(f"상승 클래스 - Precision: {metrics['precision_per_class'][1]:.4f}, "
              f"Recall: {metrics['recall_per_class'][1]:.4f}, "
              f"F1: {metrics['f1_per_class'][1]:.4f}")
        
        # 혼동 행렬
        print(f"\n혼동 행렬:")
        print(f"실제\\예측  하락  상승")
        print(f"하락      {metrics['confusion_matrix'][0,0]:4d}  {metrics['confusion_matrix'][0,1]:4d}")
        print(f"상승      {metrics['confusion_matrix'][1,0]:4d}  {metrics['confusion_matrix'][1,1]:4d}")
        
        # 시각화
        self.plot_results(test_labels, predictions, probabilities)
        
        # 결과 저장
        results_df = pd.DataFrame({
            'Actual': test_labels,
            'Predicted': predictions,
            'Prob_Down': probabilities[:, 0],
            'Prob_Up': probabilities[:, 1],
            'Correct': test_labels == predictions
        })
        
        os.makedirs('results/classification', exist_ok=True)
        results_df.to_csv('results/classification/predictions.csv', index=False)
        
        # 메트릭 저장
        metrics_to_save = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                          for k, v in metrics.items() if k != 'confusion_matrix'}
        metrics_to_save['confusion_matrix'] = metrics['confusion_matrix'].tolist()
        
        import json
        with open('results/classification/metrics.json', 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
        
        return metrics, results_df
