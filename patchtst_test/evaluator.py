import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

class ModelEvaluator:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
    
    def predict(self, test_loader):
        """테스트 데이터에 대한 예측"""
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_sequences, batch_targets in test_loader:
                batch_sequences = batch_sequences.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                pred = self.model(batch_sequences)
                
                # 타겟 차원 맞추기
                if len(batch_targets.shape) > len(pred.shape):
                    batch_targets = batch_targets.squeeze(-1)
                
                predictions.extend(pred.cpu().numpy())
                actuals.extend(batch_targets.cpu().numpy())
        
        return np.array(predictions), np.array(actuals)
    
    def calculate_metrics(self, predictions, actuals):
        """평가 지표 계산"""
        # 1차원으로 변환
        if len(predictions.shape) > 1:
            predictions = predictions.flatten()
        if len(actuals.shape) > 1:
            actuals = actuals.flatten()
        
        metrics = {
            'MSE': mean_squared_error(actuals, predictions),
            'RMSE': np.sqrt(mean_squared_error(actuals, predictions)),
            'MAE': mean_absolute_error(actuals, predictions),
            'R2': r2_score(actuals, predictions),
            'MAPE': np.mean(np.abs((actuals - predictions) / actuals)) * 100
        }
        
        return metrics
    
    def plot_predictions(self, predictions, actuals, save_path='outputs/visualizations/predictions.png'):
        """예측 결과 시각화"""
        plt.figure(figsize=(15, 8))
        
        # 샘플 수 제한 (시각화를 위해)
        n_samples = min(500, len(predictions))
        indices = np.arange(n_samples)
        
        plt.subplot(2, 1, 1)
        plt.plot(indices, actuals[:n_samples], label='Actual', alpha=0.7)
        plt.plot(indices, predictions[:n_samples], label='Predicted', alpha=0.7)
        plt.title('Bitcoin Price Prediction Results')
        plt.xlabel('Time Steps')
        plt.ylabel('Normalized Price')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        errors = predictions[:n_samples] - actuals[:n_samples]
        plt.plot(indices, errors, color='red', alpha=0.7)
        plt.title('Prediction Errors')
        plt.xlabel('Time Steps')
        plt.ylabel('Error')
        plt.grid(True)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_scatter(self, predictions, actuals, save_path='outputs/visualizations/scatter.png'):
        """산점도 그래프"""
        plt.figure(figsize=(10, 8))
        plt.scatter(actuals, predictions, alpha=0.5)
        plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        plt.grid(True)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate_model(self, test_loader, save_results=True):
        """전체 평가 과정"""
        print("모델 평가 시작...")
        
        # 예측 수행
        predictions, actuals = self.predict(test_loader)
        
        # 지표 계산
        metrics = self.calculate_metrics(predictions, actuals)
        
        # 결과 출력
        print("\n=== 평가 결과 ===")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.6f}")
        
        if save_results:
            # 시각화
            self.plot_predictions(predictions, actuals)
            self.plot_scatter(predictions.flatten(), actuals.flatten())
            
            # 결과 저장
            results_df = pd.DataFrame({
                'Actual': actuals.flatten(),
                'Predicted': predictions.flatten(),
                'Error': predictions.flatten() - actuals.flatten()
            })
            
            os.makedirs('outputs/predictions', exist_ok=True)
            results_df.to_csv('outputs/predictions/test_results.csv', index=False)
            
            # 지표 저장
            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv('outputs/predictions/metrics.csv', index=False)
        
        return metrics, predictions, actuals
