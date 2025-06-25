# model_evaluator.py
import pandas as pd
from langchain_naver import ChatClovaX
from config import Config
import json
import numpy as np
from sklearn.metrics import accuracy_score
import re

class ModelEvaluator:
    def __init__(self, use_finetuned=True):
        """모델 평가기 초기화"""
        if use_finetuned and hasattr(Config, 'FINETUNED_MODEL_ID'):
            model_name = Config.FINETUNED_MODEL_ID
            self.model_type = "Fine-tuned"
        else:
            model_name = Config.HYPERCLOVA_MODEL
            self.model_type = "Base"
        
        self.client = ChatClovaX(
            api_key=Config.HYPERCLOVA_X_API_KEY,
            model=model_name,
            max_tokens=3000,
            temperature=0.3
        )
    
    def evaluate_portfolio_recommendations(self, test_data):
        """포트폴리오 추천 성능 평가"""
        results = []
        
        for idx, row in test_data.iterrows():
            try:
                # 모델 응답 생성
                response = self.client.invoke(row['Text'])
                prediction = response.content
                
                # 성능 지표 계산
                metrics = self._calculate_metrics(row['Completion'], prediction)
                
                results.append({
                    'input': row['Text'],
                    'expected': row['Completion'],
                    'predicted': prediction,
                    'json_format_score': metrics['json_format_score'],
                    'content_relevance_score': metrics['content_relevance_score'],
                    'recommendation_completeness': metrics['recommendation_completeness']
                })
                
            except Exception as e:
                print(f"Error evaluating row {idx}: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def _calculate_metrics(self, expected, predicted):
        """평가 지표 계산"""
        metrics = {}
        
        # JSON 형식 준수도
        try:
            json.loads(predicted)
            metrics['json_format_score'] = 1.0
        except:
            metrics['json_format_score'] = 0.0
        
        # 필수 요소 포함도
        required_elements = [
            '자산배분', '추천', 'ETF', '근거', '수익률', '리스크'
        ]
        
        element_scores = []
        for element in required_elements:
            if element in predicted:
                element_scores.append(1.0)
            else:
                element_scores.append(0.0)
        
        metrics['content_relevance_score'] = np.mean(element_scores)
        
        # 추천 완성도 (길이 기반)
        if len(predicted) > 500:
            metrics['recommendation_completeness'] = 1.0
        elif len(predicted) > 200:
            metrics['recommendation_completeness'] = 0.5
        else:
            metrics['recommendation_completeness'] = 0.0
        
        return metrics
    
    def generate_evaluation_report(self, results_df):
        """평가 리포트 생성"""
        report = {
            'model_type': self.model_type,
            'total_samples': len(results_df),
            'avg_json_format_score': results_df['json_format_score'].mean(),
            'avg_content_relevance_score': results_df['content_relevance_score'].mean(),
            'avg_recommendation_completeness': results_df['recommendation_completeness'].mean(),
            'overall_score': (results_df['json_format_score'].mean() + 
                            results_df['content_relevance_score'].mean() + 
                            results_df['recommendation_completeness'].mean()) / 3
        }
        
        return report

# 평가 실행 함수
def run_model_evaluation():
    """모델 평가 실행"""
    # 테스트 데이터 로드
    test_data = pd.read_csv("val_data_cleaned.csv").head(50)  # 평가용 50개 샘플
    
    # 베이스 모델 평가
    print("Evaluating base model...")
    base_evaluator = ModelEvaluator(use_finetuned=False)
    base_results = base_evaluator.evaluate_portfolio_recommendations(test_data)
    base_report = base_evaluator.generate_evaluation_report(base_results)
    
    # 파인튜닝 모델 평가 (모델이 있는 경우)
    finetuned_report = None
    if hasattr(Config, 'FINETUNED_MODEL_ID'):
        print("Evaluating fine-tuned model...")
        finetuned_evaluator = ModelEvaluator(use_finetuned=True)
        finetuned_results = finetuned_evaluator.evaluate_portfolio_recommendations(test_data)
        finetuned_report = finetuned_evaluator.generate_evaluation_report(finetuned_results)
    
    # 결과 비교
    print("\n=== 모델 성능 비교 ===")
    print(f"Base Model Score: {base_report['overall_score']:.3f}")
    if finetuned_report:
        print(f"Fine-tuned Model Score: {finetuned_report['overall_score']:.3f}")
        improvement = finetuned_report['overall_score'] - base_report['overall_score']
        print(f"Improvement: {improvement:.3f} ({improvement/base_report['overall_score']*100:.1f}%)")
    
    return base_report, finetuned_report

if __name__ == "__main__":
    base_report, finetuned_report = run_model_evaluation()
