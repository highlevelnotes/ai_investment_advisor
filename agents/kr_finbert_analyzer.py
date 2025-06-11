# agents/kr_finbert_analyzer.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class KRFinBERTAnalyzer:
    def __init__(self):
        try:
            # KR-FinBERT 모델 로드
            self.model_name = "snunlp/KR-FinBert-SC"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.eval()
            
            # 라벨 매핑 (KR-FinBERT-SC 기준)
            self.label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
            self.available = True
            
            logger.info("KR-FinBERT 모델 로드 완료")
            
        except Exception as e:
            logger.error(f"KR-FinBERT 모델 로드 실패: {e}")
            self.available = False
    
    def analyze_sentiment(self, texts: List[str]) -> List[Dict[str, Any]]:
        """KR-FinBERT를 사용한 감정 분석"""
        if not self.available:
            return self._fallback_analysis(texts)
        
        results = []
        
        try:
            for text in texts:
                # 텍스트 전처리
                text = text[:512]  # 최대 길이 제한
                
                # 토큰화
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                # 예측
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    
                    # 결과 추출
                    predicted_class = torch.argmax(predictions, dim=-1).item()
                    confidence = torch.max(predictions).item()
                    
                    # 감정 점수 계산 (-1 ~ +1)
                    sentiment_score = self._calculate_sentiment_score(predictions[0])
                    
                    results.append({
                        'text': text[:100] + "..." if len(text) > 100 else text,
                        'sentiment_label': self.label_mapping[predicted_class],
                        'sentiment_score': sentiment_score,
                        'confidence': confidence,
                        'raw_scores': {
                            'negative': predictions[0][0].item(),
                            'neutral': predictions[0][1].item(),
                            'positive': predictions[0][2].item()
                        }
                    })
                    
        except Exception as e:
            logger.error(f"KR-FinBERT 분석 오류: {e}")
            return self._fallback_analysis(texts)
        
        return results
    
    def _calculate_sentiment_score(self, predictions: torch.Tensor) -> float:
        """감정 점수 계산 (-1 ~ +1)"""
        negative_score = predictions[0].item()
        neutral_score = predictions[1].item()
        positive_score = predictions[2].item()
        
        # 가중 평균으로 감정 점수 계산
        sentiment_score = (positive_score * 1.0) + (neutral_score * 0.0) + (negative_score * -1.0)
        
        return sentiment_score
    
    def _fallback_analysis(self, texts: List[str]) -> List[Dict[str, Any]]:
        """대체 감정 분석"""
        positive_keywords = ['상승', '오름', '증가', '호재', '긍정', '성장', '개선']
        negative_keywords = ['하락', '내림', '감소', '악재', '부정', '손실', '우려']
        
        results = []
        for text in texts:
            pos_count = sum(1 for word in positive_keywords if word in text)
            neg_count = sum(1 for word in negative_keywords if word in text)
            
            if pos_count > neg_count:
                sentiment_score = 0.3
                label = 'positive'
            elif neg_count > pos_count:
                sentiment_score = -0.3
                label = 'negative'
            else:
                sentiment_score = 0.0
                label = 'neutral'
            
            results.append({
                'text': text[:100] + "..." if len(text) > 100 else text,
                'sentiment_label': label,
                'sentiment_score': sentiment_score,
                'confidence': 0.6,
                'method': 'keyword_fallback'
            })
        
        return results
