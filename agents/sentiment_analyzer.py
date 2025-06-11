# agents/sentiment_analyzer.py 수정
import asyncio
from typing import List, Dict, Any
import logging
from .kr_finbert_analyzer import KRFinBERTAnalyzer
from .hyperclova_client import HyperCLOVAXClient
import numpy as np

logger = logging.getLogger(__name__)

class SentimentAnalyzerAgent:
    def __init__(self):
        self.kr_finbert = KRFinBERTAnalyzer()
        self.clova_client = HyperCLOVAXClient()
        
    async def analyze_news_sentiment(self, news_articles: List[Dict[str, str]]) -> Dict[str, Any]:
        """KR-FinBERT + HyperCLOVA X 하이브리드 감정 분석"""
        sentiment_results = {}
        
        if not news_articles:
            logger.warning("분석할 뉴스 기사가 없습니다.")
            return sentiment_results
        
        try:
            # 종목별로 뉴스 그룹화
            ticker_news = {}
            for article in news_articles:
                ticker = article['ticker']
                if ticker not in ticker_news:
                    ticker_news[ticker] = []
                ticker_news[ticker].append(article)
            
            logger.info(f"감정 분석 대상: {len(ticker_news)}개 종목, {len(news_articles)}개 뉴스")
            
            for ticker, articles in ticker_news.items():
                if not articles:
                    continue
                
                # 뉴스 텍스트 준비
                texts = []
                for article in articles[:5]:  # 최대 5개 기사
                    text = f"{article['title']} {article['summary']}"
                    texts.append(text)
                
                if not texts:
                    continue
                
                logger.info(f"{ticker} KR-FinBERT 감정 분석 시작")
                
                # KR-FinBERT로 개별 기사 분석
                kr_finbert_results = self.kr_finbert.analyze_sentiment(texts)
                
                # HyperCLOVA X로 종합 분석
                combined_text = "\n".join(texts)
                clova_analysis = await self._analyze_with_clova(combined_text, ticker)
                
                # 결과 통합
                sentiment_results[ticker] = self._combine_analysis_results(
                    kr_finbert_results, clova_analysis, articles
                )
                
                logger.info(f"{ticker} 감정 분석 완료: KR-FinBERT 점수 {sentiment_results[ticker]['kr_finbert_score']:.2f}")
                
                await asyncio.sleep(0.5)
                
        except Exception as e:
            logger.error(f"감정 분석 전체 오류: {e}")
            
        return sentiment_results
    
    def _combine_analysis_results(self, kr_finbert_results: List[Dict], 
                                clova_analysis: Dict, articles: List[Dict]) -> Dict[str, Any]:
        """KR-FinBERT와 HyperCLOVA X 결과 통합"""
        
        # KR-FinBERT 평균 점수 계산
        if kr_finbert_results:
            kr_scores = [result['sentiment_score'] for result in kr_finbert_results]
            kr_confidences = [result['confidence'] for result in kr_finbert_results]
            
            kr_avg_score = np.mean(kr_scores)
            kr_avg_confidence = np.mean(kr_confidences)
        else:
            kr_avg_score = 0.0
            kr_avg_confidence = 0.5
        
        # 가중 평균으로 최종 점수 계산
        kr_weight = 0.7  # KR-FinBERT 가중치
        clova_weight = 0.3  # HyperCLOVA X 가중치
        
        final_score = (kr_avg_score * kr_weight) + (clova_analysis['score'] * clova_weight)
        final_confidence = (kr_avg_confidence * kr_weight) + (clova_analysis['confidence'] * clova_weight)
        
        return {
            'sentiment_score': final_score,
            'confidence': final_confidence,
            'article_count': len(articles),
            'kr_finbert_score': kr_avg_score,
            'kr_finbert_confidence': kr_avg_confidence,
            'clova_score': clova_analysis['score'],
            'clova_summary': clova_analysis['summary'],
            'method': 'kr_finbert_clova_hybrid',
            'detailed_results': kr_finbert_results[:3],  # 상위 3개 결과
            'sample_titles': [article['title'][:50] for article in articles[:3]]
        }
    
    async def _analyze_with_clova(self, text: str, ticker: str) -> Dict[str, Any]:
        """HyperCLOVA X 보조 분석"""
        prompt = f"""
다음은 {ticker} 종목 관련 뉴스입니다. 투자 관점에서 감정을 분석해주세요.

뉴스: {text[:800]}

감정 점수 (-1.0 ~ +1.0)와 한 줄 요약을 제공해주세요.
"""
        
        messages = [
            {"role": "system", "content": "한국 금융 시장 감정 분석 전문가입니다."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self.clova_client.generate_response(messages, max_tokens=200)
            
            # 간단한 파싱
            score = self._extract_score_from_response(response)
            summary = response.split('\n')[-1] if '\n' in response else response
            
            return {
                'score': score,
                'confidence': 0.7,
                'summary': summary[:100]
            }
            
        except Exception as e:
            logger.error(f"HyperCLOVA X 보조 분석 오류: {e}")
            return {'score': 0.0, 'confidence': 0.5, 'summary': '분석 실패'}
    
    def _extract_score_from_response(self, response: str) -> float:
        """응답에서 점수 추출"""
        import re
        numbers = re.findall(r'-?\d+\.?\d*', response)
        if numbers:
            score = float(numbers[0])
            return max(-1.0, min(1.0, score))
        return 0.0
