# agents/sentiment_analyzer.py
import asyncio
from typing import List, Dict, Any
import logging
from .hyperclova_client import HyperCLOVAXClient

logger = logging.getLogger(__name__)

class SentimentAnalyzerAgent:
    def __init__(self):
        self.clova_client = HyperCLOVAXClient()
        
    async def analyze_news_sentiment(self, news_articles: List[Dict[str, str]]) -> Dict[str, Any]:
        """LangChain HyperCLOVA X를 사용한 뉴스 감정 분석"""
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
                
                # 뉴스 텍스트 결합
                combined_text = "\n".join([
                    f"제목: {article['title']}\n요약: {article['summary']}" 
                    for article in articles[:5]  # 최대 5개 기사
                ])
                
                if not combined_text.strip():
                    logger.warning(f"{ticker}: 분석할 텍스트가 없습니다.")
                    continue
                
                logger.info(f"{ticker} HyperCLOVA X 감정 분석 시작 (뉴스 {len(articles)}개)")
                
                # HyperCLOVA X로 감정 분석
                sentiment_analysis = await self._analyze_sentiment_with_clova(combined_text, ticker)
                
                sentiment_results[ticker] = {
                    'sentiment_score': sentiment_analysis['score'],
                    'confidence': sentiment_analysis['confidence'],
                    'article_count': len(articles),
                    'analysis_summary': sentiment_analysis['summary'],
                    'method': 'langchain_hyperclova_x',
                    'sample_titles': [article['title'][:50] for article in articles[:3]]
                }
                
                logger.info(f"{ticker} 감정 분석 완료: 점수 {sentiment_analysis['score']:.2f}")
                
                # API 호출 제한을 위한 지연
                await asyncio.sleep(0.5)
                
        except Exception as e:
            logger.error(f"감정 분석 전체 오류: {e}")
            
        return sentiment_results
    
    async def _analyze_sentiment_with_clova(self, text: str, ticker: str) -> Dict[str, Any]:
        """LangChain HyperCLOVA X를 사용한 개별 감정 분석"""
        
        # 텍스트 길이 제한 (한국어 특성 고려)
        if len(text) > 1000:
            text = text[:1000] + "..."
        
        prompt = f"""
다음은 {ticker} 종목과 관련된 한국 금융 뉴스입니다. 이 뉴스들의 전반적인 투자 감정을 분석해주세요.

뉴스 내용:
{text}

분석 기준:
- 주가에 긍정적 영향: +1.0 ~ +0.1
- 중립적 영향: +0.1 ~ -0.1  
- 주가에 부정적 영향: -0.1 ~ -1.0

다음 형식으로 정확히 응답해주세요:
감정 점수: [숫자]
신뢰도: [숫자]
분석 요약: [2-3문장 설명]

예시:
감정 점수: 0.3
신뢰도: 0.8
분석 요약: 실적 개선과 신제품 출시 소식으로 긍정적인 시장 반응이 예상됩니다.
"""
        
        messages = [
            {"role": "system", "content": "당신은 한국 주식시장 전문 감정 분석가입니다. 뉴스가 주가에 미치는 영향을 정확하게 분석해주세요."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self.clova_client.generate_response(
                messages, 
                max_tokens=300, 
                temperature=0.3
            )
            
            # 응답 파싱
            sentiment_score = self._extract_sentiment_score(response)
            confidence = self._extract_confidence(response)
            summary = self._extract_summary(response)
            
            # 유효성 검증
            if abs(sentiment_score) > 1.0:
                sentiment_score = max(-1.0, min(1.0, sentiment_score))
            
            if confidence < 0 or confidence > 1:
                confidence = 0.7  # 기본값
            
            return {
                'score': sentiment_score,
                'confidence': confidence,
                'summary': summary,
                'raw_response': response[:200]  # 디버깅용
            }
            
        except Exception as e:
            logger.error(f"HyperCLOVA X 감정 분석 오류 {ticker}: {e}")
            return {
                'score': 0.0,
                'confidence': 0.5,
                'summary': f'분석 중 오류가 발생했습니다: {str(e)[:50]}'
            }
    
    def _extract_sentiment_score(self, response: str) -> float:
        """응답에서 감정 점수 추출"""
        try:
            import re
            
            # 다양한 패턴으로 점수 추출 시도
            patterns = [
                r'감정\s*점수\s*[:：]\s*([+-]?\d+\.?\d*)',
                r'점수\s*[:：]\s*([+-]?\d+\.?\d*)',
                r'([+-]?\d+\.?\d*)\s*점',
                r'([+-]?[01]\.?\d*)'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, response)
                if matches:
                    score = float(matches[0])
                    return max(-1.0, min(1.0, score))
            
            # 키워드 기반 대체 분석
            if '긍정' in response or '호재' in response or '상승' in response:
                return 0.3
            elif '부정' in response or '악재' in response or '하락' in response:
                return -0.3
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"감정 점수 추출 실패: {e}")
            return 0.0
    
    def _extract_confidence(self, response: str) -> float:
        """신뢰도 추출"""
        try:
            import re
            
            patterns = [
                r'신뢰도\s*[:：]\s*(\d+\.?\d*)',
                r'확신도\s*[:：]\s*(\d+\.?\d*)',
                r'(\d+\.?\d*)\s*%'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, response)
                if matches:
                    confidence = float(matches[0])
                    if confidence > 1:  # 백분율인 경우
                        confidence = confidence / 100
                    return max(0.0, min(1.0, confidence))
            
            return 0.7  # 기본값
            
        except Exception as e:
            logger.warning(f"신뢰도 추출 실패: {e}")
            return 0.7
    
    def _extract_summary(self, response: str) -> str:
        """분석 요약 추출"""
        try:
            lines = response.split('\n')
            
            for i, line in enumerate(lines):
                if '분석' in line and '요약' in line and ':' in line:
                    summary = line.split(':', 1)[1].strip()
                    if summary:
                        return summary
                elif '분석' in line and '요약' in line and i + 1 < len(lines):
                    return lines[i + 1].strip()
            
            # 마지막 줄 반환
            non_empty_lines = [line.strip() for line in lines if line.strip()]
            if non_empty_lines:
                return non_empty_lines[-1]
            
            return "감정 분석이 완료되었습니다."
            
        except Exception as e:
            logger.warning(f"요약 추출 실패: {e}")
            return "분석 요약을 추출할 수 없습니다."
