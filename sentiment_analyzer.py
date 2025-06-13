# sentiment_analyzer.py
import asyncio
from typing import List, Dict, Any
import logging
from langchain_naver import ChatClovaX
from langchain_core.messages import HumanMessage, SystemMessage
from news_collector import NewsCollector

logger = logging.getLogger(__name__)

class RealSentimentAnalyzer:
    def __init__(self):
        try:
            self.llm = ChatClovaX(
                model="HCX-005",
                temperature=0.1,  # 감정 분석은 일관성이 중요
                max_tokens=200
            )
            self.llm_available = True
        except:
            self.llm_available = False
            logger.warning("HyperCLOVA X 모델 로드 실패 - 대체 분석 사용")
    
    async def analyze_stock_sentiment(self, ticker: str) -> Dict[str, Any]:
        """실제 뉴스 데이터 기반 감정 분석"""
        
        # 1단계: 뉴스 데이터 수집
        async with NewsCollector() as collector:
            news_articles = await collector.collect_news(ticker, max_articles=15)
        
        if not news_articles:
            logger.warning(f"{ticker}: 수집된 뉴스가 없습니다")
            return self._get_default_sentiment(ticker)
        
        logger.info(f"{ticker}: {len(news_articles)}개 뉴스 수집 완료")
        
        # 2단계: 뉴스별 감정 분석
        sentiment_scores = []
        analyzed_articles = []
        
        for i, article in enumerate(news_articles[:10]):  # 최대 10개 분석
            try:
                article_sentiment = await self._analyze_single_article(article, ticker)
                if article_sentiment:
                    sentiment_scores.append(article_sentiment['score'])
                    analyzed_articles.append({
                        'title': article['title'][:80] + "..." if len(article['title']) > 80 else article['title'],
                        'sentiment': article_sentiment['label'],
                        'score': article_sentiment['score'],
                        'source': article['source']
                    })
                
                # API 호출 제한 (0.5초 간격)
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.warning(f"기사 분석 실패: {e}")
                continue
        
        # 3단계: 종합 감정 점수 계산
        if sentiment_scores:
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            confidence = min(0.95, 0.6 + (len(sentiment_scores) * 0.05))  # 분석한 기사 수에 따라 신뢰도 증가
            
            # 감정 라벨 결정
            if avg_sentiment > 0.15:
                sentiment_label = "긍정적"
                sentiment_emoji = "😊"
            elif avg_sentiment < -0.15:
                sentiment_label = "부정적"
                sentiment_emoji = "😟"
            else:
                sentiment_label = "중립적"
                sentiment_emoji = "😐"
            
            return {
                'sentiment_score': avg_sentiment,
                'sentiment_label': sentiment_label,
                'sentiment_emoji': sentiment_emoji,
                'confidence': confidence,
                'article_count': len(analyzed_articles),
                'analyzed_articles': analyzed_articles,
                'news_sources': list(set([article['source'] for article in analyzed_articles])),
                'method': 'real_news_analysis'
            }
        else:
            return self._get_default_sentiment(ticker)
    
    async def _analyze_single_article(self, article: Dict, ticker: str) -> Dict[str, Any]:
        """개별 기사 감정 분석"""
        
        title = article.get('title', '')
        summary = article.get('summary', '')
        
        if not title:
            return None
        
        # 분석할 텍스트 준비
        text_to_analyze = f"제목: {title}"
        if summary and summary != title:
            text_to_analyze += f"\n내용: {summary}"
        
        if self.llm_available:
            return await self._analyze_with_hyperclova(text_to_analyze, ticker)
        else:
            return self._analyze_with_keywords(text_to_analyze)
    
    async def _analyze_with_hyperclova(self, text: str, ticker: str) -> Dict[str, Any]:
        """HyperCLOVA X를 사용한 감정 분석"""
        
        prompt = f"""
다음은 {ticker} 주식과 관련된 뉴스입니다. 이 뉴스가 주가에 미치는 영향을 분석해주세요.

뉴스 내용:
{text}

다음 기준으로 분석해주세요:
- 매우 긍정적: +0.8 ~ +1.0 (강한 호재)
- 긍정적: +0.3 ~ +0.7 (일반적 호재)
- 약간 긍정적: +0.1 ~ +0.2 (약한 호재)
- 중립적: -0.1 ~ +0.1 (영향 없음)
- 약간 부정적: -0.2 ~ -0.1 (약한 악재)
- 부정적: -0.7 ~ -0.3 (일반적 악재)
- 매우 부정적: -1.0 ~ -0.8 (강한 악재)

응답 형식:
점수: [숫자]
이유: [한 줄 설명]
"""
        
        messages = [
            SystemMessage(content="당신은 전문 금융 뉴스 분석가입니다. 뉴스가 주가에 미치는 영향을 정확하게 분석해주세요."),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = self.llm.invoke(messages)
            return self._parse_hyperclova_response(response.content)
            
        except Exception as e:
            logger.warning(f"HyperCLOVA X 분석 실패: {e}")
            return self._analyze_with_keywords(text)
    
    def _parse_hyperclova_response(self, response: str) -> Dict[str, Any]:
        """HyperCLOVA X 응답 파싱"""
        try:
            lines = response.split('\n')
            score = 0.0
            reason = ""
            
            for line in lines:
                if '점수' in line and ':' in line:
                    score_text = line.split(':')[1].strip()
                    # 숫자 추출
                    import re
                    numbers = re.findall(r'-?\d+\.?\d*', score_text)
                    if numbers:
                        score = float(numbers[0])
                        score = max(-1.0, min(1.0, score))  # -1.0 ~ 1.0 범위 제한
                
                elif '이유' in line and ':' in line:
                    reason = line.split(':', 1)[1].strip()
            
            # 감정 라벨 결정
            if score > 0.1:
                label = "긍정적"
            elif score < -0.1:
                label = "부정적"
            else:
                label = "중립적"
            
            return {
                'score': score,
                'label': label,
                'reason': reason,
                'method': 'hyperclova_x'
            }
            
        except Exception as e:
            logger.warning(f"응답 파싱 실패: {e}")
            return {'score': 0.0, 'label': '중립적', 'reason': '분석 실패'}
    
    def _analyze_with_keywords(self, text: str) -> Dict[str, Any]:
        """키워드 기반 대체 감정 분석"""
        
        # 한국어 + 영어 키워드
        positive_keywords = [
            # 영어
            'beat', 'exceed', 'strong', 'growth', 'profit', 'gain', 'rise', 'up', 'bullish',
            'positive', 'good', 'excellent', 'outstanding', 'surge', 'rally', 'boom',
            # 한국어 (영어 뉴스에서도 종종 사용)
            '상승', '호재', '성장', '이익', '증가'
        ]
        
        negative_keywords = [
            # 영어
            'miss', 'weak', 'decline', 'loss', 'fall', 'down', 'bearish', 'negative',
            'poor', 'bad', 'disappointing', 'crash', 'plunge', 'drop', 'concern',
            # 한국어
            '하락', '악재', '손실', '감소', '우려'
        ]
        
        neutral_keywords = [
            'stable', 'maintain', 'hold', 'unchanged', 'flat', 'sideways'
        ]
        
        text_lower = text.lower()
        
        pos_count = sum(1 for word in positive_keywords if word in text_lower)
        neg_count = sum(1 for word in negative_keywords if word in text_lower)
        neu_count = sum(1 for word in neutral_keywords if word in text_lower)
        
        # 점수 계산
        total_words = pos_count + neg_count + neu_count
        if total_words == 0:
            score = 0.0
            label = "중립적"
        else:
            score = (pos_count - neg_count) / max(total_words, 1) * 0.5  # 최대 ±0.5
            
            if score > 0.1:
                label = "긍정적"
            elif score < -0.1:
                label = "부정적"
            else:
                label = "중립적"
        
        return {
            'score': score,
            'label': label,
            'reason': f"키워드 분석 (긍정: {pos_count}, 부정: {neg_count})",
            'method': 'keyword_based'
        }
    
    def _get_default_sentiment(self, ticker: str) -> Dict[str, Any]:
        """기본 감정 분석 결과"""
        return {
            'sentiment_score': 0.0,
            'sentiment_label': "중립적",
            'sentiment_emoji': "😐",
            'confidence': 0.5,
            'article_count': 0,
            'analyzed_articles': [],
            'news_sources': [],
            'method': 'no_news_available',
            'message': f"{ticker}에 대한 최신 뉴스를 찾을 수 없습니다."
        }
