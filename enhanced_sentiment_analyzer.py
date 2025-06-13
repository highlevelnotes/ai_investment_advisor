import asyncio
import aiohttp
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from langchain_naver import ChatClovaX

class EnhancedSentimentAnalyzer:
    def __init__(self):
        self.llm = ChatClovaX(model="HCX-005", temperature=0.2, max_tokens=2000)
        self.social_sources = ['reddit', 'twitter', 'stocktwits']
        self.analyst_sources = ['seeking_alpha', 'morningstar', 'yahoo_finance']
    
    async def analyze_multi_source_sentiment(self, ticker: str, 
                                           include_social: bool = True,
                                           include_analyst: bool = True) -> Dict:
        """다중 소스 감정 분석"""
        
        results = {}
        
        # 1. 뉴스 감정 분석 (기존 + 네이버)
        news_sentiment = await self._analyze_news_sentiment_enhanced(ticker)
        results['news_sentiment'] = news_sentiment
        
        # 2. 소셜미디어 감정 분석
        if include_social:
            social_sentiment = await self._analyze_social_media_sentiment(ticker)
            results['social_sentiment'] = social_sentiment
        
        # 3. 애널리스트 리포트 분석
        if include_analyst:
            analyst_sentiment = await self._analyze_analyst_reports(ticker)
            results['analyst_sentiment'] = analyst_sentiment
        
        # 4. 종합 감정 점수 계산
        composite_sentiment = self._calculate_composite_sentiment(results)
        results['composite'] = composite_sentiment
        
        # 5. 신뢰도 계산
        confidence = self._calculate_multi_source_confidence(results)
        results['confidence'] = confidence
        
        return results
    
    async def _analyze_news_sentiment_enhanced(self, ticker: str) -> Dict:
        """향상된 뉴스 감정 분석 (기존 + 네이버 뉴스)"""
        
        # 기존 뉴스 분석
        from sentiment_analyzer import RealSentimentAnalyzer
        base_analyzer = RealSentimentAnalyzer()
        base_sentiment = await base_analyzer.analyze_stock_sentiment(ticker)
        
        # 네이버 뉴스 추가 분석
        from naver_news_collector import NaverNewsCollector
        naver_collector = NaverNewsCollector()
        naver_news = await naver_collector.collect_enhanced_news(ticker, count=30)
        
        # 네이버 뉴스 감정 분석
        naver_sentiments = []
        for news in naver_news[:15]:  # 상위 15개 분석
            try:
                sentiment = await self._analyze_single_news_ai(news, ticker)
                naver_sentiments.append(sentiment)
                await asyncio.sleep(0.3)  # API 제한 고려
            except Exception as e:
                continue
        
        # 통합 뉴스 감정 계산
        if naver_sentiments:
            avg_naver_sentiment = np.mean(naver_sentiments)
            # 기존 감정과 네이버 감정을 가중 평균 (7:3)
            combined_sentiment = (base_sentiment.get('sentiment_score', 0) * 0.7 + 
                                avg_naver_sentiment * 0.3)
        else:
            combined_sentiment = base_sentiment.get('sentiment_score', 0)
        
        return {
            'score': combined_sentiment,
            'label': self._sentiment_to_label(combined_sentiment),
            'base_sentiment': base_sentiment.get('sentiment_score', 0),
            'naver_sentiment': np.mean(naver_sentiments) if naver_sentiments else 0,
            'news_count': len(naver_news),
            'analyzed_count': len(naver_sentiments)
        }
    
    async def _analyze_social_media_sentiment(self, ticker: str) -> Dict:
        """소셜미디어 감정 분석 (모의 데이터)"""
        
        # 실제 구현에서는 Reddit API, Twitter API 등 사용
        import random
        
        social_data = {}
        overall_scores = []
        
        for source in self.social_sources:
            # 모의 소셜미디어 데이터 생성
            sentiment_score = random.uniform(-1, 1)
            mention_count = random.randint(50, 500)
            engagement_rate = random.uniform(0.1, 0.8)
            
            social_data[source] = {
                'sentiment_score': sentiment_score,
                'mention_count': mention_count,
                'engagement_rate': engagement_rate,
                'trending_score': sentiment_score * engagement_rate
            }
            
            # 언급 횟수로 가중치 적용
            weighted_score = sentiment_score * (mention_count / 1000)
            overall_scores.append(weighted_score)
        
        avg_sentiment = np.mean(overall_scores) if overall_scores else 0
        
        return {
            'score': avg_sentiment,
            'label': self._sentiment_to_label(avg_sentiment),
            'sources': social_data,
            'total_mentions': sum(data['mention_count'] for data in social_data.values()),
            'avg_engagement': np.mean([data['engagement_rate'] for data in social_data.values()])
        }
    
    async def _analyze_analyst_reports(self, ticker: str) -> Dict:
        """애널리스트 리포트 분석 (모의 데이터)"""
        
        import random
        
        analyst_data = {}
        ratings = []
        target_prices = []
        
        rating_map = {
            'Strong Buy': 5, 'Buy': 4, 'Hold': 3, 'Sell': 2, 'Strong Sell': 1
        }
        
        for source in self.analyst_sources:
            rating_text = random.choice(list(rating_map.keys()))
            rating_score = rating_map[rating_text]
            target_price = random.uniform(50, 200)
            confidence = random.uniform(0.6, 0.95)
            
            analyst_data[source] = {
                'rating': rating_text,
                'rating_score': rating_score,
                'target_price': target_price,
                'confidence': confidence,
                'report_date': datetime.now() - timedelta(days=random.randint(1, 30))
            }
            
            ratings.append(rating_score)
            target_prices.append(target_price)
        
        avg_rating = np.mean(ratings) if ratings else 3
        avg_target = np.mean(target_prices) if target_prices else 0
        
        # 평점을 -1~1 스케일로 정규화
        normalized_score = (avg_rating - 3) / 2  # 3이 중립(Hold)
        
        return {
            'average_rating': avg_rating,
            'normalized_score': normalized_score,
            'label': self._rating_to_label(avg_rating),
            'average_target_price': avg_target,
            'sources': analyst_data,
            'consensus_strength': 1 - (np.std(ratings) / 2) if ratings else 0
        }
    
    def _calculate_composite_sentiment(self, results: Dict) -> Dict:
        """종합 감정 점수 계산"""
        
        scores = []
        weights = []
        
        # 뉴스 감정 (가중치: 40%)
        if 'news_sentiment' in results:
            scores.append(results['news_sentiment']['score'])
            weights.append(0.4)
        
        # 소셜미디어 감정 (가중치: 30%)
        if 'social_sentiment' in results:
            scores.append(results['social_sentiment']['score'])
            weights.append(0.3)
        
        # 애널리스트 평가 (가중치: 30%)
        if 'analyst_sentiment' in results:
            scores.append(results['analyst_sentiment']['normalized_score'])
            weights.append(0.3)
        
        if scores:
            # 가중 평균 계산
            weighted_score = np.average(scores, weights=weights)
        else:
            weighted_score = 0
        
        return {
            'score': weighted_score,
            'label': self._sentiment_to_label(weighted_score),
            'component_scores': {
                'news': results.get('news_sentiment', {}).get('score', 0),
                'social': results.get('social_sentiment', {}).get('score', 0),
                'analyst': results.get('analyst_sentiment', {}).get('normalized_score', 0)
            }
        }
    
    def _calculate_multi_source_confidence(self, results: Dict) -> float:
        """다중 소스 신뢰도 계산"""
        
        confidence_factors = []
        
        # 뉴스 분석 신뢰도
        if 'news_sentiment' in results:
            news_count = results['news_sentiment'].get('analyzed_count', 0)
            news_confidence = min(0.9, 0.5 + (news_count * 0.03))
            confidence_factors.append(news_confidence)
        
        # 소셜미디어 신뢰도
        if 'social_sentiment' in results:
            total_mentions = results['social_sentiment'].get('total_mentions', 0)
            social_confidence = min(0.8, 0.4 + (total_mentions / 5000))
            confidence_factors.append(social_confidence)
        
        # 애널리스트 신뢰도
        if 'analyst_sentiment' in results:
            consensus_strength = results['analyst_sentiment'].get('consensus_strength', 0)
            analyst_confidence = 0.6 + (consensus_strength * 0.3)
            confidence_factors.append(analyst_confidence)
        
        # 전체 신뢰도는 개별 신뢰도의 평균
        return np.mean(confidence_factors) if confidence_factors else 0.5
    
    async def _analyze_single_news_ai(self, news: Dict, ticker: str) -> float:
        """개별 뉴스 기사 AI 감정 분석"""
        
        prompt = f"""
다음 뉴스 기사가 {ticker} 주식에 미치는 감정을 분석해주세요.

제목: {news.get('title', '')}
내용: {news.get('content', news.get('snippet', ''))[:500]}

-1(매우 부정)부터 +1(매우 긍정) 사이의 숫자로만 답변해주세요.
예: 0.7, -0.3, 0.0
"""
        
        try:
            response = await self.llm.ainvoke(prompt)
            # 숫자 추출
            score_match = re.search(r'-?\d+\.?\d*', response.content)
            if score_match:
                score = float(score_match.group())
                return max(-1, min(1, score))  # -1~1 범위로 제한
            else:
                return 0
        except Exception:
            return 0
    
    def _sentiment_to_label(self, score: float) -> str:
        """감정 점수를 라벨로 변환"""
        if score > 0.3:
            return 'positive'
        elif score < -0.3:
            return 'negative'
        else:
            return 'neutral'
    
    def _rating_to_label(self, rating: float) -> str:
        """평점을 라벨로 변환"""
        if rating >= 4:
            return 'Strong Buy/Buy'
        elif rating >= 3.5:
            return 'Buy/Hold'
        elif rating >= 2.5:
            return 'Hold'
        elif rating >= 2:
            return 'Hold/Sell'
        else:
            return 'Sell/Strong Sell'
