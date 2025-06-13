# enhanced_news_collector.py
import aiohttp
import asyncio
import requests
from datetime import datetime, timedelta
import yfinance as yf
from bs4 import BeautifulSoup
import feedparser
from typing import List, Dict, Any
import logging
import os
import json

logger = logging.getLogger(__name__)

class EnhancedNewsCollector:
    def __init__(self):
        self.naver_client_id = os.getenv('NAVER_CLIENT_ID')
        self.naver_client_secret = os.getenv('NAVER_CLIENT_SECRET')
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def collect_comprehensive_news(self, ticker: str, max_articles: int = 30) -> List[Dict[str, Any]]:
        """종합 뉴스 수집 (국내외 소스 통합)"""
        all_news = []
        
        try:
            # 병렬 뉴스 수집
            tasks = [
                self._get_naver_news(ticker, 10),
                self._get_yahoo_news(ticker, 8),
                self._get_google_news(ticker, 7),
                self._get_marketwatch_news(ticker, 5)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, list):
                    all_news.extend(result)
            
            # 데이터 품질 검증 및 중복 제거
            validated_news = self._validate_and_deduplicate(all_news)
            
            # 최신순 정렬 후 제한
            sorted_news = sorted(validated_news, 
                               key=lambda x: x.get('published_time', 0), 
                               reverse=True)
            
            return sorted_news[:max_articles]
            
        except Exception as e:
            logger.error(f"종합 뉴스 수집 오류: {e}")
            return []
    
    async def _get_naver_news(self, ticker: str, count: int) -> List[Dict[str, Any]]:
        """네이버 뉴스 API 활용"""
        if not self.naver_client_id or not self.naver_client_secret:
            logger.warning("네이버 API 키가 설정되지 않음")
            return []
        
        try:
            company_name = self._get_company_name(ticker)
            query = f"{company_name} 주식"
            
            url = "https://openapi.naver.com/v1/search/news.json"
            headers = {
                'X-Naver-Client-Id': self.naver_client_id,
                'X-Naver-Client-Secret': self.naver_client_secret
            }
            params = {
                'query': query,
                'display': count,
                'start': 1,
                'sort': 'date'
            }
            
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    naver_news = []
                    
                    for item in data.get('items', []):
                        # HTML 태그 제거
                        title = BeautifulSoup(item['title'], 'html.parser').get_text()
                        description = BeautifulSoup(item['description'], 'html.parser').get_text()
                        
                        # 날짜 파싱
                        pub_date = datetime.strptime(item['pubDate'], '%a, %d %b %Y %H:%M:%S %z')
                        
                        naver_news.append({
                            'title': title,
                            'summary': description,
                            'url': item['link'],
                            'published_time': int(pub_date.timestamp()),
                            'source': 'Naver News',
                            'ticker': ticker,
                            'quality_score': self._calculate_quality_score(title, description)
                        })
                    
                    return naver_news
            
            return []
            
        except Exception as e:
            logger.warning(f"네이버 뉴스 수집 실패: {e}")
            return []
    
    async def _get_yahoo_news(self, ticker: str, count: int) -> List[Dict[str, Any]]:
        """Yahoo Finance 뉴스 수집"""
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
            
            yahoo_news = []
            for article in news[:count]:
                yahoo_news.append({
                    'title': article.get('title', ''),
                    'summary': article.get('summary', ''),
                    'url': article.get('link', ''),
                    'published_time': article.get('providerPublishTime', 0),
                    'source': 'Yahoo Finance',
                    'ticker': ticker,
                    'quality_score': self._calculate_quality_score(
                        article.get('title', ''), 
                        article.get('summary', '')
                    )
                })
            
            return yahoo_news
            
        except Exception as e:
            logger.warning(f"Yahoo 뉴스 수집 실패: {e}")
            return []
    
    async def _get_google_news(self, ticker: str, count: int) -> List[Dict[str, Any]]:
        """Google News RSS 피드"""
        try:
            company_name = self._get_company_name(ticker)
            query = f"{ticker} {company_name} stock"
            rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
            
            async with self.session.get(rss_url) as response:
                if response.status == 200:
                    content = await response.text()
                    feed = feedparser.parse(content)
                    
                    google_news = []
                    for entry in feed.entries[:count]:
                        published_time = 0
                        if hasattr(entry, 'published_parsed'):
                            published_time = int(datetime(*entry.published_parsed[:6]).timestamp())
                        
                        title = entry.get('title', '')
                        summary = entry.get('summary', '')[:200]
                        
                        google_news.append({
                            'title': title,
                            'summary': summary,
                            'url': entry.get('link', ''),
                            'published_time': published_time,
                            'source': 'Google News',
                            'ticker': ticker,
                            'quality_score': self._calculate_quality_score(title, summary)
                        })
                    
                    return google_news
            
            return []
            
        except Exception as e:
            logger.warning(f"Google 뉴스 수집 실패: {e}")
            return []
    
    async def _get_marketwatch_news(self, ticker: str, count: int) -> List[Dict[str, Any]]:
        """MarketWatch 뉴스 수집"""
        try:
            url = f"https://www.marketwatch.com/investing/stock/{ticker.lower()}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    news_items = soup.find_all('div', class_='article__content')
                    
                    marketwatch_news = []
                    for item in news_items[:count]:
                        title_elem = item.find('h3') or item.find('h2')
                        if title_elem:
                            title = title_elem.get_text().strip()
                            
                            marketwatch_news.append({
                                'title': title,
                                'summary': title,
                                'url': '',
                                'published_time': int(datetime.now().timestamp()),
                                'source': 'MarketWatch',
                                'ticker': ticker,
                                'quality_score': self._calculate_quality_score(title, title)
                            })
                    
                    return marketwatch_news
            
            return []
            
        except Exception as e:
            logger.warning(f"MarketWatch 뉴스 수집 실패: {e}")
            return []
    
    def _calculate_quality_score(self, title: str, summary: str) -> float:
        """뉴스 품질 점수 계산"""
        score = 0.5  # 기본 점수
        
        # 제목 길이 (너무 짧거나 길면 감점)
        if 10 <= len(title) <= 100:
            score += 0.2
        
        # 요약 존재 여부
        if summary and len(summary) > 20:
            score += 0.2
        
        # 금융 관련 키워드
        financial_keywords = ['stock', 'shares', 'earnings', 'revenue', 'profit', 'loss', 'market', '주식', '수익', '매출']
        keyword_count = sum(1 for keyword in financial_keywords if keyword.lower() in title.lower() or keyword.lower() in summary.lower())
        score += min(keyword_count * 0.1, 0.3)
        
        return min(score, 1.0)
    
    def _validate_and_deduplicate(self, news_list: List[Dict]) -> List[Dict]:
        """데이터 품질 검증 및 중복 제거"""
        validated_news = []
        seen_titles = set()
        
        for news in news_list:
            # 품질 검증
            if not self._is_valid_news(news):
                continue
            
            # 중복 제거
            title_key = news.get('title', '').lower().strip()
            if title_key and title_key not in seen_titles and len(title_key) > 10:
                seen_titles.add(title_key)
                validated_news.append(news)
        
        return validated_news
    
    def _is_valid_news(self, news: Dict) -> bool:
        """뉴스 유효성 검증"""
        # 필수 필드 확인
        if not news.get('title') or not news.get('source'):
            return False
        
        # 품질 점수 확인
        if news.get('quality_score', 0) < 0.3:
            return False
        
        # 스팸 키워드 필터링
        spam_keywords = ['advertisement', 'sponsored', 'ad', '광고']
        title = news.get('title', '').lower()
        if any(keyword in title for keyword in spam_keywords):
            return False
        
        return True
    
    def _get_company_name(self, ticker: str) -> str:
        """티커에서 회사명 추출"""
        company_names = {
            'AAPL': 'Apple',
            'MSFT': 'Microsoft',
            'GOOGL': 'Google Alphabet',
            'AMZN': 'Amazon',
            'TSLA': 'Tesla',
            'NVDA': 'NVIDIA'
        }
        return company_names.get(ticker, ticker)
