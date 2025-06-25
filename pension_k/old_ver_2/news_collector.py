# news_collector.py
import aiohttp
import asyncio
from datetime import datetime, timedelta
import yfinance as yf
from bs4 import BeautifulSoup
import feedparser
import requests
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class NewsCollector:
    def __init__(self):
        self.session = None
        # 무료 뉴스 소스들
        self.news_sources = {
            'yahoo_finance': self._get_yahoo_news,
            'google_news': self._get_google_news,
            'marketwatch': self._get_marketwatch_news,
            'seeking_alpha': self._get_seeking_alpha_news
        }
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def collect_news(self, ticker: str, max_articles: int = 20) -> List[Dict[str, Any]]:
        """종목 관련 뉴스 수집"""
        all_news = []
        
        try:
            # 각 뉴스 소스에서 병렬로 수집
            tasks = []
            for source_name, source_func in self.news_sources.items():
                tasks.append(self._safe_collect(source_func, ticker, source_name))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 결과 통합
            for result in results:
                if isinstance(result, list):
                    all_news.extend(result)
            
            # 중복 제거 및 정렬
            unique_news = self._deduplicate_news(all_news)
            
            # 최신순 정렬 후 제한
            sorted_news = sorted(unique_news, 
                               key=lambda x: x.get('published_time', 0), 
                               reverse=True)
            
            return sorted_news[:max_articles]
            
        except Exception as e:
            logger.error(f"뉴스 수집 오류: {e}")
            return []
    
    async def _safe_collect(self, func, ticker: str, source_name: str) -> List[Dict]:
        """안전한 뉴스 수집 (에러 처리)"""
        try:
            return await func(ticker)
        except Exception as e:
            logger.warning(f"{source_name} 뉴스 수집 실패: {e}")
            return []
    
    async def _get_yahoo_news(self, ticker: str) -> List[Dict[str, Any]]:
        """Yahoo Finance 뉴스 수집"""
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
            
            yahoo_news = []
            for article in news[:10]:  # 최대 10개
                yahoo_news.append({
                    'title': article.get('title', ''),
                    'summary': article.get('summary', ''),
                    'url': article.get('link', ''),
                    'published_time': article.get('providerPublishTime', 0),
                    'source': 'Yahoo Finance',
                    'ticker': ticker
                })
            
            return yahoo_news
            
        except Exception as e:
            logger.warning(f"Yahoo 뉴스 수집 실패: {e}")
            return []
    
    async def _get_google_news(self, ticker: str) -> List[Dict[str, Any]]:
        """Google News RSS 피드에서 뉴스 수집"""
        try:
            # Google News RSS 피드 URL
            company_name = self._get_company_name(ticker)
            query = f"{ticker} {company_name} stock"
            rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
            
            async with self.session.get(rss_url) as response:
                if response.status == 200:
                    content = await response.text()
                    feed = feedparser.parse(content)
                    
                    google_news = []
                    for entry in feed.entries[:8]:  # 최대 8개
                        # 발행 시간 파싱
                        published_time = 0
                        if hasattr(entry, 'published_parsed'):
                            published_time = int(datetime(*entry.published_parsed[:6]).timestamp())
                        
                        google_news.append({
                            'title': entry.get('title', ''),
                            'summary': entry.get('summary', '')[:200],  # 요약 제한
                            'url': entry.get('link', ''),
                            'published_time': published_time,
                            'source': 'Google News',
                            'ticker': ticker
                        })
                    
                    return google_news
            
            return []
            
        except Exception as e:
            logger.warning(f"Google 뉴스 수집 실패: {e}")
            return []
    
    async def _get_marketwatch_news(self, ticker: str) -> List[Dict[str, Any]]:
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
                    
                    # MarketWatch 뉴스 섹션 찾기
                    news_items = soup.find_all('div', class_='article__content')
                    
                    marketwatch_news = []
                    for item in news_items[:5]:  # 최대 5개
                        title_elem = item.find('h3') or item.find('h2')
                        if title_elem:
                            title = title_elem.get_text().strip()
                            
                            marketwatch_news.append({
                                'title': title,
                                'summary': title,  # 제목을 요약으로 사용
                                'url': '',
                                'published_time': int(datetime.now().timestamp()),
                                'source': 'MarketWatch',
                                'ticker': ticker
                            })
                    
                    return marketwatch_news
            
            return []
            
        except Exception as e:
            logger.warning(f"MarketWatch 뉴스 수집 실패: {e}")
            return []
    
    async def _get_seeking_alpha_news(self, ticker: str) -> List[Dict[str, Any]]:
        """Seeking Alpha RSS 피드에서 뉴스 수집"""
        try:
            rss_url = f"https://seekingalpha.com/api/sa/combined/{ticker}.xml"
            
            async with self.session.get(rss_url) as response:
                if response.status == 200:
                    content = await response.text()
                    feed = feedparser.parse(content)
                    
                    seeking_alpha_news = []
                    for entry in feed.entries[:5]:  # 최대 5개
                        published_time = 0
                        if hasattr(entry, 'published_parsed'):
                            published_time = int(datetime(*entry.published_parsed[:6]).timestamp())
                        
                        seeking_alpha_news.append({
                            'title': entry.get('title', ''),
                            'summary': entry.get('summary', '')[:200],
                            'url': entry.get('link', ''),
                            'published_time': published_time,
                            'source': 'Seeking Alpha',
                            'ticker': ticker
                        })
                    
                    return seeking_alpha_news
            
            return []
            
        except Exception as e:
            logger.warning(f"Seeking Alpha 뉴스 수집 실패: {e}")
            return []
    
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
    
    def _deduplicate_news(self, news_list: List[Dict]) -> List[Dict]:
        """중복 뉴스 제거"""
        seen_titles = set()
        unique_news = []
        
        for news in news_list:
            title = news.get('title', '').lower().strip()
            if title and title not in seen_titles and len(title) > 10:
                seen_titles.add(title)
                unique_news.append(news)
        
        return unique_news
