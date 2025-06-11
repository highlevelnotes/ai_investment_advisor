# agents/data_collector.py
import yfinance as yf
import aiohttp
import asyncio
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime, timedelta
import logging
from config import Config

logger = logging.getLogger(__name__)

class DataCollectorAgent:
    def __init__(self):
        self.alpha_vantage_key = Config.ALPHA_VANTAGE_KEY
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_real_time_data(self, tickers: List[str]) -> Dict[str, Any]:
        """실시간 시장 데이터 수집"""
        data = {}
        
        try:
            for ticker in tickers:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                data[ticker] = {
                    'current_price': info.get('currentPrice', 0),
                    'volume': info.get('volume', 0),
                    'market_cap': info.get('marketCap', 0),
                    'pe_ratio': info.get('trailingPE'),
                    'dividend_yield': info.get('dividendYield'),
                    'beta': info.get('beta'),
                    'sector': info.get('sector'),
                    'industry': info.get('industry'),
                    'timestamp': datetime.now().isoformat()
                }
                
                # 짧은 지연으로 API 제한 방지
                await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.error(f"실시간 데이터 수집 오류: {e}")
            
        return data
    
    async def get_historical_data(self, tickers: List[str], period: str = '1y') -> Dict[str, pd.DataFrame]:
        """과거 데이터 수집"""
        historical_data = {}
        
        try:
            for ticker in tickers:
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period)
                
                if not hist.empty:
                    historical_data[ticker] = hist
                    
                await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.error(f"과거 데이터 수집 오류: {e}")
            
        return historical_data
    
    async def get_financial_news(self, tickers: List[str]) -> List[Dict[str, str]]:
        """금융 뉴스 수집"""
        news_articles = []
        
        try:
            for ticker in tickers:
                stock = yf.Ticker(ticker)
                news = stock.news
                
                for article in news[:5]:  # 각 종목당 최대 5개 뉴스
                    news_articles.append({
                        'ticker': ticker,
                        'title': article.get('title', ''),
                        'summary': article.get('summary', ''),
                        'url': article.get('link', ''),
                        'published': article.get('providerPublishTime', 0)
                    })
                    
                await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.error(f"뉴스 수집 오류: {e}")
            
        return news_articles
