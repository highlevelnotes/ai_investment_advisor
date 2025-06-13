import aiohttp
import asyncio
from bs4 import BeautifulSoup
from typing import List, Dict
import re
from datetime import datetime
import random

class NaverNewsCollector:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36"
        }
        self.base_url = "https://search.naver.com/search.naver"
    
    async def collect_enhanced_news(self, ticker: str, count: int = 30) -> List[Dict]:
        """향상된 네이버 뉴스 수집 (30개)"""
        
        # 한국 주식 티커를 검색어로 변환
        search_queries = [
            f"{ticker} 주식",
            f"{ticker} 투자",
            f"{ticker} 실적",
            f"{ticker} 전망"
        ]
        
        all_news = []
        
        for query in search_queries:
            try:
                news_batch = await self._collect_news_batch(query, count // len(search_queries))
                all_news.extend(news_batch)
                await asyncio.sleep(1)  # 요청 간격 조절
            except Exception as e:
                continue
        
        # 중복 제거 및 정렬
        unique_news = self._remove_duplicates(all_news)
        return unique_news[:count]
    
    async def _collect_news_batch(self, query: str, count: int) -> List[Dict]:
        """뉴스 배치 수집"""
        
        params = {
            "query": query,
            "where": "news",
            "start": 1,
            "display": count,
            "sort": "date"  # 최신순 정렬
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(self.base_url, params=params, headers=self.headers) as response:
                    html = await response.text()
                    return self._parse_news_html(html, query)
            except Exception as e:
                return []
    
    def _parse_news_html(self, html: str, query: str) -> List[Dict]:
        """HTML에서 뉴스 정보 파싱"""
        
        soup = BeautifulSoup(html, "html.parser")
        news_articles = []
        
        # 네이버 뉴스 구조에 맞는 셀렉터 사용
        news_items = soup.select(".list_news .bx")
        
        for item in news_items:
            try:
                # 제목 추출
                title_elem = item.select_one(".news_tit")
                title = title_elem.text.strip() if title_elem else ""
                link = title_elem.get("href", "") if title_elem else ""
                
                # 요약 추출
                snippet_elem = item.select_one(".news_dsc")
                snippet = snippet_elem.text.strip() if snippet_elem else ""
                
                # 언론사 추출
                press_elem = item.select_one(".info.press")
                press_name = press_elem.text.strip() if press_elem else ""
                
                # 날짜 추출
                date_elem = item.select_one("span.info")
                news_date = date_elem.text.strip() if date_elem else ""
                
                # 추가 메타데이터
                article_data = {
                    "title": title,
                    "link": link,
                    "snippet": snippet,
                    "content": snippet,  # 전체 내용은 별도 크롤링 필요
                    "press_name": press_name,
                    "news_date": news_date,
                    "source": "Naver",
                    "search_query": query,
                    "relevance_score": self._calculate_relevance(title + snippet, query),
                    "collected_at": datetime.now().isoformat()
                }
                
                if title and snippet:  # 유효한 뉴스만 추가
                    news_articles.append(article_data)
                    
            except Exception as e:
                continue
        
        return news_articles
    
    def _calculate_relevance(self, text: str, query: str) -> float:
        """뉴스와 검색어의 관련성 점수 계산"""
        
        # 간단한 키워드 매칭 기반 관련성 점수
        query_words = query.split()
        text_lower = text.lower()
        
        matches = 0
        for word in query_words:
            if word.lower() in text_lower:
                matches += 1
        
        return matches / len(query_words) if query_words else 0
    
    def _remove_duplicates(self, news_list: List[Dict]) -> List[Dict]:
        """중복 뉴스 제거"""
        
        seen_titles = set()
        unique_news = []
        
        for news in news_list:
            title = news.get('title', '')
            # 제목의 첫 30자로 중복 판단
            title_key = title[:30]
            
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_news.append(news)
        
        # 관련성 점수로 정렬
        unique_news.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return unique_news
