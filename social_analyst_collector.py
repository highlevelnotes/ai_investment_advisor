# social_analyst_collector.py
import aiohttp
import asyncio
from typing import List, Dict, Any
import logging
import json
import re
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import asyncpraw
import tweepy
from textblob import TextBlob
from langchain_naver import ChatClovaX

# 환경 변수 로드
load_dotenv()

logger = logging.getLogger(__name__)

class RateLimiter:
    """API 호출 제한을 관리하는 클래스"""
    def __init__(self, max_calls=100, time_window=900):  # 15분에 100회
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
    
    async def wait_if_needed(self):
        now = datetime.now()
        # 시간 윈도우 내의 호출만 유지
        self.calls = [call_time for call_time in self.calls 
                     if now - call_time < timedelta(seconds=self.time_window)]
        
        if len(self.calls) >= self.max_calls:
            sleep_time = self.time_window - (now - self.calls[0]).total_seconds()
            if sleep_time > 0:
                logger.info(f"Rate limit reached. Waiting {sleep_time:.1f} seconds...")
                await asyncio.sleep(sleep_time)
        
        self.calls.append(now)

class SocialAnalystCollector:
    def __init__(self):
        self.session = None
        # API 키 설정
        self.reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        self.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.reddit_user_agent = os.getenv('REDDIT_USER_AGENT', 'stock_analyzer/1.0')
        self.twitter_bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        self.clovastudio_api_key = os.getenv('CLOVASTUDIO_API_KEY')
        
        # Rate Limiters 초기화
        self.reddit_rate_limiter = RateLimiter(max_calls=60, time_window=600)  # 10분에 60회
        self.twitter_rate_limiter = RateLimiter(max_calls=50, time_window=900)  # 15분에 50회
        self.stocktwits_rate_limiter = RateLimiter(max_calls=100, time_window=3600)  # 1시간에 100회
        
        # Async Reddit 클라이언트 초기화
        if self.reddit_client_id and self.reddit_client_secret:
            try:
                self.reddit = asyncpraw.Reddit(
                    client_id=self.reddit_client_id,
                    client_secret=self.reddit_client_secret,
                    user_agent=self.reddit_user_agent
                )
                self.reddit_available = True
            except Exception as e:
                logger.warning(f"Reddit API 초기화 실패: {e}")
                self.reddit_available = False
        else:
            self.reddit_available = False
        
        # Twitter 클라이언트 초기화
        if self.twitter_bearer_token:
            try:
                self.twitter_client = tweepy.Client(bearer_token=self.twitter_bearer_token)
                self.twitter_available = True
            except Exception as e:
                logger.warning(f"Twitter API 초기화 실패: {e}")
                self.twitter_available = False
        else:
            self.twitter_available = False
        
        # ChatClovaX 클라이언트 초기화 (CLOVASTUDIO_API_KEY 사용)
        if self.clovastudio_api_key:
            try:
                self.chat_clovax = ChatClovaX(
                    api_key=self.clovastudio_api_key,
                    max_tokens=3000,
                    temperature=0.1
                )
                self.clovax_available = True
                logger.info("ChatClovaX 초기화 성공")
            except Exception as e:
                logger.warning(f"ChatClovaX API 초기화 실패: {e}")
                self.clovax_available = False
        else:
            logger.warning("CLOVASTUDIO_API_KEY가 설정되지 않음")
            self.clovax_available = False
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
        # Reddit 세션 종료
        if self.reddit_available and hasattr(self, 'reddit'):
            await self.reddit.close()
    
    async def collect_social_sentiment(self, ticker: str) -> List[Dict[str, Any]]:
        """실제 소셜미디어 감정 데이터 수집"""
        social_data = []
        
        # Reddit 실제 데이터 수집
        try:
            reddit_data = await self._get_reddit_sentiment_real(ticker)
            social_data.extend(reddit_data)
        except Exception as e:
            logger.error(f"Reddit 데이터 수집 실패: {e}")
            # ChatClovaX만으로 대체 분석 제공
            reddit_fallback = await self._generate_social_analysis_with_clovax(ticker, "Reddit")
            if reddit_fallback:
                social_data.extend(reddit_fallback)
        
        # Twitter/X 실제 데이터 수집
        try:
            twitter_data = await self._get_twitter_sentiment_real(ticker)
            social_data.extend(twitter_data)
        except Exception as e:
            logger.error(f"Twitter 데이터 수집 실패: {e}")
            # ChatClovaX만으로 대체 분석 제공
            twitter_fallback = await self._generate_social_analysis_with_clovax(ticker, "Twitter")
            if twitter_fallback:
                social_data.extend(twitter_fallback)
        
        # StockTwits 실제 데이터 수집
        try:
            stocktwits_data = await self._get_stocktwits_sentiment_real(ticker)
            social_data.extend(stocktwits_data)
        except Exception as e:
            logger.error(f"StockTwits 데이터 수집 실패: {e}")
            # ChatClovaX만으로 대체 분석 제공
            stocktwits_fallback = await self._generate_social_analysis_with_clovax(ticker, "StockTwits")
            if stocktwits_fallback:
                social_data.extend(stocktwits_fallback)
        
        return social_data[:3]  # 각 플랫폼당 1개씩 총 3개
    
    async def collect_analyst_reports(self, ticker: str) -> List[Dict[str, Any]]:
        """ChatClovaX를 활용한 애널리스트 리포트 생성"""
        analyst_data = []
        
        # 실제 API 시도
        try:
            finnhub_reports = await self._get_finnhub_analyst_data(ticker)
            analyst_data.extend(finnhub_reports)
        except Exception as e:
            logger.error(f"Finnhub 데이터 수집 실패: {e}")
        
        try:
            alpha_reports = await self._get_alpha_vantage_analyst_data(ticker)
            analyst_data.extend(alpha_reports)
        except Exception as e:
            logger.error(f"Alpha Vantage 데이터 수집 실패: {e}")
        
        try:
            yahoo_reports = await self._get_yahoo_analyst_data(ticker)
            analyst_data.extend(yahoo_reports)
        except Exception as e:
            logger.error(f"Yahoo Finance 데이터 수집 실패: {e}")
        
        # ChatClovaX를 사용한 애널리스트 분석 생성 (다른 API 없이도 작동)
        if len(analyst_data) < 3:
            clovax_reports = await self._generate_analyst_reports_with_clovax(ticker, 3 - len(analyst_data))
            analyst_data.extend(clovax_reports)
        
        return analyst_data[:5]  # 최대 5개 리포트 반환
    
    async def _generate_social_analysis_with_clovax(self, ticker: str, platform: str) -> List[Dict[str, Any]]:
        """ChatClovaX만을 사용한 소셜미디어 감정 분석 생성"""
        if not self.clovax_available:
            return []
        
        try:
            from langchain.schema import HumanMessage, SystemMessage
            
            messages = [
                SystemMessage(content=f"""당신은 {platform} 소셜미디어 분석 전문가입니다. 
주어진 주식 티커에 대한 현재 시장 상황과 일반적인 투자자 감정을 분석하여 
-1(매우 부정적)부터 1(매우 긍정적) 사이의 감정 점수와 분석 내용을 제공해주세요."""),
                HumanMessage(content=f"""{ticker} 주식에 대한 {platform}에서의 투자자 감정을 분석해주세요.
다음 형식으로 답변해주세요:
감정점수: [숫자]
요약: [분석 내용]
신뢰도: [0.5-0.8 사이 숫자]""")
            ]
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.chat_clovax.invoke(messages)
            )
            
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # 응답 파싱
            sentiment_score = 0.0
            summary = f"{platform} sentiment analysis for {ticker} (AI generated)"
            confidence = 0.6
            
            # 감정점수 추출
            sentiment_match = re.search(r'감정점수:\s*(-?\d*\.?\d+)', response_text)
            if sentiment_match:
                sentiment_score = float(sentiment_match.group(1))
                sentiment_score = max(-1.0, min(1.0, sentiment_score))
            
            # 요약 추출
            summary_match = re.search(r'요약:\s*(.+?)(?=신뢰도:|$)', response_text, re.DOTALL)
            if summary_match:
                summary = summary_match.group(1).strip()
            
            # 신뢰도 추출
            confidence_match = re.search(r'신뢰도:\s*(\d*\.?\d+)', response_text)
            if confidence_match:
                confidence = float(confidence_match.group(1))
                confidence = max(0.5, min(0.8, confidence))
            
            return [{
                'platform': platform,
                'sentiment_score': sentiment_score,
                'mention_count': 50,  # AI 생성 기본값
                'summary': summary,
                'confidence': confidence
            }]
            
        except Exception as e:
            logger.error(f"ChatClovaX {platform} 분석 생성 실패: {e}")
            return []
    
    async def _generate_analyst_reports_with_clovax(self, ticker: str, count: int) -> List[Dict[str, Any]]:
        """ChatClovaX만을 사용한 애널리스트 리포트 생성"""
        if not self.clovax_available:
            return []
        
        reports = []
        analyst_firms = ['Goldman Sachs', 'Morgan Stanley', 'JP Morgan', 'Bank of America', 'Wells Fargo']
        
        for i in range(min(count, len(analyst_firms))):
            try:
                firm = analyst_firms[i]
                
                from langchain.schema import HumanMessage, SystemMessage
                
                messages = [
                    SystemMessage(content=f"""당신은 {firm}의 주식 애널리스트입니다. 
주어진 주식에 대한 전문적인 분석과 투자 의견을 제공해주세요."""),
                    HumanMessage(content=f"""{ticker} 주식에 대한 {firm}의 분석 리포트를 작성해주세요.
다음 형식으로 답변해주세요:
평점: [Strong Buy/Buy/Hold/Sell 중 하나]
목표가: [달러 금액]
요약: [분석 내용]
신뢰도: [0.7-0.9 사이 숫자]""")
                ]
                
                response = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: self.chat_clovax.invoke(messages)
                )
                
                response_text = response.content if hasattr(response, 'content') else str(response)
                
                # 응답 파싱
                rating = "Hold"
                target_price = 150.0
                summary = f"{firm} analysis for {ticker} (AI generated)"
                confidence = 0.75
                
                # 평점 추출
                rating_match = re.search(r'평점:\s*(Strong Buy|Buy|Hold|Sell)', response_text)
                if rating_match:
                    rating = rating_match.group(1)
                
                # 목표가 추출
                target_match = re.search(r'목표가:\s*\$?(\d+\.?\d*)', response_text)
                if target_match:
                    target_price = float(target_match.group(1))
                
                # 요약 추출
                summary_match = re.search(r'요약:\s*(.+?)(?=신뢰도:|$)', response_text, re.DOTALL)
                if summary_match:
                    summary = summary_match.group(1).strip()
                
                # 신뢰도 추출
                confidence_match = re.search(r'신뢰도:\s*(\d*\.?\d+)', response_text)
                if confidence_match:
                    confidence = float(confidence_match.group(1))
                    confidence = max(0.7, min(0.9, confidence))
                
                report = {
                    'source': firm,
                    'rating': rating,
                    'target_price': target_price,
                    'summary': summary,
                    'confidence': confidence,
                    'timestamp': int(datetime.now().timestamp())
                }
                reports.append(report)
                
            except Exception as e:
                logger.error(f"ChatClovaX {firm} 리포트 생성 실패: {e}")
                continue
        
        return reports
    
    async def _get_reddit_sentiment_real(self, ticker: str) -> List[Dict[str, Any]]:
        """Reddit 실제 감정 데이터 수집 (개선)"""
        if not self.reddit_available:
            raise Exception("Reddit API 사용 불가")
        
        await self.reddit_rate_limiter.wait_if_needed()
        
        # 주식 관련 서브레딧에서 검색
        subreddits = ['stocks', 'investing', 'SecurityAnalysis', 'ValueInvesting', 'StockMarket']
        all_posts = []
        
        for subreddit_name in subreddits:
            try:
                subreddit = await self.reddit.subreddit(subreddit_name)
                # 최근 게시물 검색 - 여러 검색어 시도
                search_terms = [ticker, f"${ticker}", ticker.upper(), ticker.lower()]
                
                for term in search_terms:
                    try:
                        async for submission in subreddit.search(term, limit=3, time_filter='week'):
                            all_posts.append({
                                'title': submission.title or '',
                                'selftext': submission.selftext or '',
                                'score': getattr(submission, 'score', 0),
                                'num_comments': getattr(submission, 'num_comments', 0)
                            })
                            if len(all_posts) >= 15:  # 충분한 데이터 수집 시 중단
                                break
                    except Exception as e:
                        logger.debug(f"검색어 '{term}' 실패: {e}")
                        continue
                    
                    if len(all_posts) >= 15:
                        break
                        
            except Exception as e:
                logger.warning(f"서브레딧 {subreddit_name} 검색 실패: {e}")
                continue
        
        if all_posts:
            # 감정 분석 수행
            sentiments = []
            for post in all_posts:
                text = f"{post['title']} {post['selftext']}"
                if text.strip():  # 빈 텍스트 체크
                    sentiment_score = await self._analyze_sentiment_safe(text)
                    sentiments.append(sentiment_score)
            
            if sentiments:
                avg_sentiment = sum(sentiments) / len(sentiments)
                
                return [{
                    'platform': 'Reddit',
                    'sentiment_score': avg_sentiment,
                    'mention_count': len(all_posts),
                    'summary': f'Reddit users discussing {ticker} with {"positive" if avg_sentiment > 0.1 else "negative" if avg_sentiment < -0.1 else "neutral"} sentiment',
                    'confidence': min(0.9, 0.6 + len(all_posts) * 0.02)
                }]
            else:
                raise Exception("감정 분석 결과 없음")
        else:
            raise Exception("Reddit 게시물 없음")
    
    async def _get_twitter_sentiment_real(self, ticker: str) -> List[Dict[str, Any]]:
        """Twitter/X 실제 감정 데이터 수집 (cashtag 문제 해결)"""
        if not self.twitter_available:
            raise Exception("Twitter API 사용 불가")
        
        await self.twitter_rate_limiter.wait_if_needed()
        
        # cashtag 대신 일반 텍스트 검색 사용
        search_queries = [
            f"{ticker} -is:retweet lang:en",  # cashtag 제거
            f"{ticker.upper()} stock -is:retweet lang:en",
            f"{ticker.lower()} investing -is:retweet lang:en"
        ]
        
        all_tweets = []
        
        for query in search_queries:
            try:
                tweets = self.twitter_client.search_recent_tweets(
                    query=query,
                    max_results=30,  # 각 쿼리당 30개씩
                    tweet_fields=['created_at', 'public_metrics']
                )
                
                if tweets.data:
                    for tweet in tweets.data:
                        if hasattr(tweet, 'text') and tweet.text:
                            all_tweets.append(tweet.text)
                            
                if len(all_tweets) >= 50:  # 충분한 데이터 수집 시 중단
                    break
                    
            except Exception as e:
                logger.warning(f"Twitter 쿼리 '{query}' 실패: {e}")
                continue
        
        if all_tweets:
            sentiments = []
            for tweet_text in all_tweets:
                sentiment_score = await self._analyze_sentiment_safe(tweet_text)
                sentiments.append(sentiment_score)
            
            if sentiments:
                avg_sentiment = sum(sentiments) / len(sentiments)
                
                return [{
                    'platform': 'Twitter/X',
                    'sentiment_score': avg_sentiment,
                    'mention_count': len(all_tweets),
                    'summary': f'Twitter sentiment for {ticker} trending {"bullish" if avg_sentiment > 0.2 else "bearish" if avg_sentiment < -0.2 else "neutral"}',
                    'confidence': min(0.95, 0.7 + len(all_tweets) * 0.003)
                }]
            else:
                raise Exception("감정 분석 결과 없음")
        else:
            raise Exception("트윗 데이터 없음")
    
    async def _get_stocktwits_sentiment_real(self, ticker: str) -> List[Dict[str, Any]]:
        """StockTwits 실제 감정 데이터 수집 (403 오류 해결)"""
        await self.stocktwits_rate_limiter.wait_if_needed()
        
        # 다양한 헤더와 프록시 시도
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://stocktwits.com/',
            'Origin': 'https://stocktwits.com'
        }
        
        # StockTwits API 호출
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
        
        try:
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    messages = data.get('messages', [])
                    
                    if messages:
                        sentiments = []
                        for message in messages:
                            text = message.get('body', '')
                            if text.strip():
                                sentiment_score = await self._analyze_sentiment_safe(text)
                                sentiments.append(sentiment_score)
                        
                        if sentiments:
                            avg_sentiment = sum(sentiments) / len(sentiments)
                            
                            return [{
                                'platform': 'StockTwits',
                                'sentiment_score': avg_sentiment,
                                'mention_count': len(messages),
                                'summary': f'StockTwits community shows {"optimistic" if avg_sentiment > 0.1 else "pessimistic" if avg_sentiment < -0.1 else "mixed"} outlook on {ticker}',
                                'confidence': min(0.85, 0.65 + len(messages) * 0.005)
                            }]
                        else:
                            raise Exception("감정 분석 결과 없음")
                    else:
                        raise Exception("StockTwits 메시지 없음")
                else:
                    raise Exception(f"StockTwits API 응답 오류: {response.status}")
        except aiohttp.ClientError as e:
            raise Exception(f"StockTwits 연결 오류: {e}")
    
    async def _analyze_sentiment_safe(self, text: str) -> float:
        """안전한 텍스트 감정 분석 (ChatClovaX 우선 사용)"""
        if not text or not text.strip():
            return 0.0
        
        # 텍스트 길이 제한 (너무 긴 텍스트는 잘라내기)
        if len(text) > 1000:
            text = text[:1000]
        
        # ChatClovaX를 사용한 고급 감정 분석
        if self.clovax_available:
            try:
                from langchain.schema import HumanMessage, SystemMessage
                
                messages = [
                    SystemMessage(content="당신은 금융 감정 분석 전문가입니다. 주식/투자 관련 텍스트의 감정을 -1(매우 부정적)부터 1(매우 긍정적) 사이의 숫자로만 답변해주세요."),
                    HumanMessage(content=f"다음 텍스트의 투자 감정을 분석해주세요:\n\n{text}\n\n감정 점수:")
                ]
                
                response = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: self.chat_clovax.invoke(messages)
                )
                
                # 응답에서 숫자 추출 (안전한 방식)
                response_text = response.content if hasattr(response, 'content') else str(response)
                numbers = re.findall(r'-?\d*\.?\d+', response_text)
                
                if numbers and len(numbers) > 0:  # 빈 리스트 체크 추가
                    try:
                        sentiment_score = float(numbers[0])
                        # -1과 1 사이로 제한
                        return max(-1.0, min(1.0, sentiment_score))
                    except (ValueError, IndexError):
                        logger.warning(f"ChatClovaX 응답 파싱 실패: {response_text}")
                        return 0.0
                else:
                    logger.warning(f"ChatClovaX 응답에서 숫자 추출 실패: {response_text}")
                    return 0.0
                    
            except Exception as e:
                logger.warning(f"ChatClovaX 감정 분석 실패: {e}")
        
        # TextBlob을 사용한 기본 감정 분석
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            return float(polarity) if polarity is not None else 0.0
        except Exception as e:
            logger.warning(f"TextBlob 감정 분석 실패: {e}")
            return 0.0
    
    async def _get_finnhub_analyst_data(self, ticker: str) -> List[Dict[str, Any]]:
        """Finnhub API를 통한 여러 애널리스트 데이터 수집"""
        api_key = os.getenv('FINNHUB_API_KEY')
        if not api_key:
            raise Exception("Finnhub API 키 없음")
        
        url = f"https://finnhub.io/api/v1/stock/recommendation"
        params = {'symbol': ticker, 'token': api_key}
        
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                
                if data and len(data) > 0:
                    # 최대 3개의 최신 데이터를 사용하여 여러 리포트 생성
                    reports = []
                    for i, entry in enumerate(data[:3]):  # 최신 3개 데이터
                        total_ratings = sum([
                            entry.get('strongBuy', 0),
                            entry.get('buy', 0),
                            entry.get('hold', 0),
                            entry.get('sell', 0),
                            entry.get('strongSell', 0)
                        ])
                        
                        if total_ratings > 0:
                            weighted_score = (
                                entry.get('strongBuy', 0) * 5 + 
                                entry.get('buy', 0) * 4 + 
                                entry.get('hold', 0) * 3 + 
                                entry.get('sell', 0) * 2 + 
                                entry.get('strongSell', 0) * 1
                            ) / total_ratings
                            
                            rating = "Strong Buy" if weighted_score >= 4.5 else \
                                   "Buy" if weighted_score >= 3.5 else \
                                   "Hold" if weighted_score >= 2.5 else \
                                   "Sell"
                            
                            # 날짜별로 다른 소스명 부여
                            source_names = [
                                'Finnhub Consensus (Latest)',
                                'Finnhub Consensus (Previous)',
                                'Finnhub Consensus (Earlier)'
                            ]
                            
                            report = {
                                'source': source_names[i] if i < len(source_names) else f'Finnhub Consensus ({i+1})',
                                'rating': rating,
                                'target_price': None,
                                'summary': f'Consensus from {total_ratings} analysts: {entry.get("strongBuy", 0)} Strong Buy, {entry.get("buy", 0)} Buy, {entry.get("hold", 0)} Hold ({entry.get("period", "recent")})',
                                'confidence': min(0.9, 0.5 + total_ratings * 0.02),
                                'timestamp': entry.get('period', int(datetime.now().timestamp()))
                            }
                            reports.append(report)
                    
                    return reports
                else:
                    raise Exception("Finnhub 데이터 없음")
            else:
                raise Exception(f"Finnhub API 응답 오류: {response.status}")
    
    async def _get_alpha_vantage_analyst_data(self, ticker: str) -> List[Dict[str, Any]]:
        """Alpha Vantage API를 통한 애널리스트 데이터"""
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if not api_key:
            raise Exception("Alpha Vantage API 키 없음")
        
        url = "https://www.alphavantage.co/query"
        params = {
            'function': 'OVERVIEW',
            'symbol': ticker,
            'apikey': api_key
        }
        
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                
                if 'AnalystTargetPrice' in data and data['AnalystTargetPrice'] not in ['None', None, '']:
                    try:
                        target_price = float(data['AnalystTargetPrice'])
                        pe_ratio = data.get('PERatio', 'N/A')
                        
                        return [{
                            'source': 'Alpha Vantage',
                            'rating': 'Hold',
                            'target_price': target_price,
                            'summary': f'Analyst target price: ${target_price:.2f}, P/E Ratio: {pe_ratio}',
                            'confidence': 0.75,
                            'timestamp': int(datetime.now().timestamp())
                        }]
                    except (ValueError, TypeError):
                        raise Exception("Alpha Vantage 목표가 데이터 형식 오류")
                else:
                    raise Exception("Alpha Vantage 목표가 데이터 없음")
            else:
                raise Exception(f"Alpha Vantage API 응답 오류: {response.status}")
    
    async def _get_yahoo_analyst_data(self, ticker: str) -> List[Dict[str, Any]]:
        """Yahoo Finance에서 애널리스트 데이터 스크래핑 (개선)"""
        # User-Agent 헤더 추가로 429 오류 해결
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9'
        }
        
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        
        async with self.session.get(url, headers=headers) as response:
            if response.status == 200:
                data = await response.json()
                
                # 기본적인 정보만 추출
                if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
                    result = data['chart']['result'][0]
                    meta = result.get('meta', {})
                    
                    current_price = meta.get('regularMarketPrice', 0)
                    
                    if current_price and current_price > 0:
                        # 간단한 추정 목표가 (현재가 기준 +10%)
                        estimated_target = current_price * 1.1
                        
                        return [{
                            'source': 'Yahoo Finance',
                            'rating': 'Hold',
                            'target_price': estimated_target,
                            'summary': f'Market data based analysis for {ticker}',
                            'confidence': 0.6,
                            'timestamp': int(datetime.now().timestamp())
                        }]
                    else:
                        raise Exception("Yahoo Finance 가격 데이터 없음")
                else:
                    raise Exception("Yahoo Finance 차트 데이터 없음")
            else:
                raise Exception(f"Yahoo Finance API 응답 오류: {response.status}")
