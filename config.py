# config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # HyperCLOVA X API 설정
    NCP_CLOVASTUDIO_API_KEY = os.getenv('NCP_CLOVASTUDIO_API_KEY')
    NCP_APIGW_API_KEY = os.getenv('NCP_APIGW_API_KEY')
    CLOVASTUDIO_HOST = 'https://clovastudio.stream.ntruss.com'
    
    # 다른 API 키들
    ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY')
    FINNHUB_TOKEN = os.getenv('FINNHUB_TOKEN')
    
    # 기본 설정
    DEFAULT_TICKERS = ['005930.KS', '000660.KS', '035420.KS', '051910.KS', '207940.KS']  # 한국 주식
    CACHE_DURATION = 300
    MAX_NEWS_ARTICLES = 20
