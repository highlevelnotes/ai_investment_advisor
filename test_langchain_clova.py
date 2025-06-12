# test_langchain_clova.py
import asyncio
import os
from agents.sentiment_analyzer import SentimentAnalyzerAgent
from dotenv import load_dotenv

load_dotenv()

async def test_langchain_clova():
    # 환경 변수 확인
    if not os.getenv("NCP_CLOVASTUDIO_API_KEY"):
        print("CLOVASTUDIO_API_KEY 환경 변수를 설정해주세요.")
        return
    
    analyzer = SentimentAnalyzerAgent()
    
    # 테스트 뉴스 데이터
    test_news = [
        {
            'ticker': '005930.KS',
            'title': '삼성전자, 3분기 실적 개선 전망',
            'summary': '반도체 수요 회복으로 매출 증가 예상'
        }
    ]
    
    print("LangChain HyperCLOVA X 감정 분석 테스트 시작...")
    result = await analyzer.analyze_news_sentiment(test_news)
    print("감정 분석 결과:", result)

if __name__ == "__main__":
    asyncio.run(test_langchain_clova())
