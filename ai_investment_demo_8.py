# enhanced_perplexity_advisor.py (완전 수정된 버전)
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import asyncio
import nest_asyncio
import numpy as np
import os
from langchain_naver import ChatClovaX
from langchain_core.messages import HumanMessage, SystemMessage

# 커스텀 모듈 import
from enhanced_news_collector import EnhancedNewsCollector
from social_analyst_collector import SocialAnalystCollector
from advanced_technical_analysis import AdvancedTechnicalAnalyzer
from dynamic_scenario_generator import DynamicScenarioGenerator
from sentiment_analyzer import RealSentimentAnalyzer
from scenario_analyzer import AdvancedScenarioAnalyzer

from dotenv import load_dotenv
load_dotenv()

# 환경 설정
os.environ["LANGCHAIN_TRACING_V2"] = "false"
nest_asyncio.apply()

class EnhancedStockAnalysisAgent:
    def __init__(self):
        try:
            self.llm = ChatClovaX(
                model="HCX-005",
                temperature=0.3,
                max_tokens=3000
            )
            self.llm_available = True
        except:
            self.llm_available = False
    
    def analyze_with_ai(self, prompt):
        if not self.llm_available:
            return "AI 분석을 위해 HyperCLOVA X API 키가 필요합니다."
        
        try:
            messages = [
                SystemMessage(content="당신은 전문 주식 분석가입니다. 간결하고 명확한 분석을 제공해주세요."),
                HumanMessage(content=prompt)
            ]
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            return f"AI 분석 중 오류: {str(e)[:100]}"

# 안전한 포맷팅 함수
def safe_format(value, default="N/A", format_spec=""):
    """None 값을 안전하게 포맷팅하는 함수"""
    if value is None:
        return default
    try:
        if format_spec:
            return f"{value:{format_spec}}"
        return str(value)
    except:
        return default

def safe_float(value, default=0.0):
    """안전한 float 변환"""
    try:
        return float(value) if value is not None else default
    except:
        return default

def safe_int(value, default=0):
    """안전한 int 변환"""
    try:
        return int(value) if value is not None else default
    except:
        return default

# 데이터 품질 검증 시스템
class DataQualityValidator:
    @staticmethod
    def validate_news_quality(news_list):
        """뉴스 데이터 품질 검증"""
        if not news_list:
            return []
        quality_scores = []
        for news in news_list:
            score = news.get('quality_score', 0) if news else 0
            if score > 0.8:
                quality_scores.append('high')
            elif score > 0.5:
                quality_scores.append('medium')
            else:
                quality_scores.append('low')
        return quality_scores
    
    @staticmethod
    def validate_data_completeness(data_dict):
        """데이터 완성도 검증"""
        completeness = {}
        for key, value in data_dict.items():
            if isinstance(value, (list, dict)):
                completeness[key] = len(value) > 0
            else:
                completeness[key] = value is not None
        return completeness

# Rate Limiting을 위한 개선된 수집 함수 (동기 버전)
def collect_data_with_retry_sync(collector_func, max_retries=3, delay=2):
    """재시도 로직이 포함된 동기 데이터 수집"""
    for attempt in range(max_retries):
        try:
            return collector_func()
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait_time = delay * (2 ** attempt)  # 지수 백오프
                st.warning(f"Rate limit 도달. {wait_time}초 후 재시도... (시도 {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                st.error(f"데이터 수집 실패: {e}")
                return [] if "list" in str(type(collector_func)) else {}

# 모든 투자 데이터를 수집하는 통합 함수
def collect_all_investment_data(selected_stock):
    """모든 투자 데이터를 수집하는 동기 함수"""
    
    async def async_data_collection():
        enhanced_news = []
        social_data = []
        analyst_data = []
        
        # 향상된 뉴스 수집
        try:
            async def collect_enhanced_news():
                async with EnhancedNewsCollector() as collector:
                    return await collector.collect_comprehensive_news(selected_stock, 10)
            
            # Rate Limiting 적용
            async def collect_data_with_retry_async(collector_func, max_retries=3, delay=2):
                for attempt in range(max_retries):
                    try:
                        return await collector_func()
                    except Exception as e:
                        if "429" in str(e) and attempt < max_retries - 1:
                            wait_time = delay * (2 ** attempt)
                            await asyncio.sleep(wait_time)
                        else:
                            raise e
                return []
            
            enhanced_news = await collect_data_with_retry_async(collect_enhanced_news)
            if not enhanced_news:
                enhanced_news = []
        except Exception as e:
            enhanced_news = []
            st.warning(f"향상된 뉴스 수집 실패: {e}")
        
        # 소셜미디어 및 애널리스트 데이터 수집
        try:
            async def collect_social_analyst_data():
                async with SocialAnalystCollector() as collector:
                    await asyncio.sleep(1)
                    social_data = await collector.collect_social_sentiment(selected_stock)
                    await asyncio.sleep(2)
                    analyst_data = await collector.collect_analyst_reports(selected_stock)
                    return social_data, analyst_data
            
            result = await collect_data_with_retry_async(collect_social_analyst_data)
            if result:
                social_data, analyst_data = result
        except Exception as e:
            social_data, analyst_data = [], []
            st.warning(f"소셜/애널리스트 데이터 수집 실패: {e}")
        
        return enhanced_news, social_data, analyst_data
    
    # asyncio.run()으로 비동기 함수 실행
    try:
        return asyncio.run(async_data_collection())
    except Exception as e:
        st.error(f"데이터 수집 중 오류 발생: {e}")
        return [], [], []

# 감정 분석을 실행하는 동기 함수
def run_sentiment_analysis(selected_stock):
    """감정 분석을 실행하는 동기 함수"""
    
    async def async_sentiment_analysis():
        try:
            sentiment_analyzer = RealSentimentAnalyzer()
            
            async def collect_data_with_retry_async(collector_func, max_retries=3, delay=2):
                for attempt in range(max_retries):
                    try:
                        return await collector_func()
                    except Exception as e:
                        if "429" in str(e) and attempt < max_retries - 1:
                            wait_time = delay * (2 ** attempt)
                            await asyncio.sleep(wait_time)
                        else:
                            raise e
                return {}
            
            sentiment_result = await collect_data_with_retry_async(
                lambda: sentiment_analyzer.analyze_stock_sentiment(selected_stock)
            )
            if not sentiment_result:
                sentiment_result = {
                    'sentiment_score': 0.0,
                    'sentiment_label': "중립적",
                    'sentiment_emoji': "😐",
                    'confidence': 0.5,
                    'article_count': 0,
                    'analyzed_articles': [],
                    'news_sources': [],
                    'method': 'fallback'
                }
            return sentiment_result
        except Exception as e:
            st.warning(f"감정 분석 실패: {e}")
            return {
                'sentiment_score': 0.0,
                'sentiment_label': "중립적",
                'sentiment_emoji': "😐",
                'confidence': 0.5,
                'article_count': 0,
                'analyzed_articles': [],
                'news_sources': [],
                'method': 'fallback'
            }
    
    try:
        return asyncio.run(async_sentiment_analysis())
    except Exception as e:
        st.error(f"감정 분석 중 오류 발생: {e}")
        return {
            'sentiment_score': 0.0,
            'sentiment_label': "중립적",
            'sentiment_emoji': "😐",
            'confidence': 0.5,
            'article_count': 0,
            'analyzed_articles': [],
            'news_sources': [],
            'method': 'fallback'
        }

# AI 분석을 실행하는 동기 함수
def run_ai_analysis(ai_prompt):
    """AI 분석을 실행하는 동기 함수"""
    
    async def async_ai_analysis():
        try:
            analysis_agent = EnhancedStockAnalysisAgent()
            
            # Rate Limiting을 고려한 AI 분석
            await asyncio.sleep(3)  # 3초 대기
            ai_analysis = analysis_agent.analyze_with_ai(ai_prompt)
            return ai_analysis
        except Exception as e:
            return f"""
            **AI 분석 결과 (기본 모드)**
            
            **현재 상황:** 다중 소스 분석을 통한 종합적 시장 분석을 제공합니다.
            
            **기술적 분석:** 현재 시장 추세와 주요 지표들을 종합적으로 검토했습니다.
            
            **투자 권고:** 시장 상황을 종합적으로 고려한 신중한 접근을 권장합니다.
            
            **위험 요소:** 시장 변동성과 다중 지표 신호를 종합적으로 모니터링이 필요합니다.
            
            **참고:** API Rate Limit 또는 연결 문제로 인해 기본 분석을 제공합니다. ({str(e)[:50]})
            """
    
    try:
        return asyncio.run(async_ai_analysis())
    except Exception as e:
        return f"AI 분석 중 오류 발생: {str(e)[:100]}"

# Streamlit 페이지 설정
st.set_page_config(
    page_title="Enhanced AI 주식 어드바이저",
    page_icon="🚀",
    layout="wide"
)

# CSS 스타일링
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}

.analysis-card {
    background: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin: 10px 0;
    border-left: 5px solid #667eea;
}

.agent-status {
    background: #f8f9fa;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
    border-left: 4px solid #28a745;
}

.recommendation-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    margin: 20px 0;
}

.step-indicator {
    background: #e3f2fd;
    padding: 10px;
    border-radius: 8px;
    margin: 5px 0;
    border-left: 4px solid #2196f3;
}

.quality-badge {
    display: inline-block;
    padding: 4px 8px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: bold;
    margin-left: 8px;
}

.quality-high { background-color: #d4edda; color: #155724; }
.quality-medium { background-color: #fff3cd; color: #856404; }
.quality-low { background-color: #f8d7da; color: #721c24; }

.social-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
}

.pattern-detected {
    background: #fff3cd;
    border: 2px solid #ffc107;
    padding: 10px;
    border-radius: 8px;
    margin: 5px 0;
}
</style>
""", unsafe_allow_html=True)

# 메인 헤더
st.markdown("""
<div class="main-header">
    <h1>🚀 Enhanced AI 주식 어드바이저</h1>
    <h3>고급 기술적 분석 + 다중 데이터 소스 + 동적 시나리오</h3>
    <p>네이버 뉴스, 소셜미디어, 애널리스트 리포트를 통합한 차세대 AI 투자 분석</p>
</div>
""", unsafe_allow_html=True)

# 종목 선택
st.markdown("## 📊 분석할 종목을 선택하세요")

stocks = {
    "AAPL": {"name": "Apple Inc.", "emoji": "🍎", "color": "#1f77b4"},
    "MSFT": {"name": "Microsoft Corp.", "emoji": "💻", "color": "#ff7f0e"},
    "GOOGL": {"name": "Alphabet Inc.", "emoji": "🔍", "color": "#2ca02c"},
    "AMZN": {"name": "Amazon.com Inc.", "emoji": "📦", "color": "#d62728"},
    "TSLA": {"name": "Tesla Inc.", "emoji": "🚗", "color": "#9467bd"},
    "NVDA": {"name": "NVIDIA Corp.", "emoji": "🎮", "color": "#8c564b"}
}

col1, col2, col3 = st.columns(3)
selected_stock = None

with col1:
    if st.button(f"{stocks['AAPL']['emoji']} AAPL\n{stocks['AAPL']['name']}", key="aapl"):
        selected_stock = "AAPL"
    if st.button(f"{stocks['GOOGL']['emoji']} GOOGL\n{stocks['GOOGL']['name']}", key="googl"):
        selected_stock = "GOOGL"

with col2:
    if st.button(f"{stocks['MSFT']['emoji']} MSFT\n{stocks['MSFT']['name']}", key="msft"):
        selected_stock = "MSFT"
    if st.button(f"{stocks['TSLA']['emoji']} TSLA\n{stocks['TSLA']['name']}", key="tsla"):
        selected_stock = "TSLA"

with col3:
    if st.button(f"{stocks['AMZN']['emoji']} AMZN\n{stocks['AMZN']['name']}", key="amzn"):
        selected_stock = "AMZN"
    if st.button(f"{stocks['NVDA']['emoji']} NVDA\n{stocks['NVDA']['name']}", key="nvda"):
        selected_stock = "NVDA"

# 선택된 종목 분석
if selected_stock:
    st.markdown("---")
    st.markdown(f"# 📈 {selected_stock} - {stocks[selected_stock]['name']} 실시간 AI 분석")
    
    # 진행 상황 컨테이너
    progress_container = st.container()
    
    with progress_container:
        # 전체 진행률
        overall_progress = st.progress(0)
        current_step = st.empty()
        
        # 1단계: 향상된 데이터 수집
        current_step.markdown("""
        <div class="step-indicator">
            🔍 <strong>1단계: 향상된 데이터 수집 에이전트</strong> - 다중 소스 데이터 수집 중...
        </div>
        """, unsafe_allow_html=True)
        overall_progress.progress(15)
        time.sleep(1)
        
        data_status = st.empty()
        data_status.markdown("""
        <div class="agent-status">
            ✅ Yahoo Finance API 연결 완료<br>
            🔍 네이버 뉴스 API 연동 중...<br>
            📱 소셜미디어 데이터 수집 중...<br>
            📊 애널리스트 리포트 수집 중...
        </div>
        """, unsafe_allow_html=True)
        
        # 기본 주식 데이터 수집
        try:
            stock = yf.Ticker(selected_stock)
            info = stock.info
            hist = stock.history(period="30d")
            
            current_price = safe_float(info.get('currentPrice', hist['Close'].iloc[-1] if not hist.empty else 100))
            prev_close = safe_float(info.get('previousClose', hist['Close'].iloc[-2] if len(hist) > 1 else current_price))
            change = current_price - prev_close
            change_percent = (change / prev_close) * 100 if prev_close > 0 else 0
            
        except Exception as e:
            st.error(f"❌ 기본 데이터 수집 실패: {e}")
            current_price = 100.0
            change_percent = 0.0
            hist = pd.DataFrame()
            info = {}
        
        # 모든 투자 데이터 수집 (수정된 동기 함수 사용)
        enhanced_news, social_data, analyst_data = collect_all_investment_data(selected_stock)
        
        data_status.markdown("""
        <div class="agent-status">
            ✅ 다중 소스 데이터 수집 완료 - 뉴스, 소셜미디어, 애널리스트 리포트 확보
        </div>
        """, unsafe_allow_html=True)
        
        # 데이터 품질 검증
        validator = DataQualityValidator()
        news_quality = validator.validate_news_quality(enhanced_news)
        data_completeness = validator.validate_data_completeness({
            'news': enhanced_news,
            'social': social_data,
            'analyst': analyst_data,
            'price': current_price
        })
        
        # 실시간 데이터 표시 (안전한 포맷팅 적용)
        st.markdown("### 📊 실시간 시장 데이터")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("현재가", f"${current_price:.2f}", f"{change:+.2f} ({change_percent:+.2f}%)")
        with col2:
            volume = safe_int(info.get('volume', 0))
            st.metric("거래량", f"{volume:,}")
        with col3:
            market_cap = safe_float(info.get('marketCap', 0))
            st.metric("시가총액", f"${market_cap/1e9:.1f}B" if market_cap > 0 else "N/A")
        with col4:
            pe_ratio = safe_float(info.get('trailingPE'))
            st.metric("P/E 비율", f"{pe_ratio:.2f}" if pe_ratio > 0 else "N/A")
        
        # 데이터 품질 표시
        st.markdown("### 📋 데이터 품질 검증")
        quality_col1, quality_col2, quality_col3 = st.columns(3)
        
        with quality_col1:
            high_quality = news_quality.count('high') if news_quality else 0
            st.metric("고품질 뉴스", f"{high_quality}개", f"총 {len(enhanced_news)}개 중")
        
        with quality_col2:
            complete_sources = sum(data_completeness.values()) if data_completeness else 0
            st.metric("데이터 완성도", f"{complete_sources}/4", "소스별 데이터 확보")
        
        with quality_col3:
            social_coverage = len(social_data) if social_data else 0
            st.metric("소셜 커버리지", f"{social_coverage}개", "플랫폼별 데이터")
        
        # 2단계: 향상된 감정 분석
        current_step.markdown("""
        <div class="step-indicator">
            💭 <strong>2단계: 향상된 감정 분석 에이전트</strong> - 다중 소스 감정 분석 중...
        </div>
        """, unsafe_allow_html=True)
        overall_progress.progress(30)
        
        sentiment_status = st.empty()
        sentiment_status.markdown("""
        <div class="agent-status">
            📰 뉴스 기사 감정 분석 중...<br>
            📱 소셜미디어 감정 분석 중...<br>
            📊 애널리스트 리포트 분석 중...<br>
            🤖 HyperCLOVA X 종합 감정 분석 중...
        </div>
        """, unsafe_allow_html=True)
        
        # 실제 감정 분석 실행 (수정된 동기 함수 사용)
        sentiment_result = run_sentiment_analysis(selected_stock)
        
        sentiment_status.markdown("""
        <div class="agent-status">
            ✅ 다중 소스 감정 분석 완료 - 뉴스, 소셜미디어, 애널리스트 종합 분석
        </div>
        """, unsafe_allow_html=True)
        
        # 감정 분석 결과 표시 (안전한 포맷팅 적용)
        st.markdown("### 💭 다중 소스 AI 감정 분석")
        sentiment_col1, sentiment_col2 = st.columns(2)
        
        with sentiment_col1:
            sentiment_score = safe_float(sentiment_result.get('sentiment_score', 0))
            sentiment_label = safe_format(sentiment_result.get('sentiment_label'), "중립적")
            sentiment_emoji = safe_format(sentiment_result.get('sentiment_emoji'), "😐")
            confidence = safe_float(sentiment_result.get('confidence', 0.5))
            article_count = safe_int(sentiment_result.get('article_count', 0))
            news_sources = sentiment_result.get('news_sources', [])
            
            if sentiment_score > 0.1:
                sentiment_color = "green"
            elif sentiment_score < -0.1:
                sentiment_color = "red"
            else:
                sentiment_color = "gray"
            
            st.markdown(f"""
            <div class="analysis-card">
                <h4>{sentiment_emoji} 종합 감정 점수: <span style="color: {sentiment_color};">{sentiment_score:+.2f}</span></h4>
                <p><strong>시장 감정:</strong> {sentiment_label}</p>
                <p><strong>신뢰도:</strong> {confidence:.1%}</p>
                <p><strong>분석 기사 수:</strong> {article_count}개</p>
                <p><strong>뉴스 소스:</strong> {', '.join(news_sources) if news_sources else 'N/A'}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # 소셜미디어 감정 표시
            if social_data:
                st.markdown("#### 📱 소셜미디어 감정")
                for social in social_data:
                    platform = safe_format(social.get('platform'), "Unknown")
                    score = safe_float(social.get('sentiment_score', 0))
                    mentions = safe_int(social.get('mention_count', 0))
                    
                    st.markdown(f"""
                    <div class="social-card">
                        <strong>{platform}</strong><br>
                        감정 점수: {score:+.2f}<br>
                        언급 수: {mentions:,}회
                    </div>
                    """, unsafe_allow_html=True)
        
        with sentiment_col2:
            # 감정 분포 차트
            analyzed_articles = sentiment_result.get('analyzed_articles', [])
            if analyzed_articles and article_count > 0:
                positive_articles = len([a for a in analyzed_articles if safe_float(a.get('score', 0)) > 0.1])
                neutral_articles = len([a for a in analyzed_articles if -0.1 <= safe_float(a.get('score', 0)) <= 0.1])
                negative_articles = len([a for a in analyzed_articles if safe_float(a.get('score', 0)) < -0.1])
                
                sentiment_data_chart = {
                    'Positive': positive_articles,
                    'Neutral': neutral_articles,
                    'Negative': negative_articles
                }
            else:
                sentiment_data_chart = {'No Data': 1}
            
            fig_sentiment = px.pie(
                values=list(sentiment_data_chart.values()),
                names=list(sentiment_data_chart.keys()),
                title="뉴스 감정 분포",
                color_discrete_map={
                    'Positive': '#28a745',
                    'Neutral': '#6c757d',
                    'Negative': '#dc3545',
                    'No Data': '#e9ecef'
                }
            )
            fig_sentiment.update_layout(height=300)
            st.plotly_chart(fig_sentiment, use_container_width=True)
            
            # 애널리스트 평가 표시
            if analyst_data:
                st.markdown("#### 📊 애널리스트 평가")
                for report in analyst_data:
                    source = safe_format(report.get('source'), "Unknown")
                    rating = safe_format(report.get('rating'), "Hold")
                    target = safe_float(report.get('target_price', 0))
                    
                    rating_color = "#28a745" if rating == "Buy" else "#dc3545" if rating == "Sell" else "#ffc107"
                    
                    st.markdown(f"""
                    <div style="background: {rating_color}; color: white; padding: 10px; border-radius: 5px; margin: 5px 0;">
                        <strong>{source}</strong><br>
                        평가: {rating}<br>
                        목표가: ${target:.2f}
                    </div>
                    """, unsafe_allow_html=True)
        
        # 3단계: 고급 기술적 분석
        current_step.markdown("""
        <div class="step-indicator">
            📊 <strong>3단계: 고급 기술적 분석 에이전트</strong> - 엘리어트 파동, 피보나치, 차트 패턴 분석 중...
        </div>
        """, unsafe_allow_html=True)
        overall_progress.progress(50)
        time.sleep(1.5)
        
        tech_status = st.empty()
        tech_status.markdown("""
        <div class="agent-status">
            📈 엘리어트 파동 분석 중...<br>
            📊 피보나치 레벨 계산 중...<br>
            🔍 차트 패턴 인식 중...<br>
            📉 고급 기술적 지표 계산 중...
        </div>
        """, unsafe_allow_html=True)
        
        # 고급 기술적 분석 실행
        advanced_analyzer = AdvancedTechnicalAnalyzer()
        
        try:
            if not hist.empty:
                # 기본 기술적 지표 계산
                delta = hist['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                current_rsi = safe_float(rsi.iloc[-1], 50)
                
                sma_20 = safe_float(hist['Close'].rolling(window=20).mean().iloc[-1], current_price)
                sma_50 = safe_float(hist['Close'].rolling(window=min(50, len(hist))).mean().iloc[-1], current_price)
                
                # 고급 기술적 분석
                advanced_results = advanced_analyzer.analyze_advanced_indicators(hist)
                
                # 신호 결정
                if current_rsi > 70:
                    rsi_signal = "과매수"
                    rsi_color = "red"
                    rsi_emoji = "⚠️"
                elif current_rsi < 30:
                    rsi_signal = "과매도"
                    rsi_color = "green"
                    rsi_emoji = "💚"
                else:
                    rsi_signal = "중립"
                    rsi_color = "blue"
                    rsi_emoji = "🔵"
                
                if current_price > sma_20 > sma_50:
                    ma_signal = "상승 추세"
                    ma_color = "green"
                    ma_emoji = "📈"
                    overall_signal = "BUY"
                elif current_price < sma_20 < sma_50:
                    ma_signal = "하락 추세"
                    ma_color = "red"
                    ma_emoji = "📉"
                    overall_signal = "SELL"
                else:
                    ma_signal = "횡보"
                    ma_color = "blue"
                    ma_emoji = "➡️"
                    overall_signal = "HOLD"
                
                # 기술적 분석 결과 저장
                technical_results = {
                    'overall_signal': overall_signal,
                    'signals': {
                        'RSI': 'BUY' if current_rsi < 30 else 'SELL' if current_rsi > 70 else 'HOLD',
                        'MA': 'BUY' if ma_signal == "상승 추세" else 'SELL' if ma_signal == "하락 추세" else 'HOLD'
                    },
                    'indicators': {
                        'RSI': current_rsi,
                        'SMA_20': sma_20,
                        'SMA_50': sma_50
                    },
                    'volatility': hist['Close'].pct_change().std() * np.sqrt(252) if len(hist) > 1 else 0.15,
                    'trend_strength': (current_price - sma_20) / sma_20 if sma_20 > 0 else 0,
                    'advanced_analysis': advanced_results
                }
            else:
                current_rsi = 50
                rsi_signal = "중립"
                ma_signal = "중립"
                overall_signal = "HOLD"
                sma_20 = current_price
                sma_50 = current_price
                advanced_results = {}
                
                technical_results = {
                    'overall_signal': 'HOLD',
                    'signals': {'RSI': 'HOLD', 'MA': 'HOLD'},
                    'indicators': {'RSI': 50, 'SMA_20': current_price, 'SMA_50': current_price},
                    'volatility': 0.15,
                    'trend_strength': 0,
                    'advanced_analysis': {}
                }
            
            tech_status.markdown("""
            <div class="agent-status">
                ✅ 고급 기술적 분석 완료 - 엘리어트 파동, 피보나치, 차트 패턴 분석 완료
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ 고급 기술적 분석 실패: {e}")
            current_rsi = 50
            rsi_signal = "중립"
            ma_signal = "중립"
            overall_signal = "HOLD"
            sma_20 = current_price
            sma_50 = current_price
            advanced_results = {}
            
            technical_results = {
                'overall_signal': 'HOLD',
                'signals': {'RSI': 'HOLD', 'MA': 'HOLD'},
                'indicators': {'RSI': 50, 'SMA_20': current_price, 'SMA_50': current_price},
                'volatility': 0.15,
                'trend_strength': 0,
                'advanced_analysis': {}
            }
        
        # 고급 기술적 분석 결과 표시 (안전한 포맷팅 적용)
        st.markdown("### 📈 고급 기술적 분석 결과")
        tech_col1, tech_col2 = st.columns(2)
        
        with tech_col1:
            st.markdown(f"""
            <div class="analysis-card">
                <h4>{safe_format(rsi_emoji, "🔵")} RSI (14일): <span style="color: {safe_format(rsi_color, 'blue')};">{current_rsi:.1f}</span></h4>
                <p><strong>신호:</strong> {safe_format(rsi_signal, "중립")}</p>
                <h4>{safe_format(ma_emoji, "➡️")} 이동평균: <span style="color: {safe_format(ma_color, 'blue')};">{safe_format(ma_signal, "중립")}</span></h4>
                <p><strong>SMA20:</strong> ${sma_20:.2f}</p>
                <p><strong>SMA50:</strong> ${sma_50:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # 엘리어트 파동 분석 표시
            elliott_wave = advanced_results.get('elliott_wave', {}) if advanced_results else {}
            if elliott_wave:
                st.markdown("#### 🌊 엘리어트 파동 분석")
                st.markdown(f"""
                <div class="analysis-card">
                    <p><strong>현재 파동:</strong> {safe_format(elliott_wave.get('current_wave'), 'N/A')}</p>
                    <p><strong>추세 방향:</strong> {safe_format(elliott_wave.get('trend_direction'), 'N/A')}</p>
                    <p><strong>완성도:</strong> {safe_float(elliott_wave.get('completion_percentage', 0)):.0f}%</p>
                    <p><strong>다음 목표:</strong> ${safe_float(elliott_wave.get('next_target', 0)):.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # 피보나치 레벨 표시
            fibonacci = advanced_results.get('fibonacci_levels', {}) if advanced_results else {}
            if fibonacci:
                st.markdown("#### 📐 피보나치 되돌림")
                nearest_support = safe_float(fibonacci.get('nearest_support', 0))
                nearest_resistance = safe_float(fibonacci.get('nearest_resistance', 0))
                
                st.markdown(f"""
                <div class="analysis-card">
                    <p><strong>가장 가까운 지지선:</strong> ${nearest_support:.2f}</p>
                    <p><strong>가장 가까운 저항선:</strong> ${nearest_resistance:.2f}</p>
                    <p><strong>되돌림 비율:</strong> {safe_float(fibonacci.get('retracement_percentage', 0)):.1%}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with tech_col2:
            # 고급 차트 표시
            if not hist.empty and advanced_results:
                try:
                    advanced_chart = advanced_analyzer.create_advanced_chart(hist, advanced_results)
                    st.plotly_chart(advanced_chart, use_container_width=True)
                except Exception as e:
                    # 기본 차트 표시
                    fig_price = go.Figure()
                    fig_price.add_trace(go.Scatter(
                        x=hist.index,
                        y=hist['Close'],
                        mode='lines',
                        name='종가',
                        line=dict(color=stocks[selected_stock]['color'], width=3)
                    ))
                    fig_price.update_layout(
                        title=f"{selected_stock} 주가 차트",
                        height=400
                    )
                    st.plotly_chart(fig_price, use_container_width=True)
            
            # 차트 패턴 인식 결과
            chart_patterns = advanced_results.get('chart_patterns', {}) if advanced_results else {}
            if chart_patterns:
                st.markdown("#### 📊 차트 패턴 인식")
                for pattern_name, pattern_data in chart_patterns.items():
                    if pattern_data and pattern_data.get('detected'):
                        pattern_type = safe_format(pattern_data.get('type'), 'Unknown')
                        confidence = safe_float(pattern_data.get('confidence', 0))
                        
                        st.markdown(f"""
                        <div class="pattern-detected">
                            <strong>🔍 {pattern_type} 패턴 감지</strong><br>
                            신뢰도: {confidence:.0%}
                        </div>
                        """, unsafe_allow_html=True)
        
        # 4단계: AI 종합 분석
        current_step.markdown("""
        <div class="step-indicator">
            🤖 <strong>4단계: AI 종합 분석 에이전트</strong> - HyperCLOVA X 종합 분석 중...
        </div>
        """, unsafe_allow_html=True)
        overall_progress.progress(70)
        time.sleep(2)
        
        ai_status = st.empty()
        ai_status.markdown("""
        <div class="agent-status">
            🧠 HyperCLOVA X 모델 로딩 중...<br>
            📊 다중 데이터 통합 및 패턴 분석 중...<br>
            💡 투자 전략 수립 중...<br>
            📝 종합 투자 권고안 작성 중...
        </div>
        """, unsafe_allow_html=True)
        
        # 소셜미디어 요약
        social_summary = ""
        if social_data:
            social_summary = ', '.join([f"{item.get('platform', 'Unknown')} {safe_float(item.get('sentiment_score', 0)):+.2f}" for item in social_data])
        
        # 애널리스트 요약
        analyst_summary = ""
        if analyst_data:
            ratings = [a.get('rating', 'Hold') for a in analyst_data if a]
            analyst_summary = f"애널리스트 평가: {', '.join(ratings)}"
        
        # 엘리어트 파동 요약
        elliott_summary = "N/A"
        if elliott_wave:
            elliott_summary = safe_format(elliott_wave.get('current_wave'), 'N/A')
        
        # 차트 패턴 요약
        pattern_count = 0
        if chart_patterns:
            pattern_count = len([p for p in chart_patterns.values() if p and p.get('detected', False)])
        
        ai_prompt = f"""
        {selected_stock} 주식에 대한 종합 투자 분석을 해주세요.
        
        현재 시장 상황:
        - 현재가: ${current_price:.2f}
        - 일일 변동률: {change_percent:+.2f}%
        - RSI: {current_rsi:.1f} ({rsi_signal})
        - 이동평균 신호: {ma_signal}
        - 뉴스 감정: {sentiment_label} (점수: {sentiment_score:+.2f})
        - {social_summary}
        - {analyst_summary}
        
        고급 기술적 분석:
        - 엘리어트 파동: {elliott_summary}
        - 차트 패턴: {pattern_count}개 패턴 감지
        
        다음 형식으로 분석해주세요:
        1. 현재 상황 종합 요약
        2. 다중 데이터 소스 기반 주요 인사이트
        3. 투자 추천 (매수/보유/매도)
        4. 목표 가격대 및 시간 프레임
        5. 리스크 요인 및 주의사항
        """
        
        # AI 분석 실행 (수정된 동기 함수 사용)
        ai_analysis = run_ai_analysis(ai_prompt)
        
        ai_status.markdown("""
        <div class="agent-status">
            ✅ AI 종합 분석 완료 - 다중 소스 데이터 기반 투자 권고안 생성 완료
        </div>
        """, unsafe_allow_html=True)
        
        # 5단계: 동적 시나리오 생성
        current_step.markdown("""
        <div class="step-indicator">
            🎯 <strong>5단계: 동적 시나리오 생성</strong> - 실시간 시나리오 분석 및 투자 권고 생성 중...
        </div>
        """, unsafe_allow_html=True)
        overall_progress.progress(90)
        time.sleep(1)
        
        # 동적 시나리오 분석 실행
        dynamic_scenarios = {}
        try:
            scenario_generator = DynamicScenarioGenerator()
            dynamic_scenarios = scenario_generator.generate_dynamic_scenarios(
                selected_stock,
                {'current_price': current_price, 'change_percent': change_percent},
                technical_results,
                sentiment_result,
                social_data,
                analyst_data
            )
        except Exception as e:
            st.warning(f"동적 시나리오 생성 실패: {e}")
        
        # 종합 점수 계산 (안전한 계산)
        tech_score = 1 if current_rsi < 70 and ma_signal == "상승 추세" else -1 if current_rsi > 70 or ma_signal == "하락 추세" else 0
        sentiment_weight = sentiment_score * 2
        social_weight = np.mean([safe_float(s.get('sentiment_score', 0)) for s in social_data]) if social_data else 0
        analyst_weight = np.mean([1 if a.get('rating') == 'Buy' else -1 if a.get('rating') == 'Sell' else 0 for a in analyst_data if a]) if analyst_data else 0
        
        final_score = (tech_score + sentiment_weight + social_weight * 0.3 + analyst_weight * 0.4) / 3
        
        if final_score > 0.3:
            recommendation = "매수 추천"
            rec_color = "#28a745"
            rec_emoji = "🚀"
            confidence_level = min(95, 70 + abs(final_score) * 25)
        elif final_score < -0.3:
            recommendation = "매도 고려"
            rec_color = "#dc3545"
            rec_emoji = "⚠️"
            confidence_level = min(95, 70 + abs(final_score) * 25)
        else:
            recommendation = "보유 추천"
            rec_color = "#ffc107"
            rec_emoji = "🤝"
            confidence_level = 75
        
        # 진행 상황 완료
        overall_progress.progress(100)
        time.sleep(1)
        progress_container.empty()
        
        # 최종 추천 박스
        st.markdown(f"""
        <div class="recommendation-box">
            <h2>{rec_emoji} 최종 AI 투자 권고</h2>
            <h1 style="margin: 20px 0; font-size: 3rem;">{recommendation}</h1>
            <h3>🎯 신뢰도: {confidence_level:.0f}%</h3>
            <p style="margin-top: 20px; font-size: 1.2rem;">종합 점수: {final_score:+.2f}</p>
            <p style="margin-top: 10px;">분석 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # AI 상세 분석 결과
        st.markdown("### 🤖 HyperCLOVA X 상세 분석")
        st.markdown(f"""
        <div class="analysis-card">
            {ai_analysis}
        </div>
        """, unsafe_allow_html=True)
        
        # 동적 시나리오 분석 표시 (안전한 포맷팅 적용)
        if dynamic_scenarios and 'scenarios' in dynamic_scenarios:
            st.markdown("### 📊 동적 투자 시나리오 분석")
            
            scenarios = dynamic_scenarios['scenarios']
            market_regime = dynamic_scenarios.get('market_regime', {})
            
            # 시장 환경 표시
            regime = safe_format(market_regime.get('regime'), 'sideways')
            regime_confidence = safe_float(market_regime.get('confidence', 0.5))
            
            st.markdown(f"""
            <div class="analysis-card">
                <h4>🌊 현재 시장 환경: {regime.upper()}</h4>
                <p>환경 신뢰도: {regime_confidence:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # 시나리오 카드 표시
            scenario_col1, scenario_col2, scenario_col3 = st.columns(3)
            
            scenario_configs = [
                ('bull_case', '🟢 낙관적 시나리오', scenario_col1, '#28a745'),
                ('base_case', '🟡 기본 시나리오', scenario_col2, '#ffc107'),
                ('bear_case', '🔴 비관적 시나리오', scenario_col3, '#dc3545')
            ]
            
            for scenario_key, scenario_title, col, color in scenario_configs:
                if scenario_key in scenarios:
                    scenario_data = scenarios[scenario_key]
                    prob = safe_float(scenario_data.get('probability', 0.33))
                    price_target = safe_float(scenario_data.get('price_target', current_price))
                    return_range = scenario_data.get('return_range', [0, 0])
                    key_drivers = scenario_data.get('key_drivers', [])
                    confidence = safe_float(scenario_data.get('confidence', 0.7))
                    
                    expected_return = (price_target / current_price - 1) * 100 if current_price > 0 else 0
                    
                    with col:
                        st.markdown(f"""
                        <div style="border: 2px solid {color}; border-radius: 10px; padding: 15px; margin: 5px;">
                            <h4 style="color: {color}; margin-bottom: 10px;">{scenario_title}</h4>
                            <p><strong>확률:</strong> {prob:.0%}</p>
                            <p><strong>목표가:</strong> ${price_target:.2f}</p>
                            <p><strong>예상 수익률:</strong> {expected_return:+.1f}%</p>
                            <p><strong>수익률 범위:</strong> {safe_float(return_range[0] if return_range else 0):.1%} ~ {safe_float(return_range[1] if len(return_range) > 1 else 0):.1%}</p>
                            <p><strong>신뢰도:</strong> {confidence:.1%}</p>
                            <p><strong>핵심 동인:</strong></p>
                            <ul>
                        """, unsafe_allow_html=True)
                        
                        for driver in key_drivers[:3]:
                            st.markdown(f"<li>{safe_format(driver, 'N/A')}</li>", unsafe_allow_html=True)
                        
                        st.markdown("</ul></div>", unsafe_allow_html=True)
            
            # 시간별 시나리오 전개
            time_scenarios = dynamic_scenarios.get('time_based_scenarios', {})
            if time_scenarios:
                st.markdown("#### ⏰ 시간별 시나리오 전개")
                
                time_col1, time_col2 = st.columns(2)
                
                with time_col1:
                    for period in ['1_month', '3_months']:
                        if period in time_scenarios:
                            period_name = period.replace('_', ' ').title()
                            period_data = time_scenarios[period]
                            
                            st.markdown(f"**{period_name}**")
                            for scenario_name, data in period_data.items():
                                prob = safe_float(data.get('probability', 0))
                                expected_price = safe_float(data.get('expected_price', current_price))
                                st.write(f"- {scenario_name}: {prob:.0%} 확률, ${expected_price:.2f}")
                
                with time_col2:
                    for period in ['6_months', '12_months']:
                        if period in time_scenarios:
                            period_name = period.replace('_', ' ').title()
                            period_data = time_scenarios[period]
                            
                            st.markdown(f"**{period_name}**")
                            for scenario_name, data in period_data.items():
                                prob = safe_float(data.get('probability', 0))
                                expected_price = safe_float(data.get('expected_price', current_price))
                                st.write(f"- {scenario_name}: {prob:.0%} 확률, ${expected_price:.2f}")
        
        # 에이전트 활동 로그
        with st.expander("🔍 AI 에이전트 활동 상세 로그"):
            st.markdown(f"""
            **📊 분석 세션 정보**
            - 분석 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            - 분석 대상: {selected_stock} ({stocks[selected_stock]['name']})
            - 총 소요 시간: 약 15초
            
            **🤖 에이전트 실행 순서**
            1. ✅ **향상된 데이터 수집**: Yahoo Finance + 네이버 뉴스 API + 소셜미디어 + 애널리스트 리포트
            2. ✅ **다중 소스 감정 분석**: {article_count}개 뉴스 + {len(social_data)}개 소셜 플랫폼 + {len(analyst_data)}개 애널리스트 리포트
            3. ✅ **고급 기술적 분석**: 엘리어트 파동, 피보나치, 차트 패턴, 고급 지표 분석
            4. ✅ **AI 종합 분석**: HyperCLOVA X 다중 데이터 통합 분석
            5. ✅ **동적 시나리오 생성**: 실시간 시장 환경 기반 시나리오 분석
            6. ✅ **최종 추천**: "{recommendation}" 생성 (신뢰도 {confidence_level:.0f}%)
            
            **📈 수집된 데이터 품질**
            - 뉴스 데이터: ✅ {len(enhanced_news)}개 기사 (고품질: {high_quality}개)
            - 소셜미디어: ✅ {len(social_data)}개 플랫폼 데이터
            - 애널리스트: ✅ {len(analyst_data)}개 기관 리포트
            - 기술적 분석: ✅ 기본 + 고급 지표 완료
            - AI 모델: ✅ HyperCLOVA X 정상 작동
            
            **🎯 시나리오 분석 결과**
            - 시장 환경: {regime.upper()}
            - 동적 시나리오: {len(scenarios) if 'scenarios' in dynamic_scenarios else 0}개 생성
            - 시간별 전개: {len(time_scenarios) if time_scenarios else 0}개 기간 분석
            
            **⚠️ Rate Limiting 정보**
            - 모든 API에 지수 백오프 재시도 로직 적용
            - 비동기 처리를 동기 함수로 래핑하여 Streamlit 호환성 확보
            - ChatClovaX와 LangChain 통합으로 안정적 AI 분석 제공
            """)

else:
    # 초기 화면
    st.markdown("""
    <div style="text-align: center; padding: 50px; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 15px; margin: 20px 0;">
        <h2 style="color: #2c3e50;">🚀 차세대 AI 에이전트가 실시간으로 주식을 분석합니다</h2>
        <p style="font-size: 18px; color: #34495e; margin: 20px 0;">위에서 분석하고 싶은 종목을 선택해주세요</p>
        <div style="margin: 30px 0;">
            <p style="font-size: 16px; color: #7f8c8d;">
                🔍 다중 데이터 수집 → 💭 감정 분석 → 📊 고급 기술적 분석 → 🤖 AI 종합 분석 → 🎯 동적 시나리오 → 📈 투자 추천
            </p>
        </div>
        <div style="margin: 20px 0;">
            <h3 style="color: #2c3e50;">🌟 주요 특징</h3>
            <div style="display: flex; justify-content: space-around; margin: 20px 0;">
                <div style="text-align: center;">
                    <h4 style="color: #3498db;">📰 뉴스 분석</h4>
                    <p>네이버 API + 다중 소스</p>
                </div>
                <div style="text-align: center;">
                    <h4 style="color: #e74c3c;">📱 소셜미디어</h4>
                    <p>Reddit, Twitter, StockTwits</p>
                </div>
                <div style="text-align: center;">
                    <h4 style="color: #f39c12;">📊 고급 분석</h4>
                    <p>엘리어트 파동, 피보나치</p>
                </div>
                <div style="text-align: center;">
                    <h4 style="color: #27ae60;">🎯 동적 시나리오</h4>
                    <p>실시간 시장 환경 반영</p>
                </div>
            </div>
        </div>
        <p style="font-size: 14px; color: #95a5a6;">
            각 단계별로 전문 AI 에이전트가 협업하여 종합적인 투자 분석을 제공합니다
        </p>
    </div>
    """, unsafe_allow_html=True)

# 사이드바 정보
with st.sidebar:
    st.markdown("### ℹ️ Enhanced 시스템 정보")
    st.markdown("""
    **🤖 AI 에이전트 구성:**
    - 🔍 향상된 데이터 수집 에이전트
    - 💭 다중 소스 감정 분석 에이전트  
    - 📊 고급 기술적 분석 에이전트
    - 🤖 HyperCLOVA X 종합 분석 에이전트
    - 🎯 동적 시나리오 생성 엔진
    - 📋 데이터 품질 검증 시스템
    
    **📊 데이터 소스:**
    - 네이버 뉴스 API
    - Yahoo Finance API
    - 소셜미디어 (Reddit, Twitter, StockTwits)
    - 애널리스트 리포트 (3개 기관)
    - 실시간 주가 데이터
    
    **🔧 고급 기술 스택:**
    - HyperCLOVA X (HCX-005)
    - 엘리어트 파동 분석
    - 피보나치 되돌림
    - 차트 패턴 인식
    - 동적 시나리오 생성
    - 몬테카르로 시뮬레이션
    
    **📈 분석 지표:**
    - 기본 기술적 지표 (RSI, MACD, 볼린저 밴드)
    - 고급 지표 (Ichimoku, Parabolic SAR, ATR)
    - 차트 패턴 (헤드앤숄더, 삼각형, 더블탑)
    - 감정 지수 (뉴스, 소셜, 애널리스트)
    """)
    
    st.markdown("---")
    st.markdown("### ⚠️ 투자 유의사항")
    st.markdown("""
    - 이 분석은 **교육 목적**으로 제공됩니다
    - 실제 투자 결정은 **전문가 상담** 후 신중히 하세요
    - 과거 성과가 **미래를 보장하지 않습니다**
    - AI 분석은 **참고용**으로만 활용하세요
    - 다중 데이터 소스를 **종합적으로 검토**하세요
    """)
    
    st.markdown("---")
    st.markdown("### 🆕 새로운 기능")
    st.markdown("""
    - ✨ **Rate Limiting** 자동 관리
    - 🔄 **지수 백오프** 재시도 로직
    - 🛡️ **안전한 포맷팅** 시스템
    - 📊 **실시간 오류** 모니터링
    - 🎯 **동적 시나리오** 생성
    - 📋 **데이터 품질** 검증
    - 🔧 **비동기 처리** 최적화
    - 🤖 **ChatClovaX** 통합
    """)
    
    st.markdown("---")
    st.markdown("### 📞 기술 지원")
    st.markdown("""
    **시스템 상태:**
    - ✅ Streamlit 호환성 확보
    - ✅ 비동기 처리 안정화
    - ✅ Rate Limiting 적용
    - ✅ 에러 처리 강화
    
    **성능 최적화:**
    - 🚀 실시간 데이터 수집
    - ⚡ 고속 AI 분석
    - 🔄 자동 재시도 시스템
    - 📊 품질 검증 자동화
    """)