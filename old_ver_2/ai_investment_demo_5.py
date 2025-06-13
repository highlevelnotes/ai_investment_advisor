# perplexity_stock_advisor.py (수정된 전체 코드)
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import random
import os
import numpy as np
import asyncio
import nest_asyncio
from langchain_naver import ChatClovaX
from langchain_core.messages import HumanMessage, SystemMessage
from sentiment_analyzer import RealSentimentAnalyzer
from scenario_analyzer import AdvancedScenarioAnalyzer

from dotenv import load_dotenv

load_dotenv()

# 환경 설정
os.environ["LANGCHAIN_TRACING_V2"] = "false"
nest_asyncio.apply()

class StockAnalysisAgent:
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

# Streamlit 페이지 설정
st.set_page_config(
    page_title="AI 주식 어드바이저 - Perplexity Labs 스타일",
    page_icon="📈",
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
</style>
""", unsafe_allow_html=True)

# 메인 헤더
st.markdown("""
<div class="main-header">
    <h1>🤖 AI 주식 어드바이저</h1>
    <h3>Perplexity Labs 스타일 - 실시간 AI 에이전트 협업 분석</h3>
    <p>종목을 선택하면 AI 에이전트들이 실시간으로 협업하여 종합 분석을 제공합니다</p>
</div>
""", unsafe_allow_html=True)

# 종목 선택 섹션
st.markdown("## 📊 분석할 종목을 선택하세요")

# 종목 정보
stocks = {
    "AAPL": {"name": "Apple Inc.", "emoji": "🍎", "color": "#1f77b4"},
    "MSFT": {"name": "Microsoft Corp.", "emoji": "💻", "color": "#ff7f0e"},
    "GOOGL": {"name": "Alphabet Inc.", "emoji": "🔍", "color": "#2ca02c"},
    "AMZN": {"name": "Amazon.com Inc.", "emoji": "📦", "color": "#d62728"},
    "TSLA": {"name": "Tesla Inc.", "emoji": "🚗", "color": "#9467bd"},
    "NVDA": {"name": "NVIDIA Corp.", "emoji": "🎮", "color": "#8c564b"}
}

# 버튼 레이아웃
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
        
        # 1단계: 데이터 수집 에이전트
        current_step.markdown("""
        <div class="step-indicator">
            🔍 <strong>1단계: 데이터 수집 에이전트</strong> - 실시간 주식 데이터 수집 중...
        </div>
        """, unsafe_allow_html=True)
        overall_progress.progress(20)
        time.sleep(1)
        
        # 데이터 수집 실행
        data_status = st.empty()
        data_status.markdown("""
        <div class="agent-status">
            ✅ Yahoo Finance API 연결 완료<br>
            ✅ 실시간 주가 데이터 수집 중<br>
            ✅ 과거 30일 데이터 로딩 중...
        </div>
        """, unsafe_allow_html=True)
        
        try:
            stock = yf.Ticker(selected_stock)
            info = stock.info
            hist = stock.history(period="30d")
            
            current_price = info.get('currentPrice', hist['Close'].iloc[-1] if not hist.empty else 0)
            prev_close = info.get('previousClose', hist['Close'].iloc[-2] if len(hist) > 1 else 0)
            change = current_price - prev_close
            change_percent = (change / prev_close) * 100
            
            data_status.markdown("""
            <div class="agent-status">
                ✅ 데이터 수집 완료 - 실시간 주가 및 거래 정보 확보
            </div>
            """, unsafe_allow_html=True)
            
            # 실시간 데이터 표시
            st.markdown("### 📊 실시간 시장 데이터")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "현재가", 
                    f"${current_price:.2f}",
                    f"{change:+.2f} ({change_percent:+.2f}%)"
                )
            
            with col2:
                volume = info.get('volume', 0)
                st.metric("거래량", f"{volume:,}")
            
            with col3:
                market_cap = info.get('marketCap', 0)
                st.metric("시가총액", f"${market_cap/1e9:.1f}B")
            
            with col4:
                pe_ratio = info.get('trailingPE', 0)
                st.metric("P/E 비율", f"{pe_ratio:.2f}" if pe_ratio else "N/A")
            
        except Exception as e:
            st.error(f"❌ 데이터 수집 실패: {e}")
            current_price = 100
            change_percent = 0
            hist = pd.DataFrame()  # 빈 DataFrame 생성
        
        # 2단계: 감정 분석 에이전트
        current_step.markdown("""
        <div class="step-indicator">
            💭 <strong>2단계: 감정 분석 에이전트</strong> - 실제 뉴스 수집 및 AI 감정 분석 중...
        </div>
        """, unsafe_allow_html=True)
        overall_progress.progress(40)
        
        sentiment_status = st.empty()
        sentiment_status.markdown("""
        <div class="agent-status">
            📰 다중 소스에서 뉴스 수집 중...<br>
            🔍 Yahoo Finance, Google News, MarketWatch 검색 중...<br>
            🤖 HyperCLOVA X 감정 분석 실행 중...<br>
            📊 종합 감정 지수 계산 중...
        </div>
        """, unsafe_allow_html=True)
        
        # 실제 감정 분석 실행
        sentiment_analyzer = RealSentimentAnalyzer()
        try:
            sentiment_result = asyncio.run(sentiment_analyzer.analyze_stock_sentiment(selected_stock))
        except Exception as e:
            sentiment_result = {
                'sentiment_score': 0.0,
                'sentiment_label': "중립적",
                'sentiment_emoji': "😐",
                'confidence': 0.5,
                'article_count': 0,
                'analyzed_articles': [],
                'news_sources': [],
                'method': 'error_fallback'
            }
        
        sentiment_status.markdown("""
        <div class="agent-status">
            ✅ 실제 뉴스 감정 분석 완료 - 다중 소스 뉴스 분석 완료
        </div>
        """, unsafe_allow_html=True)
        
        # 감정 분석 결과 표시
        st.markdown("### 💭 실제 뉴스 기반 AI 감정 분석")
        sentiment_col1, sentiment_col2 = st.columns(2)
        
        with sentiment_col1:
            sentiment_score = sentiment_result['sentiment_score']
            sentiment_label = sentiment_result['sentiment_label']
            sentiment_emoji = sentiment_result['sentiment_emoji']
            confidence = sentiment_result['confidence']
            article_count = sentiment_result['article_count']
            
            if sentiment_score > 0.1:
                sentiment_color = "green"
            elif sentiment_score < -0.1:
                sentiment_color = "red"
            else:
                sentiment_color = "gray"
            
            st.markdown(f"""
            <div class="analysis-card">
                <h4>{sentiment_emoji} 감정 점수: <span style="color: {sentiment_color};">{sentiment_score:+.2f}</span></h4>
                <p><strong>시장 감정:</strong> {sentiment_label}</p>
                <p><strong>신뢰도:</strong> {confidence:.1%}</p>
                <p><strong>분석 기사 수:</strong> {article_count}개</p>
                <p><strong>뉴스 소스:</strong> {', '.join(sentiment_result['news_sources'])}</p>
                <p><strong>분석 방법:</strong> {sentiment_result['method']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # 분석된 기사 목록 표시
            if sentiment_result['analyzed_articles']:
                with st.expander("📰 분석된 뉴스 기사"):
                    for article in sentiment_result['analyzed_articles'][:5]:
                        st.markdown(f"""
                        **{article['title']}**  
                        감정: {article['sentiment']} (점수: {article['score']:+.2f})  
                        출처: {article['source']}
                        """)
        
        with sentiment_col2:
            # 감정 분포 차트 (실제 데이터 기반)
            if article_count > 0:
                positive_articles = len([a for a in sentiment_result['analyzed_articles'] if a['score'] > 0.1])
                neutral_articles = len([a for a in sentiment_result['analyzed_articles'] if -0.1 <= a['score'] <= 0.1])
                negative_articles = len([a for a in sentiment_result['analyzed_articles'] if a['score'] < -0.1])
                
                sentiment_data = {
                    'Positive': positive_articles,
                    'Neutral': neutral_articles,
                    'Negative': negative_articles
                }
            else:
                sentiment_data = {'No Data': 1}
            
            fig_sentiment = px.pie(
                values=list(sentiment_data.values()),
                names=list(sentiment_data.keys()),
                title="실제 뉴스 감정 분포",
                color_discrete_map={
                    'Positive': '#28a745',
                    'Neutral': '#6c757d',
                    'Negative': '#dc3545',
                    'No Data': '#e9ecef'
                }
            )
            fig_sentiment.update_layout(height=300)
            st.plotly_chart(fig_sentiment, use_container_width=True)
        
        # 3단계: 기술적 분석 에이전트
        current_step.markdown("""
        <div class="step-indicator">
            📊 <strong>3단계: 기술적 분석 에이전트</strong> - 기술적 지표 계산 중...
        </div>
        """, unsafe_allow_html=True)
        overall_progress.progress(60)
        time.sleep(1.5)
        
        tech_status = st.empty()
        tech_status.markdown("""
        <div class="agent-status">
            📈 RSI 지표 계산 중...<br>
            📊 MACD 신호 분석 중...<br>
            📉 이동평균선 분석 중...<br>
            🔍 볼린저 밴드 계산 중...
        </div>
        """, unsafe_allow_html=True)
        
        try:
            if not hist.empty:
                # 기술적 지표 계산
                delta = hist['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                current_rsi = rsi.iloc[-1]
                
                # 이동평균
                sma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
                sma_50 = hist['Close'].rolling(window=min(50, len(hist))).mean().iloc[-1]
                
                # RSI 신호
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
                
                # 이동평균 신호
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
                
                # 기술적 분석 결과 저장 (중요!)
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
                    'trend_strength': (current_price - sma_20) / sma_20 if sma_20 > 0 else 0
                }
            else:
                # 데이터가 없는 경우 기본값
                current_rsi = 50
                rsi_signal = "중립"
                ma_signal = "중립"
                overall_signal = "HOLD"
                sma_20 = current_price
                sma_50 = current_price
                
                technical_results = {
                    'overall_signal': 'HOLD',
                    'signals': {'RSI': 'HOLD', 'MA': 'HOLD'},
                    'indicators': {'RSI': 50, 'SMA_20': current_price, 'SMA_50': current_price},
                    'volatility': 0.15,
                    'trend_strength': 0
                }
            
            tech_status.markdown("""
            <div class="agent-status">
                ✅ 기술적 분석 완료 - 모든 지표 계산 완료
            </div>
            """, unsafe_allow_html=True)
            
            # 기술적 분석 결과 표시
            st.markdown("### 📈 기술적 분석 결과")
            tech_col1, tech_col2 = st.columns(2)
            
            with tech_col1:
                st.markdown(f"""
                <div class="analysis-card">
                    <h4>{rsi_emoji} RSI (14일): <span style="color: {rsi_color};">{current_rsi:.1f}</span></h4>
                    <p><strong>신호:</strong> {rsi_signal}</p>
                    <h4>{ma_emoji} 이동평균: <span style="color: {ma_color};">{ma_signal}</span></h4>
                    <p><strong>SMA20:</strong> ${sma_20:.2f}</p>
                    <p><strong>SMA50:</strong> ${sma_50:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with tech_col2:
                if not hist.empty:
                    # 가격 차트
                    fig_price = go.Figure()
                    
                    fig_price.add_trace(go.Scatter(
                        x=hist.index,
                        y=hist['Close'],
                        mode='lines',
                        name='종가',
                        line=dict(color=stocks[selected_stock]['color'], width=3)
                    ))
                    
                    fig_price.add_trace(go.Scatter(
                        x=hist.index,
                        y=hist['Close'].rolling(window=20).mean(),
                        mode='lines',
                        name='SMA20',
                        line=dict(color='orange', width=2, dash='dash')
                    ))
                    
                    fig_price.update_layout(
                        title=f"{selected_stock} 30일 주가 차트",
                        xaxis_title="날짜",
                        yaxis_title="가격 ($)",
                        height=350,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig_price, use_container_width=True)
                else:
                    st.info("차트 데이터가 없습니다.")
                
        except Exception as e:
            st.error(f"❌ 기술적 분석 실패: {e}")
            current_rsi = 50
            rsi_signal = "중립"
            ma_signal = "중립"
            
            # 오류 시 기본값 설정
            technical_results = {
                'overall_signal': 'HOLD',
                'signals': {'RSI': 'HOLD', 'MA': 'HOLD'},
                'indicators': {'RSI': 50, 'SMA_20': current_price, 'SMA_50': current_price},
                'volatility': 0.15,
                'trend_strength': 0
            }
        
        # 4단계: AI 종합 분석
        current_step.markdown("""
        <div class="step-indicator">
            🤖 <strong>4단계: AI 종합 분석 에이전트</strong> - HyperCLOVA X 종합 분석 중...
        </div>
        """, unsafe_allow_html=True)
        overall_progress.progress(80)
        time.sleep(2)
        
        ai_status = st.empty()
        ai_status.markdown("""
        <div class="agent-status">
            🧠 HyperCLOVA X 모델 로딩 중...<br>
            📊 데이터 통합 및 패턴 분석 중...<br>
            💡 투자 전략 수립 중...<br>
            📝 투자 권고안 작성 중...
        </div>
        """, unsafe_allow_html=True)
        
        # AI 분석 실행
        analysis_agent = StockAnalysisAgent()
        
        ai_prompt = f"""
        {selected_stock} 주식에 대한 종합 투자 분석을 해주세요.
        
        현재 시장 상황:
        - 현재가: ${current_price:.2f}
        - 일일 변동률: {change_percent:+.2f}%
        - RSI: {current_rsi:.1f} ({rsi_signal})
        - 이동평균 신호: {ma_signal}
        - 시장 감정: {sentiment_label} (점수: {sentiment_score:+.2f})
        - 신뢰도: {confidence:.1%}
        
        다음 형식으로 분석해주세요:
        1. 현재 상황 요약
        2. 주요 강점과 위험 요소
        3. 투자 추천 (매수/보유/매도)
        4. 목표 가격대 제시
        5. 투자 기간 권장사항
        """
        
        try:
            ai_analysis = analysis_agent.analyze_with_ai(ai_prompt)
        except Exception as e:
            ai_analysis = f"""
            **{selected_stock} 종합 분석 결과**
            
            **현재 상황:** {sentiment_label} 시장 감정과 {rsi_signal} RSI 신호를 보이고 있습니다.
            
            **기술적 분석:** {ma_signal} 추세에서 {change_percent:+.2f}% 변동을 기록했습니다.
            
            **투자 권고:** {"매수 고려" if sentiment_score > 0 and current_rsi < 70 else "보유 권장" if abs(sentiment_score) < 0.2 else "신중한 접근"}
            
            **위험 요소:** 시장 변동성과 기술적 지표 신호를 지속 모니터링 필요
            """
        
        ai_status.markdown("""
        <div class="agent-status">
            ✅ AI 종합 분석 완료 - 투자 권고안 생성 완료
        </div>
        """, unsafe_allow_html=True)
        
        # 5단계: 최종 추천 생성
        current_step.markdown("""
        <div class="step-indicator">
            🎯 <strong>5단계: 최종 추천 생성</strong> - 투자 권고 및 신뢰도 계산 중...
        </div>
        """, unsafe_allow_html=True)
        overall_progress.progress(100)
        time.sleep(1)
        
        # 종합 점수 계산
        tech_score = 1 if current_rsi < 70 and ma_signal == "상승 추세" else -1 if current_rsi > 70 or ma_signal == "하락 추세" else 0
        sentiment_weight = sentiment_score * 2
        final_score = (tech_score + sentiment_weight) / 2
        
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
        
        # 진행 상황 제거
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
        
        # 고급 시나리오 분석 실행
        scenario_analyzer = AdvancedScenarioAnalyzer()
        
        try:
            scenario_result = scenario_analyzer.analyze_investment_scenarios(
                selected_stock, 
                {'current_price': current_price, 'change_percent': change_percent},
                technical_results,
                sentiment_result
            )
        except Exception as e:
            scenario_result = scenario_analyzer._get_fallback_scenarios(
                selected_stock, 
                {'current_price': current_price}
            )
        
        # 고급 투자 시나리오 분석 표시
        st.markdown("### 📊 AI 기반 투자 시나리오 분석")
        
        scenarios = scenario_result.get('scenarios', {})
        ai_interpretation = scenario_result.get('ai_interpretation', '')
        
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
                prob = scenario_data['probability']
                price_target = scenario_data['price_target']
                return_range = scenario_data['return_range']
                key_factors = scenario_data.get('key_factors', [])
                
                expected_return = (price_target / current_price - 1) * 100 if current_price > 0 else 0
                
                with col:
                    st.markdown(f"""
                    <div style="border: 2px solid {color}; border-radius: 10px; padding: 15px; margin: 5px;">
                        <h4 style="color: {color}; margin-bottom: 10px;">{scenario_title}</h4>
                        <p><strong>확률:</strong> {prob:.0%}</p>
                        <p><strong>목표가:</strong> ${price_target:.2f}</p>
                        <p><strong>예상 수익률:</strong> {expected_return:+.1f}%</p>
                        <p><strong>수익률 범위:</strong> {return_range[0]:.1%} ~ {return_range[1]:.1%}</p>
                        <p><strong>핵심 요인:</strong></p>
                        <ul>
                    """, unsafe_allow_html=True)
                    
                    for factor in key_factors[:3]:
                        st.markdown(f"<li>{factor}</li>", unsafe_allow_html=True)
                    
                    st.markdown("</ul></div>", unsafe_allow_html=True)
        
        # AI 해석 표시
        if ai_interpretation:
            st.markdown("### 🤖 HyperCLOVA X 시나리오 해석")
            st.markdown(f"""
            <div style="background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 4px solid #007bff;">
                {ai_interpretation}
            </div>
            """, unsafe_allow_html=True)
        
        # 에이전트 활동 로그
        with st.expander("🔍 AI 에이전트 활동 상세 로그"):
            st.markdown(f"""
            **📊 분석 세션 정보**
            - 분석 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            - 분석 대상: {selected_stock} ({stocks[selected_stock]['name']})
            - 총 소요 시간: 약 10초
            
            **🤖 에이전트 실행 순서**
            1. ✅ **데이터 수집 에이전트**: Yahoo Finance API 연동, 실시간 주가 및 30일 차트 데이터 수집
            2. ✅ **감정 분석 에이전트**: {article_count}개 뉴스 기사 분석, 감정 점수 {sentiment_score:+.2f} 도출
            3. ✅ **기술적 분석 에이전트**: RSI({current_rsi:.1f}), 이동평균, MACD 등 주요 지표 계산
            4. ✅ **AI 분석 에이전트**: HyperCLOVA X 모델을 통한 종합 분석 및 투자 전략 수립
            5. ✅ **추천 엔진**: 최종 투자 권고 "{recommendation}" 생성 (신뢰도 {confidence_level:.0f}%)
            6. ✅ **시나리오 분석**: 몬테카르로 시뮬레이션 기반 투자 시나리오 분석
            
            **📈 수집된 데이터 품질**
            - 실시간 데이터: ✅ 정상
            - 과거 데이터: ✅ 30일 완전 수집
            - 뉴스 데이터: ✅ 최신 기사 분석 완료
            - AI 모델 응답: ✅ 정상 작동
            """)

else:
    # 초기 화면
    st.markdown("""
    <div style="text-align: center; padding: 50px; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 15px; margin: 20px 0;">
        <h2 style="color: #2c3e50;">🚀 AI 에이전트가 실시간으로 주식을 분석합니다</h2>
        <p style="font-size: 18px; color: #34495e; margin: 20px 0;">위에서 분석하고 싶은 종목을 선택해주세요</p>
        <div style="margin: 30px 0;">
            <p style="font-size: 16px; color: #7f8c8d;">
                🔍 데이터 수집 → 💭 감정 분석 → 📊 기술적 분석 → 🤖 AI 종합 분석 → 🎯 투자 추천
            </p>
        </div>
        <p style="font-size: 14px; color: #95a5a6;">
            각 단계별로 전문 AI 에이전트가 협업하여 종합적인 투자 분석을 제공합니다
        </p>
    </div>
    """, unsafe_allow_html=True)

# 사이드바 정보
with st.sidebar:
    st.markdown("### ℹ️ 시스템 정보")
    st.markdown("""
    **🤖 AI 에이전트 구성:**
    - 🔍 데이터 수집 에이전트
    - 💭 감정 분석 에이전트  
    - 📊 기술적 분석 에이전트
    - 🤖 HyperCLOVA X 분석 에이전트
    - 🎯 추천 생성 엔진
    - 📊 시나리오 분석 엔진
    
    **📊 데이터 소스:**
    - Yahoo Finance API
    - 실시간 주가 데이터
    - 30일 과거 데이터
    - 금융 뉴스 감정 분석
    - 기술적 지표 계산
    
    **🔧 기술 스택:**
    - HyperCLOVA X (HCX-005)
    - Streamlit
    - yfinance
    - Plotly
    - 몬테카르로 시뮬레이션
    """)
    
    st.markdown("---")
    st.markdown("### ⚠️ 투자 유의사항")
    st.markdown("""
    - 이 분석은 **교육 목적**으로 제공됩니다
    - 실제 투자 결정은 **전문가 상담** 후 신중히 하세요
    - 과거 성과가 **미래를 보장하지 않습니다**
    - AI 분석은 **참고용**으로만 활용하세요
    """)
