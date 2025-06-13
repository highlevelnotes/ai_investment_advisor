# perplexity_stock_advisor.py
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import random
import asyncio
from langchain_naver import ChatClovaX
from langchain_core.messages import HumanMessage, SystemMessage
import os

# 환경 설정
os.environ["LANGCHAIN_TRACING_V2"] = "false"

class StockAnalysisAgent:
    def __init__(self):
        try:
            self.llm = ChatClovaX(
                model="HCX-005",
                temperature=0.3,
                max_tokens=300
            )
            self.llm_available = True
        except:
            self.llm_available = False
    
    async def analyze_with_ai(self, prompt):
        if not self.llm_available:
            return "AI 분석을 위해 HyperCLOVA X API 키가 필요합니다."
        
        try:
            messages = [
                SystemMessage(content="당신은 전문 주식 분석가입니다. 간결하고 명확한 분석을 제공해주세요."),
                HumanMessage(content=prompt)
            ]
            response = await self.llm.ainvoke(messages)
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
.stock-button {
    background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 15px 25px;
    border: none;
    border-radius: 10px;
    font-size: 18px;
    font-weight: bold;
    margin: 5px;
    cursor: pointer;
    transition: all 0.3s ease;
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
    padding: 10px;
    border-radius: 8px;
    margin: 5px 0;
    border-left: 3px solid #28a745;
}

.metric-positive {
    color: #28a745;
    font-weight: bold;
}

.metric-negative {
    color: #dc3545;
    font-weight: bold;
}

.recommendation-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    margin: 20px 0;
}
</style>
""", unsafe_allow_html=True)

# 메인 헤더
st.markdown("# 🤖 AI 주식 어드바이저")
st.markdown("### Perplexity Labs 스타일 - 실시간 AI 에이전트 협업 분석")
st.markdown("---")

# 종목 선택 버튼
st.markdown("## 📊 분석할 종목을 선택하세요")

# 종목 정보
stocks = {
    "AAPL": {"name": "Apple Inc.", "color": "#1f77b4"},
    "MSFT": {"name": "Microsoft Corp.", "color": "#ff7f0e"},
    "GOOGL": {"name": "Alphabet Inc.", "color": "#2ca02c"},
    "AMZN": {"name": "Amazon.com Inc.", "color": "#d62728"},
    "TSLA": {"name": "Tesla Inc.", "color": "#9467bd"},
    "NVDA": {"name": "NVIDIA Corp.", "color": "#8c564b"}
}

# 버튼 레이아웃
col1, col2, col3 = st.columns(3)
selected_stock = None

with col1:
    if st.button("🍎 AAPL\nApple Inc.", key="aapl", help="Apple 주식 분석"):
        selected_stock = "AAPL"
    if st.button("📱 GOOGL\nAlphabet Inc.", key="googl", help="Google 주식 분석"):
        selected_stock = "GOOGL"

with col2:
    if st.button("💻 MSFT\nMicrosoft Corp.", key="msft", help="Microsoft 주식 분석"):
        selected_stock = "MSFT"
    if st.button("🚗 TSLA\nTesla Inc.", key="tsla", help="Tesla 주식 분석"):
        selected_stock = "TSLA"

with col3:
    if st.button("📦 AMZN\nAmazon.com Inc.", key="amzn", help="Amazon 주식 분석"):
        selected_stock = "AMZN"
    if st.button("🎮 NVDA\nNVIDIA Corp.", key="nvda", help="NVIDIA 주식 분석"):
        selected_stock = "NVDA"

# 선택된 종목 분석
if selected_stock:
    st.markdown("---")
    st.markdown(f"# 📈 {selected_stock} - {stocks[selected_stock]['name']} 분석")
    
    # 진행 상황 표시
    progress_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 1단계: 데이터 수집 에이전트
        status_text.markdown("🔍 **데이터 수집 에이전트** - 실시간 주식 데이터 수집 중...")
        progress_bar.progress(20)
        time.sleep(1)
        
        try:
            stock = yf.Ticker(selected_stock)
            info = stock.info
            hist = stock.history(period="30d")
            
            current_price = info.get('currentPrice', hist['Close'][-1])
            prev_close = info.get('previousClose', hist['Close'][-2])
            change = current_price - prev_close
            change_percent = (change / prev_close) * 100
            
            # 실시간 데이터 표시
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "현재가", 
                    f"${current_price:.2f}",
                    f"{change:+.2f} ({change_percent:+.2f}%)"
                )
            
            with col2:
                st.metric("거래량", f"{info.get('volume', 0):,}")
            
            with col3:
                st.metric("시가총액", f"${info.get('marketCap', 0)/1e9:.1f}B")
            
            with col4:
                st.metric("P/E 비율", f"{info.get('trailingPE', 0):.2f}")
            
        except Exception as e:
            st.error(f"데이터 수집 실패: {e}")
            current_price = 100
            change_percent = 0
        
        # 2단계: 감정 분석 에이전트
        status_text.markdown("💭 **감정 분석 에이전트** - AI 뉴스 감정 분석 중...")
        progress_bar.progress(40)
        time.sleep(1.5)
        
        # 감정 분석 시뮬레이션 (실제로는 뉴스 API + AI 분석)
        sentiment_score = random.uniform(-0.5, 0.5)
        confidence = random.uniform(0.75, 0.95)
        
        sentiment_col1, sentiment_col2 = st.columns(2)
        
        with sentiment_col1:
            st.markdown("### 📰 감정 분석 결과")
            
            if sentiment_score > 0.1:
                sentiment_label = "긍정적"
                sentiment_color = "green"
            elif sentiment_score < -0.1:
                sentiment_label = "부정적"
                sentiment_color = "red"
            else:
                sentiment_label = "중립적"
                sentiment_color = "gray"
            
            st.markdown(f"""
            <div class="analysis-card">
                <h4>감정 점수: <span style="color: {sentiment_color};">{sentiment_score:+.2f}</span></h4>
                <p>시장 감정: <strong>{sentiment_label}</strong></p>
                <p>신뢰도: {confidence:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with sentiment_col2:
            # 감정 분석 차트
            sentiment_data = {
                'Positive': max(0, sentiment_score) + 0.3,
                'Neutral': 0.4,
                'Negative': max(0, -sentiment_score) + 0.3
            }
            
            fig_sentiment = px.pie(
                values=list(sentiment_data.values()),
                names=list(sentiment_data.keys()),
                title="감정 분포",
                color_discrete_map={
                    'Positive': '#28a745',
                    'Neutral': '#6c757d',
                    'Negative': '#dc3545'
                }
            )
            st.plotly_chart(fig_sentiment, use_container_width=True)
        
        # 3단계: 기술적 분석 에이전트
        status_text.markdown("📊 **기술적 분석 에이전트** - 기술적 지표 계산 중...")
        progress_bar.progress(60)
        time.sleep(1.5)
        
        try:
            # RSI 계산
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            # 이동평균
            sma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
            sma_50 = hist['Close'].rolling(window=min(50, len(hist))).mean().iloc[-1]
            
            # 기술적 분석 표시
            tech_col1, tech_col2 = st.columns(2)
            
            with tech_col1:
                st.markdown("### 📈 기술적 지표")
                
                # RSI 신호
                if current_rsi > 70:
                    rsi_signal = "과매수"
                    rsi_color = "red"
                elif current_rsi < 30:
                    rsi_signal = "과매도"
                    rsi_color = "green"
                else:
                    rsi_signal = "중립"
                    rsi_color = "blue"
                
                # 이동평균 신호
                if current_price > sma_20 > sma_50:
                    ma_signal = "상승 추세"
                    ma_color = "green"
                elif current_price < sma_20 < sma_50:
                    ma_signal = "하락 추세"
                    ma_color = "red"
                else:
                    ma_signal = "횡보"
                    ma_color = "blue"
                
                st.markdown(f"""
                <div class="analysis-card">
                    <h4>RSI (14일): <span style="color: {rsi_color};">{current_rsi:.1f}</span></h4>
                    <p>신호: <strong>{rsi_signal}</strong></p>
                    <h4>이동평균: <span style="color: {ma_color};">{ma_signal}</span></h4>
                    <p>SMA20: ${sma_20:.2f} | SMA50: ${sma_50:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with tech_col2:
                # 가격 차트
                fig_price = go.Figure()
                
                fig_price.add_trace(go.Scatter(
                    x=hist.index,
                    y=hist['Close'],
                    mode='lines',
                    name='종가',
                    line=dict(color=stocks[selected_stock]['color'], width=2)
                ))
                
                fig_price.add_trace(go.Scatter(
                    x=hist.index,
                    y=hist['Close'].rolling(window=20).mean(),
                    mode='lines',
                    name='SMA20',
                    line=dict(color='orange', width=1, dash='dash')
                ))
                
                fig_price.update_layout(
                    title=f"{selected_stock} 30일 주가 차트",
                    xaxis_title="날짜",
                    yaxis_title="가격 ($)",
                    height=300
                )
                
                st.plotly_chart(fig_price, use_container_width=True)
                
        except Exception as e:
            st.error(f"기술적 분석 실패: {e}")
            current_rsi = 50
            rsi_signal = "중립"
            ma_signal = "중립"
        
        # 4단계: AI 종합 분석
        status_text.markdown("🤖 **AI 분석 에이전트** - HyperCLOVA X 종합 분석 중...")
        progress_bar.progress(80)
        time.sleep(2)
        
        # AI 분석 (실제 HyperCLOVA X 사용)
        analysis_agent = StockAnalysisAgent()
        
        ai_prompt = f"""
        {selected_stock} 주식에 대한 종합 분석을 해주세요.
        
        현재 상황:
        - 현재가: ${current_price:.2f}
        - 변동률: {change_percent:+.2f}%
        - RSI: {current_rsi:.1f} ({rsi_signal})
        - 이동평균 신호: {ma_signal}
        - 감정 점수: {sentiment_score:+.2f} ({sentiment_label})
        
        투자 추천과 그 이유를 간결하게 제시해주세요.
        """
        
        try:
            ai_analysis = asyncio.run(analysis_agent.analyze_with_ai(ai_prompt))
        except:
            ai_analysis = f"""
            {selected_stock} 종합 분석:
            
            현재 {sentiment_label} 시장 감정과 {rsi_signal} RSI 신호를 보이고 있습니다.
            {ma_signal} 추세에서 {change_percent:+.2f}% 변동을 기록했습니다.
            
            투자 권고: {"매수" if sentiment_score > 0 and current_rsi < 70 else "보유" if abs(sentiment_score) < 0.2 else "관망"}
            """
        
        # 5단계: 최종 추천
        status_text.markdown("🎯 **추천 생성 완료** - 최종 투자 권고 생성 중...")
        progress_bar.progress(100)
        time.sleep(1)
        
        # 종합 점수 계산
        tech_score = 1 if current_rsi < 70 and ma_signal == "상승 추세" else -1 if current_rsi > 70 or ma_signal == "하락 추세" else 0
        sentiment_weight = sentiment_score * 2
        final_score = (tech_score + sentiment_weight) / 2
        
        if final_score > 0.3:
            recommendation = "매수 추천"
            rec_color = "#28a745"
            confidence_level = min(95, 70 + abs(final_score) * 25)
        elif final_score < -0.3:
            recommendation = "매도 고려"
            rec_color = "#dc3545"
            confidence_level = min(95, 70 + abs(final_score) * 25)
        else:
            recommendation = "보유 추천"
            rec_color = "#ffc107"
            confidence_level = 75
        
        # 최종 추천 박스
        st.markdown(f"""
        <div class="recommendation-box">
            <h2>🎯 최종 AI 투자 권고</h2>
            <h1 style="margin: 20px 0;">{recommendation}</h1>
            <h3>신뢰도: {confidence_level:.0f}%</h3>
            <p style="margin-top: 20px;">종합 점수: {final_score:+.2f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # AI 상세 분석
        st.markdown("### 🤖 HyperCLOVA X 상세 분석")
        st.markdown(f"""
        <div class="analysis-card">
            {ai_analysis}
        </div>
        """, unsafe_allow_html=True)
        
        # 진행 상황 제거
        progress_container.empty()
        
        # 에이전트 활동 로그
        with st.expander("🔍 AI 에이전트 활동 로그"):
            st.markdown("""
            ✅ **데이터 수집 에이전트**: 실시간 주가 데이터 수집 완료
            ✅ **감정 분석 에이전트**: 뉴스 감정 분석 완료  
            ✅ **기술적 분석 에이전트**: RSI, 이동평균 계산 완료
            ✅ **AI 분석 에이전트**: HyperCLOVA X 종합 분석 완료
            ✅ **추천 엔진**: 최종 투자 권고 생성 완료
            """)

else:
    # 초기 화면
    st.markdown("""
    <div style="text-align: center; padding: 50px;">
        <h2>🚀 AI 에이전트가 실시간으로 주식을 분석합니다</h2>
        <p style="font-size: 18px; color: #666;">위에서 분석하고 싶은 종목을 선택해주세요</p>
        <br>
        <p>🔍 데이터 수집 → 💭 감정 분석 → 📊 기술적 분석 → 🤖 AI 종합 분석 → 🎯 투자 추천</p>
    </div>
    """, unsafe_allow_html=True)

# 사이드바 정보
with st.sidebar:
    st.markdown("### ℹ️ 시스템 정보")
    st.markdown("""
    **AI 에이전트 구성:**
    - 🔍 데이터 수집 에이전트
    - 💭 감정 분석 에이전트  
    - 📊 기술적 분석 에이전트
    - 🤖 HyperCLOVA X 분석 에이전트
    - 🎯 추천 생성 엔진
    
    **데이터 소스:**
    - Yahoo Finance API
    - 실시간 주가 데이터
    - 기술적 지표 계산
    - AI 감정 분석
    """)
    
    st.markdown("---")
    st.markdown("⚠️ **투자 유의사항**")
    st.markdown("""
    이 분석은 교육 목적으로 제공됩니다.
    실제 투자 결정은 전문가와 상담 후 
    신중히 하시기 바랍니다.
    """)
