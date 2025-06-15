import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import asyncio
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 새로운 모듈들 import
from old_ver_2.enhanced_sentiment_analyzer import EnhancedSentimentAnalyzer
from old_ver_2.advanced_scenario_analyzer import AdvancedScenarioAnalyzer
from old_ver_2.technical_analysis_engine import TechnicalAnalysisEngine
from old_ver_2.data_quality_validator import DataQualityValidator
from old_ver_2.naver_news_collector import NaverNewsCollector

# HyperCLOVA X 클라이언트
from langchain_naver import ChatClovaX
from dotenv import load_dotenv

load_dotenv()


class StockAnalysisAgent:
    def __init__(self):
        self.llm = ChatClovaX(model="HCX-005", temperature=0.3, max_tokens=3000)
        
        # 고도화된 분석 엔진들
        self.sentiment_analyzer = EnhancedSentimentAnalyzer()
        self.scenario_analyzer = AdvancedScenarioAnalyzer()
        self.technical_engine = TechnicalAnalysisEngine()
        self.data_validator = DataQualityValidator()
        self.naver_collector = NaverNewsCollector()
    
    async def analyze_with_ai(self, prompt: str) -> str:
        try:
            response = await self.llm.ainvoke(prompt)
            return response.content
        except Exception as e:
            return f"AI 분석 중 오류 발생: {str(e)}"

def main():
    st.set_page_config(
        page_title="🚀 고도화된 AI 주식 분석 시스템",
        page_icon="📈",
        layout="wide"
    )
    
    # 커스텀 CSS (Perplexity Labs 스타일)
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .analysis-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .progress-container {
        background: #e9ecef;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 메인 헤더
    st.markdown("""
    <div class="main-header">
        <h1>🚀 고도화된 AI 주식 분석 시스템</h1>
        <p>HyperCLOVA X 기반 다중 AI 에이전트 협업 플랫폼</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 사이드바 설정
    with st.sidebar:
        st.header("📊 분석 설정")
        selected_stock = st.text_input("주식 티커", value="AAPL", help="예: AAPL, GOOGL, TSLA")
        
        st.markdown("### 🔧 고도화 기능")
        enable_naver_news = st.checkbox("네이버 뉴스 분석 (30개)", value=True)
        enable_social_analysis = st.checkbox("소셜미디어 + 애널리스트", value=True)
        enable_advanced_technical = st.checkbox("고급 기술적 분석", value=True)
        enable_dynamic_scenarios = st.checkbox("동적 시나리오 생성", value=True)
        enable_data_quality = st.checkbox("데이터 품질 검증", value=True)
        
        analysis_depth = st.selectbox("분석 깊이", ["빠른 분석", "표준 분석", "심화 분석"], index=1)
    
    if st.button("🔍 고도화된 AI 분석 시작", type="primary"):
        # 진행 상황 표시
        progress_container = st.empty()
        status_container = st.empty()
        
        with st.spinner("AI 에이전트들이 협업 중입니다..."):
            asyncio.run(run_enhanced_analysis(
                selected_stock, 
                enable_naver_news,
                enable_social_analysis,
                enable_advanced_technical,
                enable_dynamic_scenarios,
                enable_data_quality,
                analysis_depth,
                progress_container,
                status_container
            ))

async def run_enhanced_analysis(stock_ticker, enable_naver, enable_social, 
                              enable_technical, enable_scenarios, enable_quality,
                              depth, progress_container, status_container):
    
    analysis_agent = StockAnalysisAgent()
    results = {}
    
    # 1단계: 데이터 수집 및 품질 검증
    progress_container.markdown("""
    <div class="progress-container">
        <h4>🔍 1단계: 데이터 수집 및 품질 검증</h4>
        <div style="background: #667eea; height: 10px; width: 20%; border-radius: 5px;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        stock = yf.Ticker(stock_ticker)
        hist_data = stock.history(period="1y")
        stock_info = stock.info
        
        current_price = hist_data['Close'].iloc[-1]
        change_percent = ((current_price - hist_data['Close'].iloc[-2]) / hist_data['Close'].iloc[-2]) * 100
        
        # 데이터 품질 검증
        if enable_quality:
            quality_report = analysis_agent.data_validator.validate_comprehensive(hist_data, stock_info)
            results['quality'] = quality_report
        
    except Exception as e:
        st.error(f"데이터 수집 오류: {e}")
        return
    
    # 2단계: 네이버 뉴스 + 고도화된 감정 분석
    progress_container.markdown("""
    <div class="progress-container">
        <h4>💭 2단계: 다중 소스 감정 분석</h4>
        <div style="background: #667eea; height: 10px; width: 40%; border-radius: 5px;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    if enable_naver:
        # 네이버 뉴스 30개 수집
        naver_news = await analysis_agent.naver_collector.collect_enhanced_news(stock_ticker, count=30)
        results['naver_news'] = naver_news
    
    if enable_social:
        # 소셜미디어 + 애널리스트 분석
        multi_sentiment = await analysis_agent.sentiment_analyzer.analyze_multi_source_sentiment(
            stock_ticker, include_social=True, include_analyst=True
        )
        results['multi_sentiment'] = multi_sentiment
    else:
        # 기본 감정 분석
        sentiment_result = await analysis_agent.sentiment_analyzer.analyze_stock_sentiment(stock_ticker)
        results['sentiment'] = sentiment_result
    
    # 3단계: 고급 기술적 분석
    progress_container.markdown("""
    <div class="progress-container">
        <h4>📊 3단계: 고급 기술적 분석</h4>
        <div style="background: #667eea; height: 10px; width: 60%; border-radius: 5px;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    if enable_technical:
        # 엘리엇 파동, 피보나치, 차트 패턴 분석
        advanced_technical = analysis_agent.technical_engine.comprehensive_analysis(hist_data)
        results['advanced_technical'] = advanced_technical
    else:
        # 기본 기술적 분석
        basic_technical = calculate_basic_technical_indicators(hist_data)
        results['basic_technical'] = basic_technical
    
    # 4단계: 동적 시나리오 분석
    progress_container.markdown("""
    <div class="progress-container">
        <h4>🎯 4단계: 동적 시나리오 분석</h4>
        <div style="background: #667eea; height: 10px; width: 80%; border-radius: 5px;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    if enable_scenarios:
        # 시장 상황 기반 동적 시나리오
        market_regime = analysis_agent.scenario_analyzer.analyze_market_regime(hist_data)
        dynamic_scenarios = analysis_agent.scenario_analyzer.generate_adaptive_scenarios(
            stock_ticker, hist_data, market_regime, results.get('multi_sentiment', results.get('sentiment'))
        )
        results['scenarios'] = dynamic_scenarios
        results['market_regime'] = market_regime
    
    # 5단계: AI 종합 분석
    progress_container.markdown("""
    <div class="progress-container">
        <h4>🤖 5단계: AI 종합 분석</h4>
        <div style="background: #667eea; height: 10px; width: 100%; border-radius: 5px;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # 종합 AI 분석 프롬프트 생성
    comprehensive_prompt = generate_comprehensive_prompt(
        stock_ticker, current_price, change_percent, results, depth
    )
    
    ai_analysis = await analysis_agent.analyze_with_ai(comprehensive_prompt)
    results['ai_analysis'] = ai_analysis
    
    # 결과 표시
    display_enhanced_results(stock_ticker, hist_data, results, progress_container)

def generate_comprehensive_prompt(ticker, price, change, results, depth):
    """고도화된 AI 분석 프롬프트 생성"""
    
    base_info = f"""
{ticker} 주식에 대한 종합 투자 분석을 수행해주세요.

**현재 시장 상황:**
- 현재가: ${price:.2f}
- 일일 변동률: {change:+.2f}%
"""
    
    # 데이터 품질 정보 추가
    if 'quality' in results:
        quality = results['quality']
        base_info += f"""
**데이터 품질 현황:**
- 품질 점수: {quality.get('overall_score', 'N/A')}/100
- 신뢰도: {quality.get('reliability', 'N/A')}
"""
    
    # 다중 감정 분석 정보
    if 'multi_sentiment' in results:
        sentiment = results['multi_sentiment']
        base_info += f"""
**다중 소스 감정 분석:**
- 뉴스 감정: {sentiment.get('news_sentiment', {}).get('label', 'N/A')}
- 소셜미디어 감정: {sentiment.get('social_sentiment', {}).get('label', 'N/A')}
- 애널리스트 평가: {sentiment.get('analyst_sentiment', {}).get('average_rating', 'N/A')}
- 종합 신뢰도: {sentiment.get('confidence', 0):.1%}
"""
    
    # 고급 기술적 분석 정보
    if 'advanced_technical' in results:
        tech = results['advanced_technical']
        base_info += f"""
**고급 기술적 분석:**
- 엘리엇 파동: {tech.get('elliott_wave', {}).get('current_wave', 'N/A')}
- 피보나치 레벨: {tech.get('fibonacci', {}).get('key_level', 'N/A')}
- 차트 패턴: {', '.join(tech.get('chart_patterns', []))}
- 기술적 신호: {tech.get('overall_signal', 'N/A')}
"""
    
    # 시나리오 분석 정보
    if 'scenarios' in results:
        scenarios = results['scenarios']
        market_regime = results.get('market_regime', {})
        base_info += f"""
**동적 시나리오 분석:**
- 시장 체제: {market_regime.get('regime', 'N/A')}
- 변동성 수준: {market_regime.get('volatility_level', 'N/A')}
- 주요 시나리오: {scenarios.get('primary_scenario', {}).get('name', 'N/A')}
- 시나리오 신뢰도: {scenarios.get('confidence', 0):.1%}
"""
    
    # 분석 깊이에 따른 요청사항
    if depth == "심화 분석":
        analysis_request = """
**심화 분석 요청사항:**
1. 거시경제 환경과 섹터 분석
2. 경쟁사 대비 상대적 밸류에이션
3. 리스크 요인별 시나리오 분석
4. 포트폴리오 내 비중 및 헤지 전략
5. 장단기 투자 전략 구분
6. ESG 요인 및 지속가능성 평가
"""
    elif depth == "표준 분석":
        analysis_request = """
**표준 분석 요청사항:**
1. 현재 상황 종합 평가
2. 주요 강점과 위험 요소
3. 투자 추천 (매수/보유/매도)
4. 목표 가격대 및 근거
5. 투자 기간 권장사항
"""
    else:
        analysis_request = """
**빠른 분석 요청사항:**
1. 핵심 투자 포인트 3가지
2. 명확한 투자 추천
3. 주요 리스크 1-2가지
"""
    
    return base_info + analysis_request

def display_enhanced_results(ticker, hist_data, results, progress_container):
    """고도화된 결과 표시"""
    
    progress_container.empty()
    
    # 데이터 품질 대시보드
    if 'quality' in results:
        st.markdown("## 📋 데이터 품질 대시보드")
        quality = results['quality']
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{quality.get('overall_score', 0)}</h3>
                <p>품질 점수</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{quality.get('completeness', 0):.1%}</h3>
                <p>완전성</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{quality.get('accuracy', 0):.1%}</h3>
                <p>정확성</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{quality.get('reliability', 'N/A')}</h3>
                <p>신뢰도</p>
            </div>
            """, unsafe_allow_html=True)
    
    # 고급 기술적 분석 차트
    if 'advanced_technical' in results:
        st.markdown("## 📈 고급 기술적 분석 차트")
        display_advanced_technical_chart(hist_data, results['advanced_technical'])
    
    # 다중 감정 분석 대시보드
    if 'multi_sentiment' in results:
        st.markdown("## 💭 다중 소스 감정 분석")
        display_multi_sentiment_dashboard(results['multi_sentiment'])
    
    # 네이버 뉴스 분석
    if 'naver_news' in results:
        st.markdown("## 📰 네이버 뉴스 분석 (30개)")
        display_naver_news_analysis(results['naver_news'])
    
    # 동적 시나리오 분석
    if 'scenarios' in results:
        st.markdown("## 🎯 동적 시나리오 분석")
        display_dynamic_scenarios(results['scenarios'], results.get('market_regime'))
    
    # AI 종합 분석
    st.markdown("## 🤖 AI 종합 분석")
    st.markdown(f"""
    <div class="analysis-card">
        {results.get('ai_analysis', '분석 결과가 없습니다.')}
    </div>
    """, unsafe_allow_html=True)

def display_advanced_technical_chart(hist_data, technical_results):
    """고급 기술적 분석 차트 표시"""
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('주가 + 피보나치 + 엘리엇 파동', '기술적 지표', '거래량'),
        row_heights=[0.6, 0.25, 0.15]
    )
    
    # 캔들스틱 차트
    fig.add_trace(
        go.Candlestick(
            x=hist_data.index,
            open=hist_data['Open'],
            high=hist_data['High'],
            low=hist_data['Low'],
            close=hist_data['Close'],
            name="주가"
        ),
        row=1, col=1
    )
    
    # 피보나치 레벨
    if 'fibonacci' in technical_results:
        fib_data = technical_results['fibonacci']
        colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
        levels = ['0%', '23.6%', '38.2%', '50%', '61.8%', '100%']
        
        for i, (level, color) in enumerate(zip(levels, colors)):
            if f'fib_{level}' in fib_data:
                fig.add_hline(
                    y=fib_data[f'fib_{level}'],
                    line_dash="dash",
                    line_color=color,
                    annotation_text=f"Fib {level}",
                    row=1, col=1
                )
    
    # 엘리엇 파동 포인트
    if 'elliott_wave' in technical_results:
        wave_points = technical_results['elliott_wave'].get('wave_points', [])
        if wave_points:
            wave_x = [point['date'] for point in wave_points]
            wave_y = [point['price'] for point in wave_points]
            
            fig.add_trace(
                go.Scatter(
                    x=wave_x,
                    y=wave_y,
                    mode='markers+lines',
                    name='Elliott Waves',
                    line=dict(color='purple', width=2),
                    marker=dict(size=8, color='purple')
                ),
                row=1, col=1
            )
    
    # RSI
    if 'rsi' in technical_results:
        fig.add_trace(
            go.Scatter(
                x=hist_data.index,
                y=technical_results['rsi'],
                name='RSI',
                line=dict(color='orange')
            ),
            row=2, col=1
        )
        
        # RSI 과매수/과매도 선
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # 거래량
    fig.add_trace(
        go.Bar(
            x=hist_data.index,
            y=hist_data['Volume'],
            name="거래량",
            marker_color='lightblue'
        ),
        row=3, col=1
    )
    
    fig.update_layout(
        title=f"고급 기술적 분석 차트",
        height=800,
        xaxis_rangeslider_visible=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 차트 패턴 표시
    if 'chart_patterns' in technical_results:
        patterns = technical_results['chart_patterns']
        if patterns:
            st.subheader("🔍 감지된 차트 패턴")
            for pattern in patterns:
                st.success(f"✅ {pattern['name']} 패턴 (신뢰도: {pattern.get('confidence', 0):.1%})")

def display_multi_sentiment_dashboard(sentiment_data):
    """다중 소스 감정 분석 대시보드"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 감정 분석 요약")
        
        # 뉴스 감정
        news_sentiment = sentiment_data.get('news_sentiment', {})
        st.write(f"**뉴스 감정**: {news_sentiment.get('label', 'N/A')} ({news_sentiment.get('score', 0):+.2f})")
        
        # 소셜미디어 감정
        social_sentiment = sentiment_data.get('social_sentiment', {})
        st.write(f"**소셜미디어**: {social_sentiment.get('label', 'N/A')} ({social_sentiment.get('score', 0):+.2f})")
        
        # 애널리스트 평가
        analyst_sentiment = sentiment_data.get('analyst_sentiment', {})
        st.write(f"**애널리스트 평가**: {analyst_sentiment.get('average_rating', 'N/A')}")
        
        # 종합 신뢰도
        st.write(f"**종합 신뢰도**: {sentiment_data.get('confidence', 0):.1%}")
    
    with col2:
        st.subheader("📈 감정 점수 분포")
        
        # 감정 점수 차트
        sources = ['뉴스', '소셜미디어', '애널리스트']
        scores = [
            news_sentiment.get('score', 0),
            social_sentiment.get('score', 0),
            analyst_sentiment.get('normalized_score', 0)
        ]
        
        fig = go.Figure(data=[
            go.Bar(x=sources, y=scores, 
                  marker_color=['green' if s > 0 else 'red' if s < 0 else 'gray' for s in scores])
        ])
        fig.update_layout(title="소스별 감정 점수", yaxis_title="감정 점수")
        st.plotly_chart(fig, use_container_width=True)

def display_naver_news_analysis(naver_news):
    """네이버 뉴스 분석 표시"""
    
    st.write(f"총 **{len(naver_news)}개**의 네이버 뉴스를 분석했습니다.")
    
    # 뉴스 감정 분포
    if naver_news:
        sentiments = [news.get('sentiment', 'neutral') for news in naver_news]
        sentiment_counts = pd.Series(sentiments).value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure(data=[go.Pie(labels=sentiment_counts.index, values=sentiment_counts.values)])
            fig.update_layout(title="뉴스 감정 분포")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📰 주요 뉴스 헤드라인")
            for i, news in enumerate(naver_news[:5]):
                sentiment_emoji = "🟢" if news.get('sentiment') == 'positive' else "🔴" if news.get('sentiment') == 'negative' else "🟡"
                st.write(f"{sentiment_emoji} {news.get('title', 'N/A')[:60]}...")

def display_dynamic_scenarios(scenarios, market_regime):
    """동적 시나리오 분석 표시"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌡️ 현재 시장 체제")
        if market_regime:
            st.write(f"**시장 체제**: {market_regime.get('regime', 'N/A')}")
            st.write(f"**변동성 수준**: {market_regime.get('volatility_level', 'N/A')}")
            st.write(f"**트렌드 방향**: {market_regime.get('trend_direction', 'N/A')}")
            st.write(f"**체제 신뢰도**: {market_regime.get('confidence', 0):.1%}")
    
    with col2:
        st.subheader("🎯 시나리오 확률")
        if 'scenario_probabilities' in scenarios:
            probs = scenarios['scenario_probabilities']
            fig = go.Figure(data=[
                go.Pie(labels=list(probs.keys()), values=list(probs.values()))
            ])
            fig.update_layout(title="시나리오별 확률 분포")
            st.plotly_chart(fig, use_container_width=True)
    
    # 시나리오 상세 정보
    if 'detailed_scenarios' in scenarios:
        st.subheader("📊 시나리오 상세 분석")
        for scenario in scenarios['detailed_scenarios']:
            with st.expander(f"📈 {scenario.get('name', 'Unknown')} (확률: {scenario.get('probability', 0):.1%})"):
                st.write(f"**예상 수익률**: {scenario.get('expected_return', 0):+.1%}")
                st.write(f"**리스크 수준**: {scenario.get('risk_level', 'N/A')}")
                st.write(f"**주요 동인**: {scenario.get('key_drivers', 'N/A')}")
                st.write(f"**투자 전략**: {scenario.get('strategy', 'N/A')}")

def calculate_basic_technical_indicators(hist_data):
    """기본 기술적 지표 계산"""
    
    # RSI 계산
    delta = hist_data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # 이동평균
    ma_20 = hist_data['Close'].rolling(20).mean()
    ma_50 = hist_data['Close'].rolling(50).mean()
    
    # MACD
    exp1 = hist_data['Close'].ewm(span=12).mean()
    exp2 = hist_data['Close'].ewm(span=26).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9).mean()
    
    return {
        'rsi': rsi.iloc[-1],
        'ma_20': ma_20.iloc[-1],
        'ma_50': ma_50.iloc[-1],
        'macd': macd.iloc[-1],
        'signal': signal.iloc[-1],
        'rsi_signal': 'oversold' if rsi.iloc[-1] < 30 else 'overbought' if rsi.iloc[-1] > 70 else 'neutral',
        'ma_signal': 'bullish' if ma_20.iloc[-1] > ma_50.iloc[-1] else 'bearish'
    }

if __name__ == "__main__":
    main()
