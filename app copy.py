# app.py
import streamlit as st
import asyncio
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime
import logging

from workflow import InvestmentWorkflow
from agents.aimodels import UserProfile
from agents.personalization_agent import PersonalizationAgent

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 페이지 설정
st.set_page_config(
    page_title="AI 주식 투자 어드바이저",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 세션 상태 초기화
if 'workflow' not in st.session_state:
    st.session_state.workflow = InvestmentWorkflow()
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

def main():
    st.title("🤖 AI 주식 투자 어드바이저")
    st.markdown("---")
    
    # 사이드바 - 사용자 입력
    with st.sidebar:
        st.header("📋 투자자 프로필")
        
        # 기본 정보
        user_id = st.text_input("사용자 ID", value="user_001")
        age = st.number_input("나이", min_value=18, max_value=100, value=35)
        income = st.number_input("연소득 (만원)", min_value=0, value=5000)
        net_worth = st.number_input("순자산 (만원)", min_value=0, value=10000)
        
        # 투자 성향
        risk_tolerance = st.selectbox(
            "위험 성향",
            ["conservative", "moderate", "aggressive"],
            index=1
        )
        
        investment_horizon = st.selectbox(
            "투자 기간",
            ["1y", "3y", "5y", "10y+"],
            index=2
        )
        
        # 관심 종목
        st.subheader("📊 분석할 종목")
        default_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        tickers_input = st.text_area(
            "종목 코드 (쉼표로 구분)",
            value=", ".join(default_tickers)
        )
        tickers = [ticker.strip().upper() for ticker in tickers_input.split(",")]
        
        # 섹터 선호도
        sector_preferences = st.multiselect(
            "선호 섹터",
            ["technology", "healthcare", "finance", "energy", "consumer"],
            default=["technology"]
        )
        
        # 분석 실행 버튼
        if st.button("🚀 AI 분석 시작", type="primary"):
            run_analysis(user_id, age, income, net_worth, risk_tolerance, 
                        investment_horizon, tickers, sector_preferences)
    
    # 메인 영역 - 결과 표시
    if st.session_state.analysis_results:
        display_results(st.session_state.analysis_results)
    else:
        display_welcome_screen()

def run_analysis(user_id, age, income, net_worth, risk_tolerance, 
                investment_horizon, tickers, sector_preferences):
    """분석 실행"""
    
    # 사용자 프로필 생성
    user_profile = UserProfile(
        user_id=user_id,
        age=age,
        income=income,
        net_worth=net_worth,
        risk_tolerance=risk_tolerance,
        investment_horizon=investment_horizon,
        investment_goals=[],
        sector_preferences=sector_preferences,
        current_portfolio={}
    )
    
    # 개인화 에이전트에 프로필 등록
    st.session_state.workflow.personalization_agent.update_user_profile(user_profile)
    
    user_preferences = {
        'risk_tolerance': risk_tolerance,
        'investment_horizon': investment_horizon,
        'sector_preferences': sector_preferences,
        'age': age,
        'income': income,
        'net_worth': net_worth
    }
    
    # 진행 상황 표시
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        with st.spinner("AI 분석 진행 중..."):
            # 비동기 분석 실행
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            status_text.text("📊 시장 데이터 수집 중...")
            progress_bar.progress(20)
            
            result = loop.run_until_complete(
                st.session_state.workflow.run_analysis(
                    user_id, tickers, user_preferences
                )
            )
            
            progress_bar.progress(100)
            status_text.text("✅ 분석 완료!")
            
            st.session_state.analysis_results = result
            st.rerun()
            
    except Exception as e:
        st.error(f"분석 중 오류가 발생했습니다: {str(e)}")
        logger.error(f"분석 오류: {e}")

def display_welcome_screen():
    """환영 화면 표시"""
    st.markdown("""
    ## 🎯 AI 기반 개인 맞춤형 투자 분석 서비스
    
    ### 🔍 주요 기능
    - **실시간 시장 데이터 분석**: 최신 주가 및 거래량 정보
    - **감정 분석**: 뉴스 및 소셜미디어 심리 분석
    - **기술적 분석**: RSI, MACD, 볼린저 밴드 등 50여 개 지표
    - **리스크 관리**: 포트폴리오 최적화 및 VaR 계산
    - **개인화 추천**: 투자 성향 기반 맞춤형 전략
    
    ### 📋 사용 방법
    1. 왼쪽 사이드바에서 투자자 프로필을 입력하세요
    2. 분석할 종목 코드를 입력하세요
    3. "AI 분석 시작" 버튼을 클릭하세요
    4. 분석 결과를 확인하고 투자 결정에 활용하세요
    
    ### ⚠️ 주의사항
    이 서비스는 교육 및 참고 목적으로 제공됩니다. 
    실제 투자 결정은 전문가와 상담 후 신중히 하시기 바랍니다.
    """)

def display_results(results):
    """분석 결과 표시"""
    if 'error' in results:
        st.error(f"분석 오류: {results['error']}")
        return
    
    # 탭 생성
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 투자 추천", "📈 기술적 분석", "💭 감정 분석", 
        "⚠️ 리스크 분석", "📋 상세 데이터"
    ])
    
    with tab1:
        display_recommendations(results)
    
    with tab2:
        display_technical_analysis(results)
    
    with tab3:
        display_sentiment_analysis(results)
    
    with tab4:
        display_risk_analysis(results)
    
    with tab5:
        display_detailed_data(results)

def display_detailed_data(results):
    """상세 데이터 표시"""
    st.header("📋 상세 분석 데이터")
    
    # 종목명 변환 딕셔너리
    ticker_names = {
        "005930.KS": "삼성전자",
        "000660.KS": "SK하이닉스", 
        "035420.KS": "NAVER",
        "051910.KS": "LG화학",
        "207940.KS": "삼성바이오로직스"
    }
    
    # 실시간 데이터
    raw_data = results.get('raw_data', {})
    if raw_data and 'real_time' in raw_data:
        st.subheader("📊 실시간 시장 데이터")
        
        real_time_df = pd.DataFrame([
            {
                '종목명': ticker_names.get(ticker, ticker),
                '종목코드': ticker,
                '현재가': f"{data.get('current_price', 0):,.0f}원",
                '거래량': f"{data.get('volume', 0):,}주",
                '시가총액': f"{data.get('market_cap', 0):,}원",
                'PER': f"{data.get('pe_ratio', 0):.2f}" if data.get('pe_ratio') else 'N/A',
                '배당수익률': f"{data.get('dividend_yield', 0):.2%}" if data.get('dividend_yield') else 'N/A',
                '베타': f"{data.get('beta', 0):.2f}" if data.get('beta') else 'N/A'
            }
            for ticker, data in raw_data['real_time'].items()
        ])
        
        st.dataframe(real_time_df, use_container_width=True)
    
    # 과거 주가 데이터
    historical_data = results.get('historical_data', {})
    if historical_data:
        st.subheader("📈 과거 주가 데이터")
        
        selected_ticker = st.selectbox(
            "상세 조회할 종목 선택",
            list(historical_data.keys()),
            format_func=lambda x: f"{ticker_names.get(x, x)} ({x})"
        )
        
        if selected_ticker and selected_ticker in historical_data:
            df = historical_data[selected_ticker]
            
            if not df.empty:
                # 최근 30일 데이터만 표시
                recent_df = df.tail(30).copy()
                recent_df.index = recent_df.index.strftime('%Y-%m-%d')
                
                # 컬럼명 한글화
                display_df = recent_df.rename(columns={
                    'Open': '시가',
                    'High': '고가', 
                    'Low': '저가',
                    'Close': '종가',
                    'Volume': '거래량'
                })
                
                # 숫자 포맷팅
                for col in ['시가', '고가', '저가', '종가']:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:,.0f}원")
                
                if '거래량' in display_df.columns:
                    display_df['거래량'] = display_df['거래량'].apply(lambda x: f"{x:,}주")
                
                st.dataframe(display_df, use_container_width=True)
                
                # 주가 차트
                fig = go.Figure()
                
                fig.add_trace(go.Candlestick(
                    x=df.tail(30).index,
                    open=df.tail(30)['Open'],
                    high=df.tail(30)['High'],
                    low=df.tail(30)['Low'],
                    close=df.tail(30)['Close'],
                    name=ticker_names.get(selected_ticker, selected_ticker)
                ))
                
                fig.update_layout(
                    title=f"{ticker_names.get(selected_ticker, selected_ticker)} 최근 30일 주가 차트",
                    xaxis_title="날짜",
                    yaxis_title="주가 (원)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"{selected_ticker} 데이터가 없습니다.")
    
    # 뉴스 데이터
    if raw_data and 'news' in raw_data:
        st.subheader("📰 관련 뉴스")
        
        news_data = raw_data['news']
        if news_data:
            for article in news_data[:10]:  # 최대 10개 뉴스
                with st.expander(f"📰 {article.get('title', 'N/A')}"):
                    st.write(f"**종목:** {ticker_names.get(article.get('ticker', ''), article.get('ticker', 'N/A'))}")
                    st.write(f"**요약:** {article.get('summary', 'N/A')}")
                    if article.get('url'):
                        st.write(f"**링크:** [기사 보기]({article['url']})")
        else:
            st.info("수집된 뉴스가 없습니다.")
    
    # 분석 메타데이터
    st.subheader("🔍 분석 정보")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("분석 종목 수", len(results.get('tickers', [])))
    
    with col2:
        timestamp = results.get('timestamp', '')
        if timestamp:
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                formatted_time = dt.strftime('%Y-%m-%d %H:%M')
                st.metric("분석 시점", formatted_time)
            except:
                st.metric("분석 시점", timestamp[:16])
    
    with col3:
        # 데이터 품질 점수 (간단한 계산)
        quality_score = 0
        if raw_data:
            quality_score += 30
        if historical_data:
            quality_score += 40
        if results.get('sentiment_data'):
            quality_score += 30
        
        st.metric("데이터 품질", f"{quality_score}%")
    
    # JSON 원본 데이터 (개발자용)
    with st.expander("🔧 원본 데이터 (JSON)"):
        st.json({
            'raw_data_keys': list(raw_data.keys()) if raw_data else [],
            'historical_data_tickers': list(historical_data.keys()) if historical_data else [],
            'analysis_timestamp': results.get('timestamp', 'N/A'),
            'user_preferences': results.get('user_preferences', {})
        })


def display_recommendations(results):
    """투자 추천 표시"""
    recommendations = results.get('recommendations', {})
    
    if not recommendations or 'recommendations' not in recommendations:
        st.warning("추천 데이터가 없습니다.")
        return
    
    st.header("🎯 AI 투자 추천")
    
    portfolio = recommendations['recommendations']
    
    # 포트폴리오 파이 차트
    if portfolio:
        fig = go.Figure(data=[go.Pie(
            labels=list(portfolio.keys()),
            values=[data['weight'] for data in portfolio.values()],
            hole=0.3
        )])
        fig.update_layout(
            title="추천 포트폴리오 구성",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 추천 종목 테이블
        portfolio_df = pd.DataFrame([
            {
                '종목': ticker,
                '비중': f"{data['weight']:.1%}",
                '추천 이유': data.get('reasoning', 'N/A')
            }
            for ticker, data in portfolio.items()
        ])
        
        st.subheader("📋 추천 종목 상세")
        st.dataframe(portfolio_df, use_container_width=True)
        
        # 추천 이유
        if 'reasoning' in recommendations:
            st.subheader("💡 추천 근거")
            for reason in recommendations['reasoning']:
                st.write(f"• {reason}")
        
        # 예상 결과
        if 'expected_outcomes' in recommendations:
            outcomes = recommendations['expected_outcomes']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "예상 연간 수익률",
                    f"{outcomes.get('expected_annual_return', 0):.1%}"
                )
            
            with col2:
                st.metric(
                    "예상 변동성",
                    f"{outcomes.get('expected_volatility', 0):.1%}"
                )
            
            with col3:
                st.metric(
                    "손실 확률",
                    f"{outcomes.get('probability_of_loss', 0):.1%}"
                )

def display_technical_analysis(results):
    """기술적 분석 표시"""
    technical_analysis = results.get('technical_analysis', {})
    
    if not technical_analysis:
        st.warning("기술적 분석 데이터가 없습니다.")
        return
    
    st.header("📈 기술적 분석")
    
    # 종목별 신호 요약
    signals_data = []
    for ticker, data in technical_analysis.items():
        signals_data.append({
            '종목': ticker,
            '전체 신호': data.get('overall_signal', 'N/A'),
            '변동성': f"{data.get('volatility', 0):.2%}",
            '트렌드 강도': f"{data.get('trend_strength', 0):.2f}"
        })
    
    if signals_data:
        signals_df = pd.DataFrame(signals_data)
        st.dataframe(signals_df, use_container_width=True)
        
        # 개별 종목 상세 분석
        selected_ticker = st.selectbox(
            "상세 분석할 종목 선택",
            list(technical_analysis.keys())
        )
        
        if selected_ticker in technical_analysis:
            ticker_data = technical_analysis[selected_ticker]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"{selected_ticker} 기술적 지표")
                signals = ticker_data.get('signals', {})
                for indicator, signal in signals.items():
                    color = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "🟡"
                    st.write(f"{color} {indicator}: {signal}")
            
            with col2:
                st.subheader("리스크 메트릭")
                st.metric("변동성", f"{ticker_data.get('volatility', 0):.2%}")
                st.metric("트렌드 강도", f"{ticker_data.get('trend_strength', 0):.2f}")

def display_sentiment_analysis(results):
    """감정 분석 표시"""
    sentiment_data = results.get('sentiment_data', {})
    
    if not sentiment_data:
        st.warning("감정 분석 데이터가 없습니다.")
        return
    
    st.header("💭 시장 감정 분석")
    
    # 감정 점수 차트
    if sentiment_data:
        tickers = list(sentiment_data.keys())
        sentiment_scores = [data.get('sentiment_score', 0) for data in sentiment_data.values()]
        
        fig = go.Figure(data=[
            go.Bar(
                x=tickers,
                y=sentiment_scores,
                marker_color=['green' if score > 0 else 'red' for score in sentiment_scores]
            )
        ])
        fig.update_layout(
            title="종목별 감정 점수",
            xaxis_title="종목",
            yaxis_title="감정 점수",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 감정 분석 테이블
        sentiment_df = pd.DataFrame([
            {
                '종목': ticker,
                '감정 점수': f"{data.get('sentiment_score', 0):.3f}",
                '신뢰도': f"{data.get('confidence', 0):.2%}",
                '뉴스 수': data.get('article_count', 0)
            }
            for ticker, data in sentiment_data.items()
        ])
        
        st.dataframe(sentiment_df, use_container_width=True)

def display_risk_analysis(results):
    """리스크 분석 표시"""
    risk_analysis = results.get('risk_analysis', {})
    
    if not risk_analysis or 'error' in risk_analysis:
        st.warning("리스크 분석 데이터가 없습니다.")
        return
    
    st.header("⚠️ 포트폴리오 리스크 분석")
    
    # 최적화 결과
    if 'optimization' in risk_analysis:
        opt_data = risk_analysis['optimization']

main()