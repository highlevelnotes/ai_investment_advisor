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
    page_title="HyperCLOVA X AI 주식 투자 어드바이저",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 세션 상태 초기화
if 'workflow' not in st.session_state:
    st.session_state.workflow = InvestmentWorkflow()
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

def main():
    st.title("🤖 HyperCLOVA X AI 주식 투자 어드바이저")
    st.markdown("**네이버 HyperCLOVA X 기반 한국형 AI 투자 분석 서비스**")
    st.markdown("---")
    
    # 사이드바 - 사용자 입력
    with st.sidebar:
        st.header("📋 투자자 프로필")
        
        # 기본 정보
        user_id = st.text_input("사용자 ID", value="korean_investor_001")
        age = st.number_input("나이", min_value=18, max_value=100, value=35)
        income = st.number_input("연소득 (만원)", min_value=0, value=5000)
        net_worth = st.number_input("순자산 (만원)", min_value=0, value=10000)
        
        # 투자 성향
        risk_tolerance = st.selectbox(
            "위험 성향",
            ["conservative", "moderate", "aggressive"],
            index=1,
            format_func=lambda x: {"conservative": "보수적", "moderate": "중도적", "aggressive": "공격적"}[x]
        )
        
        investment_horizon = st.selectbox(
            "투자 기간",
            ["1y", "3y", "5y", "10y+"],
            index=2,
            format_func=lambda x: {"1y": "1년", "3y": "3년", "5y": "5년", "10y+": "10년 이상"}[x]
        )
        
        # 관심 종목 (한국 주식)
        st.subheader("📊 분석할 한국 주식")
        default_korean_tickers = ["005930.KS", "000660.KS", "035420.KS", "051910.KS", "207940.KS"]
        ticker_names = {
            "005930.KS": "삼성전자",
            "000660.KS": "SK하이닉스", 
            "035420.KS": "NAVER",
            "051910.KS": "LG화학",
            "207940.KS": "삼성바이오로직스"
        }
        
        selected_tickers = st.multiselect(
            "종목 선택",
            options=default_korean_tickers,
            default=default_korean_tickers,
            format_func=lambda x: f"{ticker_names.get(x, x)} ({x})"
        )
        
        # 추가 종목 입력
        additional_tickers = st.text_input(
            "추가 종목 (쉼표로 구분, 예: 005380.KS,012330.KS)",
            help="한국 주식은 .KS를 붙여주세요"
        )
        
        if additional_tickers:
            additional_list = [ticker.strip() for ticker in additional_tickers.split(",")]
            selected_tickers.extend(additional_list)
        
        # 섹터 선호도
        sector_preferences = st.multiselect(
            "선호 섹터",
            ["기술", "금융", "화학", "자동차", "바이오"],
            default=["기술"]
        )
        
        # 분석 실행 버튼
        if st.button("🚀 HyperCLOVA X AI 분석 시작", type="primary"):
            if selected_tickers:
                run_analysis(user_id, age, income, net_worth, risk_tolerance, 
                           investment_horizon, selected_tickers, sector_preferences)
            else:
                st.error("분석할 종목을 선택해주세요.")
    
    # 메인 영역 - 결과 표시
    if st.session_state.analysis_results:
        display_results(st.session_state.analysis_results)
    else:
        display_welcome_screen()

def display_welcome_screen():
    """환영 화면 표시"""
    st.markdown("""
    ## 🎯 HyperCLOVA X 기반 한국형 AI 투자 분석 서비스
    
    ### 🔍 주요 특징
    - **🤖 HyperCLOVA X 활용**: 네이버의 한국어 특화 AI 모델 사용
    - **🇰🇷 한국 시장 특화**: 한국 주식시장과 문화에 최적화된 분석
    - **📊 실시간 분석**: 최신 주가 및 뉴스 감정 분석
    - **🎯 개인화 추천**: 투자 성향 기반 맞춤형 포트폴리오
    - **⚠️ 리스크 관리**: AI 기반 위험 평가 및 관리 방안
    
    ### 📋 사용 방법
    1. 왼쪽 사이드바에서 투자자 정보를 입력하세요
    2. 분석할 한국 주식을 선택하세요
    3. "HyperCLOVA X AI 분석 시작" 버튼을 클릭하세요
    4. AI 분석 결과를 확인하고 투자에 활용하세요
    
    ### 🌟 HyperCLOVA X의 장점
    - **한국어 이해도**: 한국 금융 뉴스와 시장 상황을 정확히 분석
    - **문화적 맥락**: 한국 투자자의 성향과 시장 특성 반영
    - **실시간 처리**: 빠른 응답 속도와 안정적인 서비스
    
    ### ⚠️ 주의사항
    이 서비스는 교육 및 참고 목적으로 제공됩니다. 
    실제 투자 결정은 충분한 검토와 전문가 상담 후 신중히 하시기 바랍니다.
    """)

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
    progress_container = st.container()
    
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            with st.spinner("🤖 HyperCLOVA X AI 분석 진행 중..."):
                # 비동기 분석 실행
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                status_text.text("📊 한국 주식 데이터 수집 중...")
                progress_bar.progress(20)
                
                status_text.text("💭 HyperCLOVA X 뉴스 감정 분석 중...")
                progress_bar.progress(40)
                
                status_text.text("📈 기술적 지표 계산 중...")
                progress_bar.progress(60)
                
                status_text.text("⚠️ 포트폴리오 리스크 분석 중...")
                progress_bar.progress(80)
                
                status_text.text("🎯 개인화 투자 추천 생성 중...")
                progress_bar.progress(90)
                
                result = loop.run_until_complete(
                    st.session_state.workflow.run_analysis(
                        user_id, tickers, user_preferences
                    )
                )
                
                progress_bar.progress(100)
                status_text.text("✅ HyperCLOVA X 분석 완료!")
                
                st.session_state.analysis_results = result
                
                # 성공 메시지
                st.success("🎉 AI 분석이 완료되었습니다! 아래에서 결과를 확인하세요.")
                
                # 진행 상황 표시 제거
                progress_container.empty()
                
                st.rerun()
                
        except Exception as e:
            st.error(f"❌ 분석 중 오류가 발생했습니다: {str(e)}")
            logger.error(f"분석 오류: {e}")

def display_results(results):
    """분석 결과 표시"""
    if 'error' in results:
        st.error(f"❌ 분석 오류: {results['error']}")
        return
    
    # AI 요약 먼저 표시
    if 'recommendations' in results and 'ai_summary' in results['recommendations']:
        st.markdown("## 🤖 HyperCLOVA X AI 종합 분석 요약")
        st.markdown(results['recommendations']['ai_summary'])
        st.markdown("---")
    
    # 탭 생성
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 AI 투자 추천", "📈 기술적 분석", "💭 감정 분석", 
        "⚠️ 리스크 분석", "📋 상세 데이터"
    ])
    
    with tab1:
        display_ai_recommendations(results)
    
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
        
        st.subheader("📊 포트폴리오 최적화 결과")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "예상 연간 수익률",
                f"{opt_data.get('expected_return', 0):.2%}"
            )
        
        with col2:
            st.metric(
                "예상 변동성",
                f"{opt_data.get('volatility', 0):.2%}"
            )
        
        with col3:
            st.metric(
                "샤프 비율",
                f"{opt_data.get('sharpe_ratio', 0):.2f}"
            )
    
    # 리스크 메트릭
    if 'risk_metrics' in risk_analysis:
        risk_metrics = risk_analysis['risk_metrics']
        
        st.subheader("📈 리스크 지표")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "최대 낙폭 (MDD)",
                f"{risk_metrics.get('max_drawdown', 0):.2%}"
            )
            st.metric(
                "베타",
                f"{risk_metrics.get('beta', 0):.2f}"
            )
        
        with col2:
            st.metric(
                "소르티노 비율",
                f"{risk_metrics.get('sortino_ratio', 0):.2f}"
            )
            st.metric(
                "정보 비율",
                f"{risk_metrics.get('information_ratio', 0):.2f}"
            )
    
    # VaR 분석
    if 'var_analysis' in risk_analysis:
        var_data = risk_analysis['var_analysis']
        
        st.subheader("💰 VaR (Value at Risk) 분석")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Historical VaR (5%)",
                f"{var_data.get('historical_var_5%', 0):.2%}"
            )
        
        with col2:
            st.metric(
                "Expected Shortfall",
                f"{var_data.get('expected_shortfall', 0):.2%}"
            )
        
        st.info("💡 VaR는 95% 신뢰구간에서 예상되는 최대 손실률을 나타냅니다.")


def display_ai_recommendations(results):
    """AI 투자 추천 표시"""
    recommendations = results.get('recommendations', {})
    
    if not recommendations or 'recommendations' not in recommendations:
        st.warning("AI 추천 데이터가 없습니다.")
        return
    
    st.header("🎯 HyperCLOVA X AI 투자 추천")
    
    portfolio = recommendations['recommendations']
    
    # 포트폴리오 파이 차트
    if portfolio:
        # 종목명 변환
        ticker_names = {
            "005930.KS": "삼성전자",
            "000660.KS": "SK하이닉스", 
            "035420.KS": "NAVER",
            "051910.KS": "LG화학",
            "207940.KS": "삼성바이오로직스"
        }
        
        labels = [ticker_names.get(ticker, ticker) for ticker in portfolio.keys()]
        values = [data['weight'] for data in portfolio.values()]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.3,
            textinfo='label+percent',
            textfont_size=12
        )])
        fig.update_layout(
            title="🥧 AI 추천 포트폴리오 구성",
            height=400,
            font=dict(size=14)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 추천 종목 테이블
        portfolio_df = pd.DataFrame([
            {
                '종목명': ticker_names.get(ticker, ticker),
                '종목코드': ticker,
                '추천 비중': f"{data['weight']:.1%}",
                'AI 분석': data.get('reasoning', 'N/A')[:50] + "..." if len(data.get('reasoning', '')) > 50 else data.get('reasoning', 'N/A')
            }
            for ticker, data in portfolio.items()
        ])
        
        st.subheader("📋 AI 추천 종목 상세")
        st.dataframe(portfolio_df, use_container_width=True)
        
        # AI 추천 이유
        if 'ai_reasoning' in recommendations:
            st.subheader("🤖 HyperCLOVA X 추천 근거")
            for reason in recommendations['ai_reasoning']:
                st.write(f"• {reason}")
        
        # AI 리스크 평가
        if 'risk_assessment' in recommendations:
            risk_assessment = recommendations['risk_assessment']
            
            st.subheader("⚠️ AI 리스크 평가")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "분산화 점수",
                    f"{risk_assessment.get('diversification_score', 0):.1f}/1.0"
                )
            
            with col2:
                alignment = risk_assessment.get('risk_alignment', 'UNKNOWN')
                color = "🟢" if alignment == "ALIGNED" else "🟡" if alignment == "CONSERVATIVE" else "🔴"
                st.metric("위험 성향 적합성", f"{color} {alignment}")
            
            # AI 평가 내용
            if 'ai_assessment' in risk_assessment:
                st.write("**AI 상세 평가:**")
                st.write(risk_assessment['ai_assessment'])

def display_sentiment_analysis(results):
    """감정 분석 표시 (HyperCLOVA X 결과)"""
    sentiment_data = results.get('sentiment_data', {})
    
    if not sentiment_data:
        st.warning("감정 분석 데이터가 없습니다.")
        return
    
    st.header("💭 HyperCLOVA X 시장 감정 분석")
    
    # 종목명 변환
    ticker_names = {
        "005930.KS": "삼성전자",
        "000660.KS": "SK하이닉스", 
        "035420.KS": "NAVER",
        "051910.KS": "LG화학",
        "207940.KS": "삼성바이오로직스"
    }
    
    # 감정 점수 차트
    if sentiment_data:
        tickers = list(sentiment_data.keys())
        ticker_labels = [ticker_names.get(ticker, ticker) for ticker in tickers]
        sentiment_scores = [data.get('sentiment_score', 0) for data in sentiment_data.values()]
        
        fig = go.Figure(data=[
            go.Bar(
                x=ticker_labels,
                y=sentiment_scores,
                marker_color=['green' if score > 0 else 'red' if score < 0 else 'gray' for score in sentiment_scores],
                text=[f"{score:.2f}" for score in sentiment_scores],
                textposition='auto'
            )
        ])
        fig.update_layout(
            title="📊 종목별 AI 감정 점수",
            xaxis_title="종목",
            yaxis_title="감정 점수 (-1.0 ~ +1.0)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # 감정 분석 상세 테이블
        sentiment_df = pd.DataFrame([
            {
                '종목명': ticker_names.get(ticker, ticker),
                '감정 점수': f"{data.get('sentiment_score', 0):.3f}",
                '신뢰도': f"{data.get('confidence', 0):.1%}",
                '뉴스 수': data.get('article_count', 0),
                'AI 분석 요약': data.get('analysis_summary', 'N/A')[:100] + "..." if len(data.get('analysis_summary', '')) > 100 else data.get('analysis_summary', 'N/A')
            }
            for ticker, data in sentiment_data.items()
        ])
        
        st.subheader("📰 HyperCLOVA X 뉴스 감정 분석 상세")
        st.dataframe(sentiment_df, use_container_width=True)
        
        # 감정 분석 방법 설명
        st.info("🤖 **HyperCLOVA X 감정 분석**: 한국어에 특화된 AI 모델을 사용하여 금융 뉴스의 감정을 정확하게 분석합니다.")

# 나머지 함수들은 기존과 동일하게 유지...
def display_technical_analysis(results):
    """기술적 분석 표시"""
    technical_analysis = results.get('technical_analysis', {})
    
    if not technical_analysis:
        st.warning("기술적 분석 데이터가 없습니다.")
        return
    
    st.header("📈 기술적 분석")
    
    # 종목명 변환
    ticker_names = {
        "005930.KS": "삼성전자",
        "000660.KS": "SK하이닉스", 
        "035420.KS": "NAVER",
        "051910.KS": "LG화학",
        "207940.KS": "삼성바이오로직스"
    }
    
    # 종목별 신호 요약
    signals_data = []
    for ticker, data in technical_analysis.items():
        signals_data.append({
            '종목명': ticker_names.get(ticker, ticker),
            '종목코드': ticker,
            '전체 신호': data.get('overall_signal', 'N/A'),
            '변동성': f"{data.get('volatility', 0):.2%}",
            '트렌드 강도': f"{data.get('trend_strength', 0):.2f}"
        })
    
    if signals_data:
        signals_df = pd.DataFrame(signals_data)
        st.dataframe(signals_df, use_container_width=True)

def display_risk_analysis(results):
    """리스크 분석 표시"""
    risk_analysis = results.get('risk_analysis', {})

main()