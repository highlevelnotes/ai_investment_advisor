# main.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 환경변수 기반 설정 로드
from config import Config, validate_config, get_api_status, APP_CONFIG
from data_collector import DataCollector
from ai_analyzer import AIAnalyzer
from portfolio_optimizer import PortfolioOptimizer
from visualization import create_portfolio_pie_chart, create_performance_chart
from utils import calculate_portfolio_metrics, format_currency

from dotenv import load_dotenv

load_dotenv()

# 페이지 설정
st.set_page_config(
    page_title=APP_CONFIG['TITLE'],
    page_icon=APP_CONFIG['PAGE_ICON'],
    layout=APP_CONFIG['LAYOUT']
)

def main():
    # 환경변수 검증
    config_valid = validate_config()
    api_status = get_api_status()
    
    # 헤더
    st.title(APP_CONFIG['TITLE'])
    st.markdown(f"*{APP_CONFIG['DESCRIPTION']}*")
    
    # API 상태 표시
    with st.expander("🔧 시스템 상태", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            status_icon = "✅" if api_status['ecos'] else "❌"
            st.write(f"{status_icon} ECOS API")
        with col2:
            status_icon = "✅" if api_status['hyperclova_x'] else "❌"
            st.write(f"{status_icon} HyperClova X API")
        with col3:
            status_icon = "✅" if api_status['pykrx'] else "❌"
            st.write(f"{status_icon} PyKRX")
        
        if not config_valid:
            st.warning("일부 API 키가 설정되지 않아 샘플 데이터로 동작합니다.")
    
    # 사이드바 - 사용자 프로필
    st.sidebar.header("👤 사용자 프로필")
    
    user_profile = {
        'age': st.sidebar.slider("나이", 20, 70, 35),
        'risk_tolerance': st.sidebar.selectbox(
            "투자성향", 
            ['안정형', '안정추구형', '위험중립형', '적극투자형']
        ),
        'investment_period': st.sidebar.slider("투자기간 (년)", 5, 40, 20),
        'current_assets': st.sidebar.number_input(
            "현재 자산 (만원)", 
            min_value=0, 
            max_value=100000, 
            value=1000,
            step=100
        ) * 10000,
        'monthly_contribution': st.sidebar.number_input(
            "월 납입액 (만원)", 
            min_value=0, 
            max_value=1000, 
            value=50,
            step=10
        ) * 10000
    }
    
    # 데이터 로드
    if 'etf_data' not in st.session_state:
        with st.spinner('📊 ETF 데이터를 수집하고 있습니다...'):
            data_collector = DataCollector()
            st.session_state.etf_data = data_collector.get_etf_data()
            st.session_state.economic_data = data_collector.get_economic_indicators()
            st.session_state.market_data = data_collector.get_market_data()
    
    # 탭 구성
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 대시보드", 
        "📈 시장 분석", 
        "⚖️ 포트폴리오 최적화", 
        "🤖 AI 분석", 
        "📋 백테스팅"
    ])
    
    with tab1:
        show_dashboard(user_profile)
    
    with tab2:
        show_market_analysis()
    
    with tab3:
        show_portfolio_optimization(user_profile)
    
    with tab4:
        show_ai_analysis(user_profile)
    
    with tab5:
        show_backtesting(user_profile)

def show_dashboard(user_profile):
    """대시보드 페이지"""
    st.header("📊 포트폴리오 대시보드")
    
    # 시장 현황
    if 'market_data' in st.session_state:
        market_data = st.session_state.market_data
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            kospi_change_color = "green" if market_data['kospi']['change'] >= 0 else "red"
            st.metric(
                "KOSPI",
                f"{market_data['kospi']['current']:.2f}",
                f"{market_data['kospi']['change']:+.2f} ({market_data['kospi']['change_pct']:+.2f}%)"
            )
        
        with col2:
            kosdaq_change_color = "green" if market_data['kosdaq']['change'] >= 0 else "red"
            st.metric(
                "KOSDAQ",
                f"{market_data['kosdaq']['current']:.2f}",
                f"{market_data['kosdaq']['change']:+.2f} ({market_data['kosdaq']['change_pct']:+.2f}%)"
            )
        
        with col3:
            st.metric(
                "상장 ETF 수",
                f"{market_data['etf_count']}개",
                "PyKRX 기반"
            )
    
    # 사용자 포트폴리오 요약
    st.subheader("💼 내 포트폴리오 현황")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.metric("현재 자산", format_currency(user_profile['current_assets']))
        st.metric("월 납입액", format_currency(user_profile['monthly_contribution']))
        st.metric("투자기간", f"{user_profile['investment_period']}년")
    
    with col2:
        st.metric("투자성향", user_profile['risk_tolerance'])
        st.metric("나이", f"{user_profile['age']}세")
        
        # 예상 은퇴 자산 계산
        expected_assets = calculate_expected_retirement_assets(user_profile)
        st.metric("예상 은퇴자산", format_currency(expected_assets))

def show_market_analysis():
    """시장 분석 페이지"""
    st.header("📈 시장 분석")
    
    if 'etf_data' not in st.session_state:
        st.warning("데이터를 먼저 로드해주세요.")
        return
    
    etf_data = st.session_state.etf_data
    
    # ETF 성과 차트
    st.subheader("📊 ETF 성과 비교")
    performance_chart = create_performance_chart(etf_data)
    st.plotly_chart(performance_chart, use_container_width=True)
    
    # 카테고리별 요약
    st.subheader("📋 카테고리별 현황")
    
    for category, etfs in etf_data.items():
        with st.expander(f"{category} ({len(etfs)}개 ETF)"):
            etf_summary = []
            
            for name, data in etfs.items():
                if 'returns' in data and not data['returns'].empty:
                    annual_return = data['returns'].mean() * 252 * 100
                    annual_vol = data['returns'].std() * np.sqrt(252) * 100
                else:
                    annual_return = 0
                    annual_vol = 0
                
                etf_summary.append({
                    'ETF명': name,
                    '현재가': f"{data['price']:,.0f}원",
                    '연환산수익률': f"{annual_return:.2f}%",
                    '변동성': f"{annual_vol:.2f}%",
                    '거래량': f"{data['volume']:,}주"
                })
            
            summary_df = pd.DataFrame(etf_summary)
            st.dataframe(summary_df, use_container_width=True)

def show_portfolio_optimization(user_profile):
    """포트폴리오 최적화 페이지"""
    st.header("⚖️ 포트폴리오 최적화")
    
    if 'etf_data' not in st.session_state:
        st.warning("데이터를 먼저 로드해주세요.")
        return
    
    optimizer = PortfolioOptimizer()
    etf_data = st.session_state.etf_data
    
    # 최적화 방법 선택
    optimization_method = st.selectbox(
        "최적화 방법 선택",
        ['lifecycle', 'max_sharpe', 'min_variance'],
        format_func=lambda x: {
            'lifecycle': '생애주기별 배분',
            'max_sharpe': '최대 샤프비율',
            'min_variance': '최소분산'
        }[x]
    )
    
    if st.button("포트폴리오 최적화 실행"):
        with st.spinner("최적화를 수행하고 있습니다..."):
            optimal_portfolio = optimizer.optimize_portfolio(
                etf_data, 
                method=optimization_method,
                user_profile=user_profile
            )
            
            if optimal_portfolio:
                st.session_state.optimal_portfolio = optimal_portfolio
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("📊 최적 포트폴리오 구성")
                    pie_chart = create_portfolio_pie_chart(optimal_portfolio['weights'])
                    st.plotly_chart(pie_chart, use_container_width=True)
                
                with col2:
                    st.subheader("📈 예상 성과")
                    st.metric("기대수익률", f"{optimal_portfolio['expected_return']*100:.2f}%")
                    st.metric("변동성", f"{optimal_portfolio['volatility']*100:.2f}%")
                    st.metric("샤프비율", f"{optimal_portfolio['sharpe_ratio']:.3f}")
                    st.metric("최적화 방법", optimal_portfolio['method'])
                
                # 상세 구성
                st.subheader("📋 상세 구성")
                weights_df = pd.DataFrame([
                    {'ETF명': name, '비중': f"{weight*100:.2f}%", '투자금액': format_currency(weight * user_profile['current_assets'])}
                    for name, weight in optimal_portfolio['weights'].items()
                ])
                st.dataframe(weights_df, use_container_width=True)

def show_ai_analysis(user_profile):
    """AI 분석 페이지"""
    st.header("🤖 AI 분석 및 추천")
    
    if 'etf_data' not in st.session_state or 'economic_data' not in st.session_state:
        st.warning("데이터를 먼저 로드해주세요.")
        return
    
    ai_analyzer = AIAnalyzer()
    
    if not ai_analyzer.available:
        st.warning("HyperClova X API가 설정되지 않아 샘플 분석을 제공합니다.")
    
    # 분석 유형 선택
    analysis_type = st.selectbox(
        "분석 유형 선택",
        ['market_analysis', 'portfolio_recommendation', 'performance_analysis'],
        format_func=lambda x: {
            'market_analysis': '시장 상황 분석',
            'portfolio_recommendation': '포트폴리오 추천',
            'performance_analysis': '성과 분석'
        }[x]
    )
    
    if st.button("AI 분석 실행"):
        with st.spinner("AI가 분석하고 있습니다..."):
            if analysis_type == 'market_analysis':
                analysis_result = ai_analyzer.analyze_market_situation(
                    st.session_state.economic_data,
                    st.session_state.etf_data
                )
                st.markdown("### 🏛️ 시장 상황 분석")
                st.markdown(analysis_result)
                
            elif analysis_type == 'portfolio_recommendation':
                recommendation = ai_analyzer.generate_portfolio_recommendation(
                    st.session_state.economic_data,
                    st.session_state.etf_data,
                    user_profile
                )
                st.markdown("### 🎯 개인 맞춤형 포트폴리오 추천")
                st.markdown(recommendation)
                
            elif analysis_type == 'performance_analysis':
                # 샘플 성과 데이터
                portfolio_data = {
                    'return': 7.2,
                    'volatility': 12.5,
                    'sharpe_ratio': 0.52,
                    'max_drawdown': -8.3
                }
                benchmark_data = {
                    'return': 5.4,
                    'volatility': 15.2,
                    'sharpe_ratio': 0.31,
                    'max_drawdown': -12.1
                }
                
                performance_analysis = ai_analyzer.analyze_portfolio_performance(
                    portfolio_data,
                    benchmark_data
                )
                st.markdown("### 📊 포트폴리오 성과 분석")
                st.markdown(performance_analysis)

def show_backtesting(user_profile):
    """백테스팅 페이지"""
    st.header("📋 백테스팅 및 시뮬레이션")
    
    if 'etf_data' not in st.session_state:
        st.warning("데이터를 먼저 로드해주세요.")
        return
    
    st.subheader("⚙️ 백테스팅 설정")
    
    col1, col2 = st.columns(2)
    
    with col1:
        backtest_period = st.selectbox(
            "백테스팅 기간",
            ['1y', '3y', '5y'],
            format_func=lambda x: {'1y': '1년', '3y': '3년', '5y': '5년'}[x]
        )
        
        rebalancing_freq = st.selectbox(
            "리밸런싱 주기",
            ['monthly', 'quarterly', 'annually'],
            format_func=lambda x: {'monthly': '월별', 'quarterly': '분기별', 'annually': '연별'}[x]
        )
    
    with col2:
        initial_investment = st.number_input(
            "초기 투자금액 (만원)",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100
        ) * 10000
        
        monthly_contribution = st.number_input(
            "월 적립금액 (만원)",
            min_value=0,
            max_value=500,
            value=50,
            step=10
        ) * 10000
    
    if st.button("백테스팅 실행"):
        with st.spinner("백테스팅을 실행하고 있습니다..."):
            # 간단한 백테스팅 시뮬레이션
            backtest_results = run_simple_backtest(
                user_profile, 
                initial_investment, 
                monthly_contribution,
                backtest_period
            )
            
            if backtest_results:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("최종 자산", format_currency(backtest_results['final_value']))
                    st.metric("총 수익률", f"{backtest_results['total_return']:.2f}%")
                    st.metric("연평균 수익률", f"{backtest_results['annual_return']:.2f}%")
                
                with col2:
                    st.metric("최대 낙폭", f"{backtest_results['max_drawdown']:.2f}%")
                    st.metric("샤프 비율", f"{backtest_results['sharpe_ratio']:.3f}")
                    st.metric("변동성", f"{backtest_results['volatility']:.2f}%")

def calculate_expected_retirement_assets(user_profile):
    """예상 은퇴자산 계산"""
    current_assets = user_profile['current_assets']
    monthly_contribution = user_profile['monthly_contribution']
    investment_period = user_profile['investment_period']
    
    # 가정: 연 6% 수익률
    annual_return = 0.06
    monthly_return = annual_return / 12
    
    # 복리 계산
    future_value_current = current_assets * (1 + annual_return) ** investment_period
    future_value_monthly = monthly_contribution * (((1 + monthly_return) ** (investment_period * 12) - 1) / monthly_return)
    
    return future_value_current + future_value_monthly

def run_simple_backtest(user_profile, initial_investment, monthly_contribution, period):
    """간단한 백테스팅 실행"""
    # 샘플 백테스팅 결과
    period_years = {'1y': 1, '3y': 3, '5y': 5}[period]
    
    # 가정된 수익률 (실제로는 ETF 데이터 기반으로 계산)
    annual_return = 0.07  # 7%
    volatility = 0.15     # 15%
    
    # 몬테카를로 시뮬레이션
    np.random.seed(42)
    monthly_returns = np.random.normal(annual_return/12, volatility/np.sqrt(12), period_years * 12)
    
    # 포트폴리오 가치 계산
    portfolio_values = [initial_investment]
    
    for i, monthly_return in enumerate(monthly_returns):
        current_value = portfolio_values[-1]
        new_value = current_value * (1 + monthly_return) + monthly_contribution
        portfolio_values.append(new_value)
    
    final_value = portfolio_values[-1]
    total_invested = initial_investment + monthly_contribution * len(monthly_returns)
    total_return = ((final_value - total_invested) / total_invested) * 100
    annual_return_actual = ((final_value / initial_investment) ** (1/period_years) - 1) * 100
    
    # 최대 낙폭 계산
    peak = portfolio_values[0]
    max_drawdown = 0
    
    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    return {
        'final_value': final_value,
        'total_return': total_return,
        'annual_return': annual_return_actual,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': annual_return_actual / (volatility * 100),
        'volatility': volatility * 100
    }

if __name__ == "__main__":
    main()
