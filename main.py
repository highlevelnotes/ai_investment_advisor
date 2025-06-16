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
from utils import calculate_portfolio_performance, format_currency

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
    
    # 프로젝트 목적 설명
    st.info("""
    🇰🇷 **AI 기반 국내 ETF 퇴직연금 포트폴리오 관리 시스템**
    
    HyperClova X AI가 매크로 경제 상황을 실시간 분석하여 개인 맞춤형 포트폴리오를 제공합니다.
    순수 국내 ETF만을 활용하여 국내 자본시장 활성화에 기여합니다.
    
    ✅ AI 기반 시장 분석  ✅ 개인 맞춤 포트폴리오  ✅ 실시간 성과 계산  ✅ 국내 ETF 특화
    """)
    
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
    
    # 메인 AI 분석 및 포트폴리오 생성
    st.header("🤖 AI 기반 종합 분석 및 포트폴리오 최적화")
    
    # AI 분석 실행
    if st.button("🚀 AI 종합 분석 및 포트폴리오 생성", type="primary", use_container_width=True):
        ai_analyzer = AIAnalyzer()
        
        with st.spinner("AI가 시장을 분석하고 최적 포트폴리오를 구성하고 있습니다..."):
            # 종합 분석 실행
            comprehensive_result = ai_analyzer.comprehensive_market_analysis(
                st.session_state.get('economic_data', {}),
                st.session_state.get('etf_data', {}),
                user_profile
            )
            
            if comprehensive_result:
                st.session_state.ai_analysis_result = comprehensive_result
                
                # 결과 표시
                analysis = comprehensive_result['analysis']
                portfolio = comprehensive_result['portfolio']
                
                # 1. 시장 분석 결과
                st.subheader("📊 AI 시장 분석")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🏛️ 매크로 경제 분석**")
                    st.write(analysis['macro_analysis'])
                    
                    st.markdown("**📈 ETF 시장 동향**")
                    st.write(analysis['market_trends'])
                
                with col2:
                    st.markdown("**🎯 투자 전략**")
                    st.write(analysis['investment_strategy'])
                    
                    st.markdown("**⚠️ 리스크 요인**")
                    st.write(analysis['risk_factors'])
                
                # 2. 포트폴리오 결과
                st.subheader("💼 AI 추천 포트폴리오")
                
                weights = portfolio['weights']
                
                if weights:
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown("**📊 포트폴리오 구성**")
                        pie_chart = create_portfolio_pie_chart(weights)
                        st.plotly_chart(pie_chart, use_container_width=True)
                    
                    with col2:
                        st.markdown("**📋 상세 구성**")
                        weights_df = pd.DataFrame([
                            {
                                'ETF명': name,
                                '비중': f"{weight*100:.1f}%",
                                '투자금액': format_currency(weight * user_profile['current_assets'])
                            }
                            for name, weight in weights.items()
                        ])
                        st.dataframe(weights_df, use_container_width=True)
                    
                    # 3. 실제 성과 계산
                    st.subheader("📈 예상 성과 (실제 데이터 기반)")
                    
                    with st.spinner("실제 ETF 데이터로 성과를 계산하고 있습니다..."):
                        performance = calculate_portfolio_performance(
                            weights, 
                            st.session_state.etf_data
                        )
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            expected_return = performance['expected_return'] * 100
                            st.metric(
                                "연환산 수익률", 
                                f"{expected_return:.2f}%",
                                help="실제 ETF 과거 수익률 기반"
                            )
                        
                        with col2:
                            volatility = performance['volatility'] * 100
                            st.metric(
                                "연환산 변동성", 
                                f"{volatility:.2f}%",
                                help="실제 ETF 과거 변동성 기반"
                            )
                        
                        with col3:
                            sharpe_ratio = performance['sharpe_ratio']
                            st.metric(
                                "샤프 비율", 
                                f"{sharpe_ratio:.3f}",
                                help="위험 대비 수익률"
                            )
                        
                        with col4:
                            max_drawdown = performance['max_drawdown'] * 100
                            st.metric(
                                "최대 낙폭", 
                                f"{max_drawdown:.2f}%",
                                help="최대 손실 구간"
                            )
                    
                    # 4. AI 분석 근거
                    with st.expander("🧠 AI 분석 근거", expanded=False):
                        st.markdown("**배분 근거:**")
                        st.write(portfolio['allocation_reasoning'])
                        
                        st.markdown("**예상 수익률:**")
                        st.write(portfolio['expected_return'])
                        
                        st.markdown("**리스크 수준:**")
                        st.write(portfolio['risk_level'])
                        
                        st.markdown("**데이터 소스:**")
                        st.write(comprehensive_result['source'])
                        
                        if performance.get('data_points', 0) > 0:
                            st.success(f"✅ 실제 ETF 데이터 {performance['data_points']}일 기반 계산")
                        else:
                            st.warning("⚠️ 샘플 데이터 기반 계산")
                
                else:
                    st.error("포트폴리오 생성에 실패했습니다.")
            
            else:
                st.error("AI 분석에 실패했습니다. 다시 시도해주세요.")
    
    # 이전 분석 결과 표시
    if 'ai_analysis_result' in st.session_state:
        st.markdown("---")
        st.subheader("📋 최근 분석 결과")
        
        result = st.session_state.ai_analysis_result
        portfolio = result['portfolio']
        
        if portfolio['weights']:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("포트폴리오 ETF 수", len(portfolio['weights']))
            
            with col2:
                st.metric("예상 수익률", portfolio['expected_return'])
            
            with col3:
                st.metric("리스크 수준", portfolio['risk_level'])
            
            # 간단한 포트폴리오 요약
            st.markdown("**포트폴리오 요약:**")
            for etf_name, weight in portfolio['weights'].items():
                st.write(f"• {etf_name}: {weight*100:.1f}%")

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

if __name__ == "__main__":
    main()
