# main.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import os
import pickle
import json
from pathlib import Path

warnings.filterwarnings('ignore')

# 환경변수 기반 설정 로드
from config import Config, validate_config, get_api_status, APP_CONFIG
from data_collector import DataCollector
from ai_analyzer import AIAnalyzer
from portfolio_optimizer import PortfolioOptimizer
from visualization import create_portfolio_pie_chart, create_performance_chart
from utils import calculate_portfolio_performance, format_currency

# 캐시 설정
CACHE_DIR = 'cache'
ETF_CACHE_FILE = os.path.join(CACHE_DIR, 'etf_data_cache.pkl')
ECONOMIC_CACHE_FILE = os.path.join(CACHE_DIR, 'economic_data_cache.pkl')
MARKET_CACHE_FILE = os.path.join(CACHE_DIR, 'market_data_cache.pkl')
CACHE_EXPIRY_HOURS = 6  # 6시간마다 캐시 갱신

# 페이지 설정
st.set_page_config(
    page_title=APP_CONFIG['TITLE'],
    page_icon=APP_CONFIG['PAGE_ICON'],
    layout=APP_CONFIG['LAYOUT']
)

def ensure_cache_dir():
    """캐시 디렉토리 생성"""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

def is_cache_valid(cache_file, expiry_hours=CACHE_EXPIRY_HOURS):
    """캐시 유효성 검사"""
    if not os.path.exists(cache_file):
        return False
    
    file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
    current_time = datetime.now()
    
    return (current_time - file_time).total_seconds() < expiry_hours * 3600

def load_cached_data(cache_file, data_type="데이터"):
    """캐시 데이터 로드"""
    if not is_cache_valid(cache_file):
        return None
    
    try:
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        print(f"✅ {data_type} 캐시 로드 성공")
        return data
    except Exception as e:
        print(f"❌ {data_type} 캐시 로드 실패: {e}")
        return None

def save_cached_data(data, cache_file, data_type="데이터"):
    """캐시 데이터 저장"""
    ensure_cache_dir()
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"✅ {data_type} 캐시 저장 성공")
    except Exception as e:
        print(f"❌ {data_type} 캐시 저장 실패: {e}")

def get_cache_info():
    """캐시 정보 조회"""
    cache_info = {}
    
    for cache_file, data_type in [
        (ETF_CACHE_FILE, "ETF 데이터"),
        (ECONOMIC_CACHE_FILE, "경제지표"),
        (MARKET_CACHE_FILE, "시장 데이터")
    ]:
        if os.path.exists(cache_file):
            file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            is_valid = is_cache_valid(cache_file)
            cache_info[data_type] = {
                'last_updated': file_time.strftime('%Y-%m-%d %H:%M:%S'),
                'is_valid': is_valid,
                'status': '유효' if is_valid else '만료됨'
            }
        else:
            cache_info[data_type] = {
                'last_updated': '없음',
                'is_valid': False,
                'status': '캐시 없음'
            }
    
    return cache_info

def load_or_collect_etf_data():
    """ETF 데이터 로드 또는 수집 (캐싱 적용)"""
    # 캐시 시도
    cached_data = load_cached_data(ETF_CACHE_FILE, "ETF 데이터")
    if cached_data is not None:
        return cached_data
    
    # 캐시가 없거나 만료된 경우 새로 수집
    with st.spinner('📊 ETF 데이터를 수집하고 있습니다... (최초 실행시 시간이 소요됩니다)'):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        data_collector = DataCollector()
        
        # 진행상황 표시를 위한 콜백 함수
        def progress_callback(current, total, message):
            progress = current / total
            progress_bar.progress(progress)
            status_text.text(f"{message} ({current}/{total})")
        
        etf_data = data_collector.get_etf_data(progress_callback=progress_callback)
        
        progress_bar.empty()
        status_text.empty()
        
        # 캐시 저장
        save_cached_data(etf_data, ETF_CACHE_FILE, "ETF 데이터")
        
        return etf_data

def load_or_collect_economic_data():
    """경제지표 데이터 로드 또는 수집 (캐싱 적용)"""
    cached_data = load_cached_data(ECONOMIC_CACHE_FILE, "경제지표")
    if cached_data is not None:
        return cached_data
    
    with st.spinner('📈 경제지표 데이터를 수집하고 있습니다...'):
        data_collector = DataCollector()
        economic_data = data_collector.get_economic_indicators()
        save_cached_data(economic_data, ECONOMIC_CACHE_FILE, "경제지표")
        return economic_data

def load_or_collect_market_data():
    """시장 데이터 로드 또는 수집 (캐싱 적용)"""
    cached_data = load_cached_data(MARKET_CACHE_FILE, "시장 데이터")
    if cached_data is not None:
        return cached_data
    
    with st.spinner('📉 시장 데이터를 수집하고 있습니다...'):
        data_collector = DataCollector()
        market_data = data_collector.get_market_data()
        save_cached_data(market_data, MARKET_CACHE_FILE, "시장 데이터")
        return market_data

def clear_cache():
    """캐시 삭제"""
    cache_files = [ETF_CACHE_FILE, ECONOMIC_CACHE_FILE, MARKET_CACHE_FILE]
    cleared_count = 0
    
    for cache_file in cache_files:
        if os.path.exists(cache_file):
            try:
                os.remove(cache_file)
                cleared_count += 1
            except Exception as e:
                print(f"캐시 삭제 실패 {cache_file}: {e}")
    
    return cleared_count

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
    
    HyperClova X AI가 매크로 경제 상황을 실시간 분석하여 개인 맞춤형 다중 ETF 포트폴리오를 제공합니다.
    순수 국내 ETF만을 활용하여 국내 자본시장 활성화에 기여합니다.
    
    ✅ AI 기반 시장 분석  ✅ 다중 ETF 분산투자  ✅ 실시간 성과 계산  ✅ 국내 ETF 특화
    """)
    
    # 시스템 상태 및 캐시 정보
    with st.expander("🔧 시스템 상태 및 캐시 정보", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**API 연결 상태**")
            status_icon = "✅" if api_status['ecos'] else "❌"
            st.write(f"{status_icon} ECOS API")
            status_icon = "✅" if api_status['hyperclova_x'] else "❌"
            st.write(f"{status_icon} HyperClova X API")
            status_icon = "✅" if api_status['pykrx'] else "❌"
            st.write(f"{status_icon} PyKRX")
        
        with col2:
            st.markdown("**캐시 상태**")
            cache_info = get_cache_info()
            for data_type, info in cache_info.items():
                status_color = "🟢" if info['is_valid'] else "🔴"
                st.write(f"{status_color} {data_type}: {info['status']}")
                if info['last_updated'] != '없음':
                    st.caption(f"마지막 업데이트: {info['last_updated']}")
        
        # 캐시 관리 버튼
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 캐시 새로고침"):
                cleared_count = clear_cache()
                st.success(f"{cleared_count}개 캐시 파일이 삭제되었습니다. 페이지를 새로고침하세요.")
        
        with col2:
            if st.button("📊 데이터 강제 업데이트"):
                clear_cache()
                st.rerun()
        
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
    
    # 데이터 로드 (캐싱 적용)
    if 'etf_data' not in st.session_state:
        st.session_state.etf_data = load_or_collect_etf_data()
    
    if 'economic_data' not in st.session_state:
        st.session_state.economic_data = load_or_collect_economic_data()
    
    if 'market_data' not in st.session_state:
        st.session_state.market_data = load_or_collect_market_data()
    
    # 데이터 로드 상태 표시
    col1, col2, col3 = st.columns(3)
    with col1:
        etf_count = sum(len(etfs) for etfs in st.session_state.etf_data.values())
        st.metric("로드된 ETF 수", etf_count)
    with col2:
        economic_count = len(st.session_state.economic_data)
        st.metric("경제지표 수", economic_count)
    with col3:
        market_status = "정상" if st.session_state.market_data else "오류"
        st.metric("시장 데이터", market_status)
    
    # 메인 AI 분석 및 포트폴리오 생성
    st.header("🤖 AI 기반 종합 분석 및 다중 ETF 포트폴리오 최적화")
    
    # AI 분석 실행
    if st.button("🚀 AI 종합 분석 및 다중 ETF 포트폴리오 생성", type="primary", use_container_width=True):
        ai_analyzer = AIAnalyzer()
        
        with st.spinner("AI가 시장을 분석하고 다중 ETF 최적 포트폴리오를 구성하고 있습니다..."):
            # 종합 분석 실행
            comprehensive_result = ai_analyzer.comprehensive_market_analysis(
                st.session_state.economic_data,
                st.session_state.etf_data,
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
                
                # 2. 다중 ETF 포트폴리오 결과
                st.subheader("💼 AI 추천 다중 ETF 포트폴리오")
                
                weights = portfolio['weights']
                
                if weights:
                    # 포트폴리오 개요
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("총 ETF 수", portfolio.get('etf_count', len(weights)))
                    with col2:
                        st.metric("분산투자 전략", "다중 ETF 조합")
                    with col3:
                        category_count = len(portfolio.get('category_distribution', {}).get('category_weights', {}))
                        st.metric("자산군 수", category_count)
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown("**📊 전체 포트폴리오 구성**")
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
                    
                    # 3. 자산군별 분포 표시
                    if 'category_distribution' in portfolio:
                        st.subheader("📈 자산군별 ETF 분포")
                        
                        category_dist = portfolio['category_distribution']
                        
                        for category, category_weight in category_dist.get('category_weights', {}).items():
                            if category_weight > 0:
                                with st.expander(f"{category} ({category_weight*100:.1f}%)", expanded=False):
                                    category_etfs = category_dist.get('etfs_by_category', {}).get(category, [])
                                    
                                    for etf_info in category_etfs:
                                        col1, col2 = st.columns([3, 1])
                                        with col1:
                                            st.write(f"• {etf_info['name']}")
                                        with col2:
                                            st.write(f"{etf_info['weight']*100:.1f}%")
                    
                    # 4. 실제 성과 계산
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
                                help="다중 ETF 분산효과 반영"
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
                                help="분산투자 효과로 낙폭 감소"
                            )
                    
                    # 5. 분산투자 효과 분석
                    with st.expander("🔍 분산투자 효과 분석", expanded=False):
                        st.markdown("**다중 ETF 분산투자 전략:**")
                        st.write(portfolio.get('diversification_strategy', '자산군 내외 이중 분산투자'))
                        
                        st.markdown("**배분 근거:**")
                        st.write(portfolio.get('allocation_reasoning', 'AI 기반 최적 배분'))
                        
                        st.markdown("**분산투자 장점:**")
                        st.write("• 각 자산군 내에서 2-3개 ETF 조합으로 이중 분산효과")
                        st.write("• 개별 ETF 리스크 최소화")
                        st.write("• 시장 변동성에 대한 안정성 확보")
                        st.write("• 상관관계가 낮은 ETF 조합으로 포트폴리오 효율성 증대")
                        
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
        st.subheader("📋 최근 분석 결과 요약")
        
        result = st.session_state.ai_analysis_result
        portfolio = result['portfolio']
        
        if portfolio['weights']:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("포트폴리오 ETF 수", len(portfolio['weights']))
            
            with col2:
                st.metric("분산투자 전략", "다중 ETF")
            
            with col3:
                category_count = len(portfolio.get('category_distribution', {}).get('category_weights', {}))
                st.metric("자산군 수", category_count)
            
            with col4:
                st.metric("데이터 소스", result.get('source', 'unknown'))
            
            # 간단한 포트폴리오 요약
            with st.expander("포트폴리오 상세 구성", expanded=False):
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
