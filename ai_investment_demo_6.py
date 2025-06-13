import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from bs4 import BeautifulSoup
import json
import asyncio
import aiohttp
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 1. 다중 데이터 통합 - Naver News API 클래스
class NaverNewsCollector:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36"
        }
    
    def collect_naver_news(self, ticker, count=30):
        """네이버 뉴스에서 주식 관련 뉴스 수집"""
        try:
            # 한국 주식 티커를 검색어로 변환
            search_query = f"{ticker} 주식"
            
            params = {
                "query": search_query,
                "where": "news",
                "start": 1,
                "display": count
            }
            
            response = requests.get(
                "https://search.naver.com/search.naver", 
                params=params, 
                headers=self.headers
            )
            
            soup = BeautifulSoup(response.text, "html.parser")
            news_articles = []
            
            for news_result in soup.select(".list_news .bx")[:count]:
                try:
                    title = news_result.select_one(".news_tit").text.strip()
                    link = news_result.select_one(".news_tit")["href"]
                    snippet = news_result.select_one(".news_dsc").text.strip()
                    press_name = news_result.select_one(".info.press").text.strip()
                    date_elem = news_result.select_one("span.info")
                    news_date = date_elem.text.strip() if date_elem else "날짜 없음"
                    
                    news_articles.append({
                        "title": title,
                        "link": link,
                        "snippet": snippet,
                        "press_name": press_name,
                        "news_date": news_date,
                        "source": "Naver"
                    })
                except Exception as e:
                    continue
            
            return news_articles
        except Exception as e:
            st.error(f"네이버 뉴스 수집 오류: {e}")
            return []

# 2. 데이터 품질 검증 시스템
class DataQualityValidator:
    def __init__(self):
        self.quality_report = {}
    
    def validate_stock_data(self, data):
        """주식 데이터 품질 검증"""
        quality_issues = []
        
        # 완전성 검증 (Completeness)
        missing_data = data.isnull().sum()
        if missing_data.sum() > 0:
            quality_issues.append(f"누락 데이터: {missing_data.to_dict()}")
        
        # 일관성 검증 (Consistency)
        if 'Close' in data.columns:
            # 가격 데이터 일관성 검증
            negative_prices = (data['Close'] < 0).sum()
            if negative_prices > 0:
                quality_issues.append(f"음수 가격 데이터: {negative_prices}개")
            
            # 이상치 검증
            price_std = data['Close'].std()
            price_mean = data['Close'].mean()
            outliers = ((data['Close'] - price_mean).abs() > 3 * price_std).sum()
            if outliers > 0:
                quality_issues.append(f"이상치 데이터: {outliers}개")
        
        # 유효성 검증 (Validity)
        if 'Volume' in data.columns:
            negative_volume = (data['Volume'] < 0).sum()
            if negative_volume > 0:
                quality_issues.append(f"음수 거래량: {negative_volume}개")
        
        self.quality_report = {
            "total_records": len(data),
            "quality_issues": quality_issues,
            "quality_score": max(0, 100 - len(quality_issues) * 10)
        }
        
        return self.quality_report

# 3. 고급 기술적 분석 지표
class AdvancedTechnicalAnalyzer:
    def __init__(self):
        pass
    
    def calculate_fibonacci_levels(self, data, period=20):
        """피보나치 되돌림 레벨 계산"""
        high = data['High'].rolling(window=period).max()
        low = data['Low'].rolling(window=period).min()
        
        diff = high - low
        
        fib_levels = {
            'fib_0': high,
            'fib_236': high - 0.236 * diff,
            'fib_382': high - 0.382 * diff,
            'fib_500': high - 0.500 * diff,
            'fib_618': high - 0.618 * diff,
            'fib_786': high - 0.786 * diff,
            'fib_100': low
        }
        
        return pd.DataFrame(fib_levels, index=data.index)
    
    def detect_elliott_waves(self, data, window=5):
        """엘리엇 파동 패턴 감지 (간단한 버전)"""
        from scipy.signal import argrelextrema
        
        # 극값 찾기
        highs = argrelextrema(data['High'].values, np.greater, order=window)[0]
        lows = argrelextrema(data['Low'].values, np.less, order=window)[0]
        
        # 파동 포인트 생성
        wave_points = []
        for i in highs:
            wave_points.append({
                'index': i,
                'price': data['High'].iloc[i],
                'type': 'high',
                'date': data.index[i]
            })
        
        for i in lows:
            wave_points.append({
                'index': i,
                'price': data['Low'].iloc[i],
                'type': 'low',
                'date': data.index[i]
            })
        
        # 시간순 정렬
        wave_points.sort(key=lambda x: x['index'])
        
        return wave_points[:10]  # 최근 10개 파동 포인트만 반환
    
    def detect_chart_patterns(self, data):
        """차트 패턴 감지"""
        patterns = []
        
        # 간단한 삼각형 패턴 감지
        if len(data) >= 20:
            recent_data = data.tail(20)
            high_trend = np.polyfit(range(len(recent_data)), recent_data['High'], 1)[0]
            low_trend = np.polyfit(range(len(recent_data)), recent_data['Low'], 1)[0]
            
            if abs(high_trend) < 0.1 and abs(low_trend) < 0.1:
                if high_trend < 0 and low_trend > 0:
                    patterns.append("대칭 삼각형")
                elif high_trend < 0 and abs(low_trend) < 0.05:
                    patterns.append("하강 삼각형")
                elif abs(high_trend) < 0.05 and low_trend > 0:
                    patterns.append("상승 삼각형")
        
        # 헤드앤숄더 패턴 감지 (간단한 버전)
        if len(data) >= 30:
            recent_highs = data['High'].tail(30)
            max_idx = recent_highs.idxmax()
            left_shoulder = recent_highs[:max_idx].max()
            right_shoulder = recent_highs[max_idx:].max()
            head = recent_highs[max_idx]
            
            if head > left_shoulder * 1.05 and head > right_shoulder * 1.05:
                patterns.append("헤드앤숄더")
        
        return patterns

# 4. 소셜미디어 및 애널리스트 리포트 분석
class SocialMediaAnalyzer:
    def __init__(self):
        self.social_sources = ['reddit', 'twitter', 'stocktwits']
        self.analyst_sources = ['seeking_alpha', 'morningstar', 'yahoo_finance']
    
    def get_mock_social_sentiment(self, ticker):
        """소셜미디어 감정 분석 (모의 데이터)"""
        # 실제 구현에서는 Reddit API, Twitter API 등을 사용
        import random
        
        social_data = []
        for source in self.social_sources:
            sentiment_score = random.uniform(-1, 1)
            social_data.append({
                'source': source,
                'sentiment_score': sentiment_score,
                'sentiment_label': 'positive' if sentiment_score > 0.1 else 'negative' if sentiment_score < -0.1 else 'neutral',
                'mention_count': random.randint(50, 500),
                'timestamp': datetime.now()
            })
        
        return social_data
    
    def get_mock_analyst_reports(self, ticker):
        """애널리스트 리포트 분석 (모의 데이터)"""
        import random
        
        analyst_data = []
        ratings = ['Strong Buy', 'Buy', 'Hold', 'Sell', 'Strong Sell']
        
        for source in self.analyst_sources:
            rating = random.choice(ratings)
            target_price = random.uniform(50, 200)
            analyst_data.append({
                'source': source,
                'rating': rating,
                'target_price': target_price,
                'current_rating_score': {'Strong Buy': 5, 'Buy': 4, 'Hold': 3, 'Sell': 2, 'Strong Sell': 1}[rating],
                'analyst_name': f"Analyst from {source}",
                'report_date': datetime.now() - timedelta(days=random.randint(1, 30))
            })
        
        return analyst_data

# 5. 동적 시나리오 생성
class DynamicScenarioGenerator:
    def __init__(self):
        self.volatility_regimes = ['low', 'medium', 'high']
        self.market_conditions = ['bull', 'bear', 'sideways']
    
    def analyze_market_regime(self, data):
        """현재 시장 상황 분석"""
        # 변동성 계산
        returns = data['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # 연환산 변동성
        
        # 변동성 체제 분류
        if volatility < 0.15:
            vol_regime = 'low'
        elif volatility < 0.25:
            vol_regime = 'medium'
        else:
            vol_regime = 'high'
        
        # 트렌드 분석
        ma_20 = data['Close'].rolling(20).mean().iloc[-1]
        ma_50 = data['Close'].rolling(50).mean().iloc[-1]
        current_price = data['Close'].iloc[-1]
        
        if current_price > ma_20 > ma_50:
            trend = 'bull'
        elif current_price < ma_20 < ma_50:
            trend = 'bear'
        else:
            trend = 'sideways'
        
        return {
            'volatility_regime': vol_regime,
            'trend': trend,
            'volatility_value': volatility
        }
    
    def generate_dynamic_scenarios(self, data, market_conditions):
        """동적 시나리오 생성"""
        current_price = data['Close'].iloc[-1]
        scenarios = []
        
        if market_conditions['volatility_regime'] == 'high':
            # 고변동성 시나리오
            scenarios = [
                {
                    'name': '극단적 상승',
                    'probability': 0.15,
                    'price_change': 0.25,
                    'description': '고변동성 시장에서 급격한 상승'
                },
                {
                    'name': '극단적 하락',
                    'probability': 0.15,
                    'price_change': -0.25,
                    'description': '고변동성 시장에서 급격한 하락'
                },
                {
                    'name': '횡보',
                    'probability': 0.70,
                    'price_change': 0.05,
                    'description': '변동성 높은 횡보'
                }
            ]
        elif market_conditions['trend'] == 'bull':
            # 강세장 시나리오
            scenarios = [
                {
                    'name': '강세 지속',
                    'probability': 0.60,
                    'price_change': 0.15,
                    'description': '강세장 모멘텀 지속'
                },
                {
                    'name': '조정',
                    'probability': 0.30,
                    'price_change': -0.08,
                    'description': '건전한 조정'
                },
                {
                    'name': '급락',
                    'probability': 0.10,
                    'price_change': -0.20,
                    'description': '예상치 못한 급락'
                }
            ]
        else:
            # 기본 시나리오
            scenarios = [
                {
                    'name': '상승',
                    'probability': 0.35,
                    'price_change': 0.10,
                    'description': '일반적인 상승'
                },
                {
                    'name': '하락',
                    'probability': 0.35,
                    'price_change': -0.10,
                    'description': '일반적인 하락'
                },
                {
                    'name': '횡보',
                    'probability': 0.30,
                    'price_change': 0.02,
                    'description': '횡보 움직임'
                }
            ]
        
        # 시나리오별 예상 가격 계산
        for scenario in scenarios:
            scenario['target_price'] = current_price * (1 + scenario['price_change'])
        
        return scenarios

# 메인 애플리케이션
def main():
    st.set_page_config(
        page_title="고도화된 AI 주식 분석 시스템",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("🚀 고도화된 AI 주식 분석 시스템")
    st.markdown("---")
    
    # 사이드바 설정
    with st.sidebar:
        st.header("📊 분석 설정")
        ticker = st.text_input("주식 티커 입력", value="AAPL", help="예: AAPL, GOOGL, TSLA")
        period = st.selectbox("분석 기간", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
        
        st.markdown("### 🔧 고급 설정")
        enable_naver_news = st.checkbox("네이버 뉴스 분석", value=True)
        enable_social_analysis = st.checkbox("소셜미디어 분석", value=True)
        enable_technical_analysis = st.checkbox("고급 기술적 분석", value=True)
        enable_scenario_analysis = st.checkbox("동적 시나리오 분석", value=True)
    
    if st.button("🔍 종합 분석 시작", type="primary"):
        with st.spinner("데이터 수집 및 분석 중..."):
            try:
                # 1. 기본 주식 데이터 수집
                stock = yf.Ticker(ticker)
                hist_data = stock.history(period=period)
                
                if hist_data.empty:
                    st.error("주식 데이터를 찾을 수 없습니다.")
                    return
                
                # 2. 데이터 품질 검증
                validator = DataQualityValidator()
                quality_report = validator.validate_stock_data(hist_data)
                
                # 3. 고급 기술적 분석
                if enable_technical_analysis:
                    tech_analyzer = AdvancedTechnicalAnalyzer()
                    fib_levels = tech_analyzer.calculate_fibonacci_levels(hist_data)
                    elliott_waves = tech_analyzer.detect_elliott_waves(hist_data)
                    chart_patterns = tech_analyzer.detect_chart_patterns(hist_data)
                
                # 4. 네이버 뉴스 수집
                if enable_naver_news:
                    news_collector = NaverNewsCollector()
                    naver_news = news_collector.collect_naver_news(ticker, 30)
                
                # 5. 소셜미디어 및 애널리스트 분석
                if enable_social_analysis:
                    social_analyzer = SocialMediaAnalyzer()
                    social_sentiment = social_analyzer.get_mock_social_sentiment(ticker)
                    analyst_reports = social_analyzer.get_mock_analyst_reports(ticker)
                
                # 6. 동적 시나리오 생성
                if enable_scenario_analysis:
                    scenario_generator = DynamicScenarioGenerator()
                    market_conditions = scenario_generator.analyze_market_regime(hist_data)
                    dynamic_scenarios = scenario_generator.generate_dynamic_scenarios(hist_data, market_conditions)
                
                # 결과 표시
                display_results(
                    ticker, hist_data, quality_report,
                    fib_levels if enable_technical_analysis else None,
                    elliott_waves if enable_technical_analysis else None,
                    chart_patterns if enable_technical_analysis else None,
                    naver_news if enable_naver_news else None,
                    social_sentiment if enable_social_analysis else None,
                    analyst_reports if enable_social_analysis else None,
                    dynamic_scenarios if enable_scenario_analysis else None,
                    market_conditions if enable_scenario_analysis else None
                )
                
            except Exception as e:
                st.error(f"분석 중 오류가 발생했습니다: {e}")

def display_results(ticker, hist_data, quality_report, fib_levels, elliott_waves, 
                   chart_patterns, naver_news, social_sentiment, analyst_reports, 
                   dynamic_scenarios, market_conditions):
    """분석 결과 표시"""
    
    # 데이터 품질 리포트
    st.header("📋 데이터 품질 리포트")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("총 데이터 수", quality_report['total_records'])
    with col2:
        st.metric("품질 점수", f"{quality_report['quality_score']}/100")
    with col3:
        st.metric("품질 이슈", len(quality_report['quality_issues']))
    
    if quality_report['quality_issues']:
        with st.expander("품질 이슈 상세"):
            for issue in quality_report['quality_issues']:
                st.warning(issue)
    
    # 고급 차트 표시
    st.header("📈 고급 기술적 분석 차트")
    
    # 메인 차트 생성
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('주가 및 기술적 지표', '거래량'),
        row_width=[0.7, 0.3]
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
    
    # 피보나치 레벨 추가
    if fib_levels is not None:
        fib_colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'brown']
        fib_names = ['0%', '23.6%', '38.2%', '50%', '61.8%', '78.6%', '100%']
        
        for i, (level, color, name) in enumerate(zip(fib_levels.columns, fib_colors, fib_names)):
            fig.add_trace(
                go.Scatter(
                    x=fib_levels.index,
                    y=fib_levels[level],
                    mode='lines',
                    name=f'Fib {name}',
                    line=dict(color=color, width=1, dash='dash'),
                    opacity=0.7
                ),
                row=1, col=1
            )
    
    # 엘리엇 파동 포인트 추가
    if elliott_waves:
        wave_x = [point['date'] for point in elliott_waves]
        wave_y = [point['price'] for point in elliott_waves]
        wave_colors = ['red' if point['type'] == 'high' else 'blue' for point in elliott_waves]
        
        fig.add_trace(
            go.Scatter(
                x=wave_x,
                y=wave_y,
                mode='markers+lines',
                name='Elliott Waves',
                marker=dict(color=wave_colors, size=8),
                line=dict(color='gray', width=1)
            ),
            row=1, col=1
        )
    
    # 거래량 차트
    fig.add_trace(
        go.Bar(
            x=hist_data.index,
            y=hist_data['Volume'],
            name="거래량",
            marker_color='lightblue'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f"{ticker} 고급 기술적 분석",
        xaxis_rangeslider_visible=False,
        height=800
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 차트 패턴 표시
    if chart_patterns:
        st.subheader("🔍 감지된 차트 패턴")
        for pattern in chart_patterns:
            st.success(f"✅ {pattern} 패턴이 감지되었습니다.")
    
    # 네이버 뉴스 표시
    if naver_news:
        st.header("📰 네이버 뉴스 분석")
        st.write(f"총 {len(naver_news)}개의 뉴스를 수집했습니다.")
        
        for i, news in enumerate(naver_news[:10]):  # 상위 10개만 표시
            with st.expander(f"📄 {news['title'][:50]}..."):
                st.write(f"**언론사:** {news['press_name']}")
                st.write(f"**날짜:** {news['news_date']}")
                st.write(f"**요약:** {news['snippet']}")
                st.write(f"**링크:** {news['link']}")
    
    # 소셜미디어 감정 분석
    if social_sentiment:
        st.header("💬 소셜미디어 감정 분석")
        
        social_df = pd.DataFrame(social_sentiment)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("감정 점수")
            for data in social_sentiment:
                sentiment_color = "🟢" if data['sentiment_score'] > 0.1 else "🔴" if data['sentiment_score'] < -0.1 else "🟡"
                st.write(f"{sentiment_color} **{data['source'].title()}**: {data['sentiment_score']:.2f} ({data['sentiment_label']})")
        
        with col2:
            st.subheader("언급 횟수")
            fig_social = go.Figure(data=[
                go.Bar(x=[d['source'] for d in social_sentiment], 
                      y=[d['mention_count'] for d in social_sentiment])
            ])
            fig_social.update_layout(title="소셜미디어 언급 횟수")
            st.plotly_chart(fig_social, use_container_width=True)
    
    # 애널리스트 리포트
    if analyst_reports:
        st.header("👨‍💼 애널리스트 리포트")
        
        analyst_df = pd.DataFrame(analyst_reports)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("애널리스트 등급")
            avg_rating = analyst_df['current_rating_score'].mean()
            st.metric("평균 등급 점수", f"{avg_rating:.1f}/5")
            
            for report in analyst_reports:
                rating_color = "🟢" if report['current_rating_score'] >= 4 else "🔴" if report['current_rating_score'] <= 2 else "🟡"
                st.write(f"{rating_color} **{report['source']}**: {report['rating']} (목표가: ${report['target_price']:.2f})")
        
        with col2:
            st.subheader("목표 가격 분포")
            fig_target = go.Figure(data=[
                go.Bar(x=[r['source'] for r in analyst_reports], 
                      y=[r['target_price'] for r in analyst_reports])
            ])
            fig_target.update_layout(title="애널리스트 목표 가격")
            st.plotly_chart(fig_target, use_container_width=True)
    
    # 동적 시나리오 분석
    if dynamic_scenarios and market_conditions:
        st.header("🎯 동적 시나리오 분석")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("현재 시장 상황")
            st.write(f"**변동성 체제:** {market_conditions['volatility_regime'].upper()}")
            st.write(f"**트렌드:** {market_conditions['trend'].upper()}")
            st.write(f"**변동성 수치:** {market_conditions['volatility_value']:.2%}")
        
        with col2:
            st.subheader("시나리오별 확률")
            scenario_df = pd.DataFrame(dynamic_scenarios)
            
            fig_scenario = go.Figure(data=[
                go.Pie(labels=scenario_df['name'], 
                      values=scenario_df['probability'],
                      textinfo='label+percent')
            ])
            fig_scenario.update_layout(title="시나리오 확률 분포")
            st.plotly_chart(fig_scenario, use_container_width=True)
        
        st.subheader("시나리오 상세")
        for scenario in dynamic_scenarios:
            with st.expander(f"📊 {scenario['name']} (확률: {scenario['probability']:.1%})"):
                st.write(f"**예상 가격 변동:** {scenario['price_change']:+.1%}")
                st.write(f"**목표 가격:** ${scenario['target_price']:.2f}")
                st.write(f"**설명:** {scenario['description']}")

if __name__ == "__main__":
    main()
