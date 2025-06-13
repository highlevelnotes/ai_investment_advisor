# ai_investment_demo.py
import time
import random
import yfinance as yf
import pandas as pd
from datetime import datetime
import asyncio

class AIInvestmentAnalyzer:
    def __init__(self):
        self.tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        self.analysis_results = {}
        
    def print_header(self):
        print("=" * 60)
        print("🤖 AI 투자 어드바이저 - Perplexity Labs 스타일")
        print("=" * 60)
        print()
        
    def print_step(self, step, message):
        print(f"📊 Step {step}: {message}")
        time.sleep(1)
        
    def collect_market_data(self):
        self.print_step(1, "실시간 시장 데이터 수집 중...")
        
        market_data = {}
        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                hist = stock.history(period="5d")
                
                market_data[ticker] = {
                    'current_price': info.get('currentPrice', 0),
                    'change_percent': ((hist['Close'][-1] - hist['Close'][-2]) / hist['Close'][-2] * 100),
                    'volume': info.get('volume', 0),
                    'market_cap': info.get('marketCap', 0)
                }
                
                print(f"   ✓ {ticker}: ${market_data[ticker]['current_price']:.2f} ({market_data[ticker]['change_percent']:+.2f}%)")
                time.sleep(0.5)
                
            except Exception as e:
                print(f"   ⚠️ {ticker}: 데이터 수집 실패")
                market_data[ticker] = {'current_price': 0, 'change_percent': 0}
        
        self.analysis_results['market_data'] = market_data
        print("   ✅ 시장 데이터 수집 완료\n")
        
    def analyze_sentiment(self):
        self.print_step(2, "AI 감정 분석 수행 중...")
        
        sentiment_scores = {}
        
        for ticker in self.tickers:
            # 실제로는 뉴스 API나 AI 모델을 사용하지만, 데모용으로 시뮬레이션
            sentiment_score = random.uniform(-1, 1)
            confidence = random.uniform(0.6, 0.95)
            
            sentiment_scores[ticker] = {
                'score': sentiment_score,
                'confidence': confidence,
                'sentiment': 'Positive' if sentiment_score > 0.1 else 'Negative' if sentiment_score < -0.1 else 'Neutral'
            }
            
            print(f"   🔍 {ticker}: {sentiment_scores[ticker]['sentiment']} (점수: {sentiment_score:.2f}, 신뢰도: {confidence:.1%})")
            time.sleep(0.8)
        
        self.analysis_results['sentiment'] = sentiment_scores
        print("   ✅ 감정 분석 완료\n")
        
    def technical_analysis(self):
        self.print_step(3, "기술적 지표 계산 중...")
        
        technical_signals = {}
        
        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="30d")
                
                # 간단한 기술적 지표 계산
                sma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
                current_price = hist['Close'].iloc[-1]
                rsi = self.calculate_rsi(hist['Close'])
                
                signal = "BUY" if current_price > sma_20 and rsi < 70 else "SELL" if current_price < sma_20 and rsi > 30 else "HOLD"
                
                technical_signals[ticker] = {
                    'signal': signal,
                    'sma_20': sma_20,
                    'rsi': rsi,
                    'price_vs_sma': ((current_price - sma_20) / sma_20 * 100)
                }
                
                print(f"   📈 {ticker}: {signal} (RSI: {rsi:.1f}, SMA대비: {technical_signals[ticker]['price_vs_sma']:+.1f}%)")
                time.sleep(0.7)
                
            except Exception as e:
                technical_signals[ticker] = {'signal': 'HOLD', 'rsi': 50}
                print(f"   ⚠️ {ticker}: 기술적 분석 제한적")
        
        self.analysis_results['technical'] = technical_signals
        print("   ✅ 기술적 분석 완료\n")
        
    def calculate_rsi(self, prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not rsi.empty else 50
        
    def portfolio_optimization(self):
        self.print_step(4, "AI 포트폴리오 최적화 중...")
        
        # 각 종목의 종합 점수 계산
        portfolio_weights = {}
        
        for ticker in self.tickers:
            # 기술적 분석 점수
            tech_score = 1 if self.analysis_results['technical'][ticker]['signal'] == 'BUY' else -1 if self.analysis_results['technical'][ticker]['signal'] == 'SELL' else 0
            
            # 감정 분석 점수
            sentiment_score = self.analysis_results['sentiment'][ticker]['score']
            
            # 시장 성과 점수
            market_score = self.analysis_results['market_data'][ticker]['change_percent'] / 10
            
            # 종합 점수
            total_score = (tech_score * 0.4) + (sentiment_score * 0.4) + (market_score * 0.2)
            
            portfolio_weights[ticker] = max(total_score, 0)  # 음수 제거
            
            print(f"   🎯 {ticker}: 종합점수 {total_score:.2f} (기술: {tech_score}, 감정: {sentiment_score:.2f}, 시장: {market_score:.2f})")
            time.sleep(0.6)
        
        # 가중치 정규화
        total_weight = sum(portfolio_weights.values())
        if total_weight > 0:
            portfolio_weights = {k: v/total_weight for k, v in portfolio_weights.items()}
        else:
            portfolio_weights = {k: 0.2 for k in self.tickers}  # 균등 분배
        
        self.analysis_results['portfolio'] = portfolio_weights
        print("   ✅ 포트폴리오 최적화 완료\n")
        
    def generate_recommendations(self):
        self.print_step(5, "AI 투자 추천 생성 중...")
        
        recommendations = []
        portfolio = self.analysis_results['portfolio']
        
        # 상위 3개 종목 추천
        sorted_portfolio = sorted(portfolio.items(), key=lambda x: x[1], reverse=True)
        
        for i, (ticker, weight) in enumerate(sorted_portfolio[:3]):
            if weight > 0.05:  # 5% 이상만 추천
                tech_signal = self.analysis_results['technical'][ticker]['signal']
                sentiment = self.analysis_results['sentiment'][ticker]['sentiment']
                
                recommendation = {
                    'ticker': ticker,
                    'weight': weight,
                    'reason': f"{tech_signal} 신호, {sentiment} 감정",
                    'rank': i + 1
                }
                recommendations.append(recommendation)
                
                print(f"   🏆 #{i+1} {ticker}: {weight:.1%} 비중 추천")
                print(f"       이유: {recommendation['reason']}")
                time.sleep(0.8)
        
        self.analysis_results['recommendations'] = recommendations
        print("   ✅ 투자 추천 완료\n")
        
    def display_final_results(self):
        print("🎯 최종 AI 투자 분석 결과")
        print("=" * 60)
        
        # 포트폴리오 요약
        print("\n📊 추천 포트폴리오:")
        for rec in self.analysis_results['recommendations']:
            print(f"   {rec['ticker']}: {rec['weight']:.1%} - {rec['reason']}")
        
        # 시장 현황 요약
        print("\n📈 시장 현황:")
        for ticker, data in self.analysis_results['market_data'].items():
            print(f"   {ticker}: ${data['current_price']:.2f} ({data['change_percent']:+.2f}%)")
        
        # 위험 경고
        print("\n⚠️ 투자 유의사항:")
        print("   • 이 분석은 교육 목적으로 제공됩니다")
        print("   • 실제 투자 전 전문가와 상담하세요")
        print("   • 과거 성과가 미래를 보장하지 않습니다")
        
        print(f"\n🕐 분석 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

def main():
    analyzer = AIInvestmentAnalyzer()
    
    analyzer.print_header()
    
    print("🚀 AI 투자 분석을 시작합니다...\n")
    time.sleep(1)
    
    # 단계별 분석 실행
    analyzer.collect_market_data()
    analyzer.analyze_sentiment()
    analyzer.technical_analysis()
    analyzer.portfolio_optimization()
    analyzer.generate_recommendations()
    
    # 최종 결과 표시
    analyzer.display_final_results()

if __name__ == "__main__":
    main()
