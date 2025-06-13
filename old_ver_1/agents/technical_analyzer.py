# agents/technical_analyzer.py
import pandas as pd
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class TechnicalAnalyzerAgent:
    def __init__(self):
        self.indicators = {}
    
    def analyze_technical_indicators(self, historical_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """기술적 지표 종합 분석"""
        technical_results = {}
        
        for ticker, data in historical_data.items():
            if data.empty:
                continue
                
            try:
                indicators = self._calculate_all_indicators(data)
                signals = self._generate_signals(indicators)
                
                technical_results[ticker] = {
                    'indicators': indicators,
                    'signals': signals,
                    'overall_signal': self._calculate_overall_signal(signals),
                    'volatility': self._calculate_volatility(data),
                    'trend_strength': self._calculate_trend_strength(data)
                }
                
            except Exception as e:
                logger.error(f"{ticker} 기술적 분석 오류: {e}")
                
        return technical_results
    
    def _calculate_all_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """모든 기술적 지표 계산"""
        indicators = {}
        
        # RSI
        indicators['RSI'] = self._calculate_rsi(data)
        
        # MACD
        macd_line, signal_line, histogram = self._calculate_macd(data)
        indicators['MACD'] = {
            'macd_line': macd_line,
            'signal_line': signal_line,
            'histogram': histogram
        }
        
        # Bollinger Bands
        upper, middle, lower = self._calculate_bollinger_bands(data)
        indicators['Bollinger'] = {
            'upper': upper,
            'middle': middle,
            'lower': lower
        }
        
        # Moving Averages
        indicators['SMA_20'] = data['Close'].rolling(window=20).mean()
        indicators['SMA_50'] = data['Close'].rolling(window=50).mean()
        indicators['EMA_12'] = data['Close'].ewm(span=12).mean()
        indicators['EMA_26'] = data['Close'].ewm(span=26).mean()
        
        # Stochastic Oscillator
        indicators['Stochastic'] = self._calculate_stochastic(data)
        
        return indicators
    
    def _calculate_rsi(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """RSI 계산"""
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9):
        """MACD 계산"""
        ema_fast = data['Close'].ewm(span=fast).mean()
        ema_slow = data['Close'].ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def _calculate_bollinger_bands(self, data: pd.DataFrame, window: int = 20, num_std: float = 2):
        """볼린저 밴드 계산"""
        middle = data['Close'].rolling(window=window).mean()
        std = data['Close'].rolling(window=window).std()
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        return upper, middle, lower
    
    def _calculate_stochastic(self, data: pd.DataFrame, k_window: int = 14, d_window: int = 3):
        """스토캐스틱 오실레이터 계산"""
        low_min = data['Low'].rolling(window=k_window).min()
        high_max = data['High'].rolling(window=k_window).max()
        k_percent = 100 * ((data['Close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=d_window).mean()
        return {'%K': k_percent, '%D': d_percent}
    
    def _generate_signals(self, indicators: Dict[str, Any]) -> Dict[str, str]:
        """기술적 지표 기반 신호 생성"""
        signals = {}
        
        # RSI 신호
        current_rsi = indicators['RSI'].iloc[-1] if not indicators['RSI'].empty else 50
        if current_rsi > 70:
            signals['RSI'] = 'SELL'
        elif current_rsi < 30:
            signals['RSI'] = 'BUY'
        else:
            signals['RSI'] = 'HOLD'
        
        # MACD 신호
        macd_current = indicators['MACD']['macd_line'].iloc[-1]
        signal_current = indicators['MACD']['signal_line'].iloc[-1]
        if macd_current > signal_current:
            signals['MACD'] = 'BUY'
        else:
            signals['MACD'] = 'SELL'
        
        # 볼린저 밴드 신호
        current_price = indicators['Bollinger']['middle'].index[-1]  # 임시로 인덱스 사용
        upper_current = indicators['Bollinger']['upper'].iloc[-1]
        lower_current = indicators['Bollinger']['lower'].iloc[-1]
        
        # 실제 구현에서는 현재 가격을 사용해야 함
        signals['Bollinger'] = 'HOLD'  # 기본값
        
        return signals
    
    def _calculate_overall_signal(self, signals: Dict[str, str]) -> str:
        """전체 신호 계산"""
        buy_count = sum(1 for signal in signals.values() if signal == 'BUY')
        sell_count = sum(1 for signal in signals.values() if signal == 'SELL')
        
        if buy_count > sell_count:
            return 'BUY'
        elif sell_count > buy_count:
            return 'SELL'
        else:
            return 'HOLD'
    
    def _calculate_volatility(self, data: pd.DataFrame, window: int = 20) -> float:
        """변동성 계산"""
        returns = data['Close'].pct_change()
        volatility = returns.rolling(window=window).std() * np.sqrt(252)  # 연간화
        return volatility.iloc[-1] if not volatility.empty else 0
    
    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """트렌드 강도 계산"""
        if len(data) < 20:
            return 0
        
        sma_20 = data['Close'].rolling(window=20).mean()
        current_price = data['Close'].iloc[-1]
        sma_current = sma_20.iloc[-1]
        
        trend_strength = (current_price - sma_current) / sma_current
        return trend_strength
