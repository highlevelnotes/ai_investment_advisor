import pandas as pd
import numpy as np
from typing import Dict, List
from scipy.signal import argrelextrema

class TechnicalAnalysisEngine:
    def __init__(self):
        pass
    
    def comprehensive_analysis(self, hist_data: pd.DataFrame) -> Dict:
        """종합 기술적 분석"""
        
        results = {}
        
        # 1. 피보나치 되돌림 분석
        results['fibonacci'] = self._calculate_fibonacci_levels(hist_data)
        
        # 2. 엘리엇 파동 분석
        results['elliott_wave'] = self._analyze_elliott_waves(hist_data)
        
        # 3. 차트 패턴 인식
        results['chart_patterns'] = self._detect_chart_patterns(hist_data)
        
        # 4. 기본 기술적 지표
        results['rsi'] = self._calculate_rsi(hist_data)
        results['macd'] = self._calculate_macd(hist_data)
        results['bollinger'] = self._calculate_bollinger_bands(hist_data)
        
        # 5. 종합 기술적 신호
        results['overall_signal'] = self._generate_overall_signal(results)
        
        return results
    
    def _calculate_fibonacci_levels(self, data: pd.DataFrame, period: int = 50) -> Dict:
        """피보나치 되돌림 레벨 계산"""
        
        recent_data = data.tail(period)
        high_price = recent_data['High'].max()
        low_price = recent_data['Low'].min()
        
        diff = high_price - low_price
        
        fib_levels = {
            'high': high_price,
            'low': low_price,
            'fib_0%': high_price,
            'fib_23.6%': high_price - 0.236 * diff,
            'fib_38.2%': high_price - 0.382 * diff,
            'fib_50%': high_price - 0.500 * diff,
            'fib_61.8%': high_price - 0.618 * diff,
            'fib_78.6%': high_price - 0.786 * diff,
            'fib_100%': low_price
        }
        
        # 현재 가격이 어느 레벨에 있는지 확인
        current_price = data['Close'].iloc[-1]
        fib_levels['current_level'] = self._find_current_fib_level(current_price, fib_levels)
        
        return fib_levels
    
    def _analyze_elliott_waves(self, data: pd.DataFrame, window: int = 5) -> Dict:
        """엘리엇 파동 분석"""
        
        # 극값 찾기
        highs_idx = argrelextrema(data['High'].values, np.greater, order=window)[0]
        lows_idx = argrelextrema(data['Low'].values, np.less, order=window)[0]
        
        wave_points = []
        
        # 최고점들
        for idx in highs_idx[-8:]:
            wave_points.append({
                'date': data.index[idx],
                'price': data['High'].iloc[idx],
                'type': 'peak',
                'index': idx
            })
        
        # 최저점들
        for idx in lows_idx[-8:]:
            wave_points.append({
                'date': data.index[idx],
                'price': data['Low'].iloc[idx],
                'type': 'trough',
                'index': idx
            })
        
        # 시간순 정렬
        wave_points.sort(key=lambda x: x['index'])
        
        # 현재 파동 분석
        current_wave = self._determine_current_wave(wave_points)
        
        return {
            'wave_points': wave_points[-6:],
            'current_wave': current_wave,
            'wave_count': len(wave_points),
            'trend_strength': self._calculate_wave_strength(wave_points)
        }
    
    def _detect_chart_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """차트 패턴 감지"""
        
        patterns = []
        
        # 삼각형 패턴
        triangle = self._detect_triangle_pattern(data)
        if triangle:
            patterns.append(triangle)
        
        # 헤드앤숄더 패턴
        head_shoulder = self._detect_head_shoulder_pattern(data)
        if head_shoulder:
            patterns.append(head_shoulder)
        
        # 더블탑/바텀 패턴
        double_pattern = self._detect_double_pattern(data)
        if double_pattern:
            patterns.append(double_pattern)
        
        return patterns
    
    def _detect_triangle_pattern(self, data: pd.DataFrame, period: int = 20) -> Dict:
        """삼각형 패턴 감지"""
        
        if len(data) < period:
            return None
        
        recent_data = data.tail(period)
        
        # 고점과 저점의 추세선 계산
        highs = recent_data['High'].values
        lows = recent_data['Low'].values
        x = np.arange(len(recent_data))
        
        high_trend = np.polyfit(x, highs, 1)[0]
        low_trend = np.polyfit(x, lows, 1)[0]
        
        # 삼각형 패턴 판단
        if abs(high_trend) < 0.5 and abs(low_trend) < 0.5:
            if high_trend < 0 and low_trend > 0:
                return {
                    'name': '대칭 삼각형',
                    'type': 'triangle',
                    'confidence': 0.8,
                    'breakout_direction': 'neutral'
                }
            elif high_trend < 0 and abs(low_trend) < 0.1:
                return {
                    'name': '하강 삼각형',
                    'type': 'triangle',
                    'confidence': 0.7,
                    'breakout_direction': 'down'
                }
            elif abs(high_trend) < 0.1 and low_trend > 0:
                return {
                    'name': '상승 삼각형',
                    'type': 'triangle',
                    'confidence': 0.7,
                    'breakout_direction': 'up'
                }
        
        return None
    
    def _detect_head_shoulder_pattern(self, data: pd.DataFrame, period: int = 30) -> Dict:
        """헤드앤숄더 패턴 감지"""
        
        if len(data) < period:
            return None
        
        recent_data = data.tail(period)
        highs = recent_data['High']
        
        peaks_idx = argrelextrema(highs.values, np.greater, order=3)[0]
        
        if len(peaks_idx) >= 3:
            last_peaks = peaks_idx[-3:]
            peak_values = [highs.iloc[i] for i in last_peaks]
            
            left_shoulder, head, right_shoulder = peak_values
            
            if (head > left_shoulder * 1.02 and 
                head > right_shoulder * 1.02 and
                abs(left_shoulder - right_shoulder) / left_shoulder < 0.05):
                
                return {
                    'name': '헤드앤숄더',
                    'type': 'reversal',
                    'confidence': 0.75,
                    'breakout_direction': 'down'
                }
        
        return None
    
    def _detect_double_pattern(self, data: pd.DataFrame, period: int = 25) -> Dict:
        """더블탑/더블바텀 패턴 감지"""
        
        if len(data) < period:
            return None
        
        recent_data = data.tail(period)
        
        # 더블탑 감지
        highs = recent_data['High']
        peaks_idx = argrelextrema(highs.values, np.greater, order=3)[0]
        
        if len(peaks_idx) >= 2:
            last_two_peaks = peaks_idx[-2:]
            peak_values = [highs.iloc[i] for i in last_two_peaks]
            
            if abs(peak_values[0] - peak_values[1]) / peak_values[0] < 0.03:
                return {
                    'name': '더블탑',
                    'type': 'reversal',
                    'confidence': 0.7,
                    'breakout_direction': 'down'
                }
        
        # 더블바텀 감지
        lows = recent_data['Low']
        troughs_idx = argrelextrema(lows.values, np.less, order=3)[0]
        
        if len(troughs_idx) >= 2:
            last_two_troughs = troughs_idx[-2:]
            trough_values = [lows.iloc[i] for i in last_two_troughs]
            
            if abs(trough_values[0] - trough_values[1]) / trough_values[0] < 0.03:
                return {
                    'name': '더블바텀',
                    'type': 'reversal',
                    'confidence': 0.7,
                    'breakout_direction': 'up'
                }
        
        return None
    
    def _calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> Dict:
        """RSI 계산"""
        
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        current_rsi = rsi.iloc[-1]
        
        return {
            'value': current_rsi,
            'signal': 'overbought' if current_rsi > 70 else 'oversold' if current_rsi < 30 else 'neutral',
            'series': rsi.tail(50).tolist()
        }
    
    def _calculate_macd(self, data: pd.DataFrame) -> Dict:
        """MACD 계산"""
        
        exp1 = data['Close'].ewm(span=12).mean()
        exp2 = data['Close'].ewm(span=26).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9).mean()
        histogram = macd - signal
        
        return {
            'macd': macd.iloc[-1],
            'signal': signal.iloc[-1],
            'histogram': histogram.iloc[-1],
            'trend': 'bullish' if macd.iloc[-1] > signal.iloc[-1] else 'bearish'
        }
    
    def _calculate_bollinger_bands(self, data: pd.DataFrame, period: int = 20) -> Dict:
        """볼린저 밴드 계산"""
        
        sma = data['Close'].rolling(window=period).mean()
        std = data['Close'].rolling(window=period).std()
        
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        
        current_price = data['Close'].iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        current_sma = sma.iloc[-1]
        
        return {
            'upper_band': current_upper,
            'lower_band': current_lower,
            'middle_band': current_sma,
            'position': 'upper' if current_price > current_sma else 'lower',
            'squeeze': (current_upper - current_lower) / current_sma < 0.1
        }
    
    def _generate_overall_signal(self, results: Dict) -> str:
        """종합 기술적 신호 생성"""
        
        signals = []
        
        # RSI 신호
        rsi_signal = results.get('rsi', {}).get('signal', 'neutral')
        if rsi_signal == 'oversold':
            signals.append('buy')
        elif rsi_signal == 'overbought':
            signals.append('sell')
        
        # MACD 신호
        macd_trend = results.get('macd', {}).get('trend', 'neutral')
        if macd_trend == 'bullish':
            signals.append('buy')
        elif macd_trend == 'bearish':
            signals.append('sell')
        
        # 차트 패턴 신호
        patterns = results.get('chart_patterns', [])
        for pattern in patterns:
            if pattern.get('breakout_direction') == 'up':
                signals.append('buy')
            elif pattern.get('breakout_direction') == 'down':
                signals.append('sell')
        
        # 신호 집계
        buy_signals = signals.count('buy')
        sell_signals = signals.count('sell')
        
        if buy_signals > sell_signals:
            return 'bullish'
        elif sell_signals > buy_signals:
            return 'bearish'
        else:
            return 'neutral'
    
    def _find_current_fib_level(self, current_price: float, fib_levels: Dict) -> str:
        """현재 가격의 피보나치 레벨 찾기"""
        
        levels = [
            ('0%', fib_levels['fib_0%']),
            ('23.6%', fib_levels['fib_23.6%']),
            ('38.2%', fib_levels['fib_38.2%']),
            ('50%', fib_levels['fib_50%']),
            ('61.8%', fib_levels['fib_61.8%']),
            ('78.6%', fib_levels['fib_78.6%']),
            ('100%', fib_levels['fib_100%'])
        ]
        
        for i, (level_name, level_price) in enumerate(levels[:-1]):
            next_level_price = levels[i + 1][1]
            if next_level_price <= current_price <= level_price:
                return f"Between {levels[i + 1][0]} and {level_name}"
        
        return "Outside range"
    
    def _determine_current_wave(self, wave_points: List[Dict]) -> str:
        """현재 엘리엇 파동 판단"""
        
        if len(wave_points) < 3:
            return "Insufficient data"
        
        # 간단한 파동 카운팅 (실제로는 더 복잡함)
        recent_waves = wave_points[-3:]
        
        if len(recent_waves) >= 3:
            if (recent_waves[0]['type'] == 'trough' and 
                recent_waves[1]['type'] == 'peak' and 
                recent_waves[2]['type'] == 'trough'):
                return "Wave 2 (Correction)"
            elif (recent_waves[0]['type'] == 'peak' and 
                  recent_waves[1]['type'] == 'trough' and 
                  recent_waves[2]['type'] == 'peak'):
                return "Wave 3 (Impulse)"
        
        return "Wave analysis in progress"
    
    def _calculate_wave_strength(self, wave_points: List[Dict]) -> float:
        """파동 강도 계산"""
        
        if len(wave_points) < 2:
            return 0.5
        
        # 최근 파동들의 가격 변동 폭 계산
        price_changes = []
        for i in range(1, len(wave_points)):
            change = abs(wave_points[i]['price'] - wave_points[i-1]['price'])
            price_changes.append(change)
        
        if price_changes:
            avg_change = np.mean(price_changes)
            # 정규화 (0~1 범위)
            return min(1.0, avg_change / wave_points[-1]['price'])
        
        return 0.5
