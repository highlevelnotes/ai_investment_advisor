# advanced_technical_analysis.py (완전 수정 버전)
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from scipy.signal import find_peaks, argrelextrema
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

logger = logging.getLogger(__name__)

class AdvancedTechnicalAnalyzer:
    def __init__(self):
        self.fibonacci_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        
    def analyze_advanced_indicators(self, hist_data: pd.DataFrame) -> Dict[str, Any]:
        """고급 기술적 지표 종합 분석"""
        try:
            if hist_data.empty:
                return {}
            
            results = {}
            
            # 디버깅 정보 출력
            st.write(f"📊 데이터 길이: {len(hist_data)}일")
            st.write(f"💰 가격 범위: ${hist_data['Close'].min():.2f} - ${hist_data['Close'].max():.2f}")
            st.write(f"📈 변동성: {hist_data['Close'].std():.2f}")
            
            # 엘리어트 파동 분석 (개선됨)
            results['elliott_wave'] = self._analyze_elliott_waves_robust(hist_data)
            
            # 피보나치 레벨 계산 (개선됨)
            results['fibonacci_levels'] = self._calculate_fibonacci_levels_robust(hist_data)
            
            # 차트 패턴 인식 (개선됨)
            results['chart_patterns'] = self._detect_chart_patterns_enhanced(hist_data)
            
            # 고급 지표들
            results['advanced_indicators'] = self._calculate_advanced_indicators(hist_data)
            
            return results
            
        except Exception as e:
            logger.error(f"고급 기술적 분석 오류: {e}")
            st.error(f"고급 기술적 분석 오류: {e}")
            return {}
    
    def _analyze_elliott_waves_robust(self, data: pd.DataFrame) -> Dict[str, Any]:
        """강건한 엘리어트 파동 분석"""
        try:
            if len(data) < 30:
                return {
                    'current_wave': '데이터 부족 (최소 30일 필요)',
                    'wave_count': 0,
                    'trend_direction': 'Neutral',
                    'completion_percentage': 0,
                    'next_target': 0,
                    'confidence': 0.0,
                    'debug_info': f'데이터 길이: {len(data)}일'
                }
            
            prices = data['Close'].values
            
            # 적응적 파라미터 설정
            data_length = len(prices)
            min_distance = max(2, data_length // 20)  # 데이터 길이에 비례
            volatility = np.std(prices)
            prominence = volatility * 0.15  # 변동성의 15%
            
            # 고점 감지 (여러 스케일)
            peaks_short, _ = find_peaks(
                prices, 
                distance=min_distance,
                prominence=prominence
            )
            
            peaks_long, _ = find_peaks(
                prices, 
                distance=min_distance * 2,
                prominence=prominence * 1.5
            )
            
            # 저점 감지 (여러 스케일)
            troughs_short, _ = find_peaks(
                -prices, 
                distance=min_distance,
                prominence=prominence
            )
            
            troughs_long, _ = find_peaks(
                -prices, 
                distance=min_distance * 2,
                prominence=prominence * 1.5
            )
            
            # 디버깅 정보
            st.write(f"🔍 감지된 고점: 단기 {len(peaks_short)}개, 장기 {len(peaks_long)}개")
            st.write(f"🔍 감지된 저점: 단기 {len(troughs_short)}개, 장기 {len(troughs_long)}개")
            
            # 전환점 통합 (주요 전환점 우선)
            all_turning_points = []
            
            # 장기 고점/저점 먼저 추가 (더 중요한 전환점)
            for peak in peaks_long:
                all_turning_points.append({
                    'index': peak,
                    'price': prices[peak],
                    'type': 'peak',
                    'importance': 'high'
                })
            
            for trough in troughs_long:
                all_turning_points.append({
                    'index': trough,
                    'price': prices[trough],
                    'type': 'trough',
                    'importance': 'high'
                })
            
            # 단기 전환점 추가 (중복 제거)
            for peak in peaks_short:
                # 장기 전환점과 너무 가까우면 제외
                if not any(abs(peak - tp['index']) < min_distance for tp in all_turning_points):
                    all_turning_points.append({
                        'index': peak,
                        'price': prices[peak],
                        'type': 'peak',
                        'importance': 'medium'
                    })
            
            for trough in troughs_short:
                if not any(abs(trough - tp['index']) < min_distance for tp in all_turning_points):
                    all_turning_points.append({
                        'index': trough,
                        'price': prices[trough],
                        'type': 'trough',
                        'importance': 'medium'
                    })
            
            # 시간순 정렬
            all_turning_points.sort(key=lambda x: x['index'])
            
            st.write(f"📊 총 전환점: {len(all_turning_points)}개")
            
            if len(all_turning_points) >= 5:
                # 엘리어트 파동 패턴 분석
                wave_analysis = self._identify_elliott_pattern_robust(all_turning_points, prices, data)
                return wave_analysis
            else:
                return {
                    'current_wave': f'전환점 부족 ({len(all_turning_points)}개)',
                    'wave_count': len(all_turning_points),
                    'trend_direction': 'Insufficient Data',
                    'completion_percentage': 0,
                    'next_target': 0,
                    'confidence': 0.0,
                    'debug_info': f'전환점 {len(all_turning_points)}개 감지'
                }
                
        except Exception as e:
            error_msg = f'분석 오류: {str(e)}'
            st.error(error_msg)
            return {
                'current_wave': error_msg,
                'wave_count': 0,
                'trend_direction': 'Error',
                'completion_percentage': 0,
                'next_target': 0,
                'confidence': 0.0,
                'debug_info': error_msg
            }
    
    def _identify_elliott_pattern_robust(self, turning_points: List[Dict], prices: np.ndarray, data: pd.DataFrame) -> Dict[str, Any]:
        """강건한 엘리어트 파동 패턴 식별"""
        try:
            # 최근 전환점들로 파동 분석 (최대 8개)
            recent_points = turning_points[-8:] if len(turning_points) >= 8 else turning_points
            
            # 파동 방향 및 강도 분석
            start_price = recent_points[0]['price']
            end_price = recent_points[-1]['price']
            price_change = (end_price - start_price) / start_price
            
            # 검색 결과에서 확인한 엘리어트 파동 특성 반영
            # "5파가 3파의 고점을 넘지 못하고 종료되는 경우" (절단/truncation) 고려
            
            # 파동 분류 (실제 시장 특성 반영)
            if price_change > 0.08:  # 8% 이상 상승
                if len(recent_points) >= 5:
                    # 상승 5파 패턴 확인
                    trend_direction = "강한 상승 추세"
                    current_wave = "상승 임펄스 파동 (1-3-5파)"
                    wave_type = "impulse"
                else:
                    trend_direction = "상승 추세"
                    current_wave = "상승 파동 진행 중"
                    wave_type = "developing"
                    
            elif price_change < -0.08:  # 8% 이상 하락
                if len(recent_points) >= 3:
                    # 하락 3파 패턴 (A-B-C) 확인
                    trend_direction = "강한 하락 추세"
                    current_wave = "하락 조정 파동 (A-B-C파)"
                    wave_type = "corrective"
                else:
                    trend_direction = "하락 추세"
                    current_wave = "하락 파동 진행 중"
                    wave_type = "developing"
                    
            elif abs(price_change) <= 0.03:  # 3% 이내 횡보
                trend_direction = "횡보 조정"
                current_wave = "삼각형 조정 파동"
                wave_type = "triangle"
            else:
                trend_direction = "약한 추세"
                current_wave = "전환 파동"
                wave_type = "transitional"
            
            # 파동 완성도 계산 (검색 결과의 실제 사례 반영)
            if wave_type == "impulse":
                expected_waves = 5  # 상승 5파
                completion = min(100, (len(recent_points) / expected_waves) * 100)
            elif wave_type == "corrective":
                expected_waves = 3  # 하락 3파 (A-B-C)
                completion = min(100, (len(recent_points) / expected_waves) * 100)
            else:
                expected_waves = 5
                completion = min(100, (len(recent_points) / expected_waves) * 80)
            
            # 다음 목표가 계산 (피보나치 확장 기반)
            next_target = self._calculate_elliott_target(recent_points, wave_type, prices)
            
            # 파동 신뢰도 계산 (실제 패턴 일치도 기반)
            confidence = self._calculate_elliott_confidence(recent_points, wave_type, price_change)
            
            # 시간 분석
            time_analysis = self._analyze_wave_timing(recent_points)
            
            # 검색 결과에서 언급된 "교대 원칙" 반영
            alternation_info = self._check_alternation_principle(recent_points)
            
            return {
                'current_wave': current_wave,
                'wave_count': len(recent_points),
                'trend_direction': trend_direction,
                'completion_percentage': completion,
                'next_target': next_target,
                'wave_type': wave_type,
                'confidence': confidence,
                'price_change_percent': price_change * 100,
                'time_analysis': time_analysis,
                'alternation_info': alternation_info,
                'debug_info': f'분석 완료 - {len(recent_points)}개 전환점, {wave_type} 패턴, 신뢰도 {confidence:.1%}'
            }
            
        except Exception as e:
            return {
                'current_wave': f'패턴 분석 오류: {str(e)}',
                'wave_count': 0,
                'trend_direction': 'Error',
                'completion_percentage': 0,
                'next_target': 0,
                'confidence': 0.0,
                'debug_info': f'패턴 분석 실패: {str(e)}'
            }
    
    def _calculate_elliott_target(self, turning_points: List[Dict], wave_type: str, prices: np.ndarray) -> float:
        """엘리어트 파동 목표가 계산"""
        try:
            if len(turning_points) < 3:
                return turning_points[-1]['price']
            
            # 최근 3개 전환점으로 목표 계산
            recent_3 = turning_points[-3:]
            
            if wave_type == "impulse":
                # 상승 임펄스: 1파 길이의 1.618배 확장
                if len(recent_3) >= 3:
                    wave_1_length = abs(recent_3[1]['price'] - recent_3[0]['price'])
                    target = recent_3[-1]['price'] + (wave_1_length * 1.618)
                else:
                    target = recent_3[-1]['price'] * 1.1
                    
            elif wave_type == "corrective":
                # 조정 파동: 0.618 되돌림 목표
                high_price = max(tp['price'] for tp in recent_3)
                low_price = min(tp['price'] for tp in recent_3)
                target = high_price - ((high_price - low_price) * 0.618)
                
            else:
                # 기타: 평균 회귀
                target = np.mean([tp['price'] for tp in recent_3])
            
            return target
            
        except:
            return turning_points[-1]['price'] if turning_points else 0
    
    def _calculate_elliott_confidence(self, turning_points: List[Dict], wave_type: str, price_change: float) -> float:
        """엘리어트 파동 신뢰도 계산"""
        try:
            confidence = 0.5  # 기본 신뢰도
            
            # 전환점 개수에 따른 신뢰도
            point_count = len(turning_points)
            if point_count >= 5:
                confidence += 0.2
            elif point_count >= 3:
                confidence += 0.1
            
            # 파동 타입별 신뢰도 조정
            if wave_type == "impulse" and point_count >= 5:
                confidence += 0.2  # 완전한 5파 패턴
            elif wave_type == "corrective" and point_count >= 3:
                confidence += 0.15  # 완전한 3파 패턴
            
            # 가격 변동 일관성
            if abs(price_change) > 0.05:  # 5% 이상 명확한 방향성
                confidence += 0.1
            
            # 검색 결과에서 언급된 "예외가 많다"는 특성 반영
            confidence *= 0.8  # 엘리어트 파동의 주관성 반영
            
            return min(0.9, confidence)  # 최대 90%로 제한
            
        except:
            return 0.6
    
    def _check_alternation_principle(self, turning_points: List[Dict]) -> str:
        """교대 원칙 확인 (검색 결과 반영)"""
        try:
            if len(turning_points) < 4:
                return "교대 원칙 확인 불가"
            
            # 2파와 4파의 교대 확인
            # "2파가 가파르면 4파는 횡보" 원칙
            
            # 간단한 교대 패턴 확인
            price_ranges = []
            for i in range(1, len(turning_points)):
                price_range = abs(turning_points[i]['price'] - turning_points[i-1]['price'])
                price_ranges.append(price_range)
            
            if len(price_ranges) >= 2:
                # 교대 패턴 확인
                first_move = price_ranges[0]
                second_move = price_ranges[1]
                
                if abs(first_move - second_move) / max(first_move, second_move) > 0.3:
                    return "교대 원칙 확인됨"
                else:
                    return "교대 원칙 미확인"
            
            return "교대 원칙 분석 중"
            
        except:
            return "교대 원칙 분석 실패"
    
    def _analyze_wave_timing(self, turning_points: List[Dict]) -> Dict[str, Any]:
        """파동 시간 분석"""
        try:
            if len(turning_points) < 3:
                return {}
            
            # 전환점 간 시간 간격 계산
            time_intervals = []
            for i in range(1, len(turning_points)):
                interval = turning_points[i]['index'] - turning_points[i-1]['index']
                time_intervals.append(interval)
            
            avg_interval = np.mean(time_intervals) if time_intervals else 0
            
            return {
                'average_wave_duration': f"{avg_interval:.0f}일",
                'total_cycle_duration': f"{turning_points[-1]['index'] - turning_points[0]['index']}일",
                'wave_acceleration': "가속" if len(time_intervals) > 1 and time_intervals[-1] < time_intervals[0] else "감속"
            }
            
        except:
            return {}
    
    def _calculate_fibonacci_levels_robust(self, data: pd.DataFrame) -> Dict[str, Any]:
        """강건한 피보나치 되돌림 레벨 계산"""
        try:
            prices = data['Close']
            
            if len(prices) < 10:
                return {'error': '데이터 부족 (최소 10일 필요)'}
            
            # 다중 기간 분석
            results = {}
            periods = []
            
            # 데이터 길이에 따른 적응적 기간 설정
            if len(prices) >= 60:
                periods = [20, 30, 60]
            elif len(prices) >= 30:
                periods = [15, 20, 30]
            else:
                periods = [10, 15]
            
            for period in periods:
                if len(prices) >= period:
                    recent_data = prices.tail(period)
                    high = recent_data.max()
                    low = recent_data.min()
                    diff = high - low
                    current_price = prices.iloc[-1]
                    
                    # 최소 변동성 확인 (현재가의 2% 이상)
                    if diff > current_price * 0.02:
                        fib_levels = {}
                        for level in self.fibonacci_levels:
                            fib_levels[f'fib_{level}'] = high - (diff * level)
                        
                        # 지지/저항 레벨 식별
                        support_levels = [level for level in fib_levels.values() if level < current_price]
                        resistance_levels = [level for level in fib_levels.values() if level > current_price]
                        
                        results[f'{period}d'] = {
                            'fibonacci_levels': fib_levels,
                            'high': high,
                            'low': low,
                            'current_price': current_price,
                            'nearest_support': max(support_levels) if support_levels else low,
                            'nearest_resistance': min(resistance_levels) if resistance_levels else high,
                            'retracement_percentage': (high - current_price) / diff if diff > 0 else 0,
                            'strength': self._calculate_fib_strength(current_price, fib_levels)
                        }
            
            if results:
                # 가장 적절한 기간 선택
                primary_key = '30d' if '30d' in results else '20d' if '20d' in results else list(results.keys())[0]
                primary_result = results[primary_key]
                
                st.success(f"✅ 피보나치 레벨 계산 완료: {len(results)}개 기간 분석")
                
                return {
                    'primary': primary_result,
                    'all_periods': results,
                    'analysis_summary': self._summarize_fibonacci_analysis(results),
                    'status': 'success'
                }
            else:
                return {'error': '변동성 부족 (2% 미만)', 'status': 'insufficient_volatility'}
                
        except Exception as e:
            return {'error': f'계산 오류: {str(e)}', 'status': 'error'}
    
    def _calculate_fib_strength(self, current_price: float, fib_levels: Dict[str, float]) -> str:
        """피보나치 레벨 강도 계산"""
        try:
            # 현재가가 어떤 피보나치 레벨 근처에 있는지 확인
            min_distance = float('inf')
            closest_level = None
            
            for level_name, level_value in fib_levels.items():
                distance = abs(current_price - level_value) / current_price
                if distance < min_distance:
                    min_distance = distance
                    closest_level = level_name.split('_')[1]
            
            if min_distance < 0.01:  # 1% 이내
                return f"강한 {closest_level} 레벨"
            elif min_distance < 0.02:  # 2% 이내
                return f"중간 {closest_level} 레벨"
            else:
                return "약한 피보나치 신호"
                
        except:
            return "분석 불가"
    
    def _summarize_fibonacci_analysis(self, fib_results: Dict) -> str:
        """피보나치 분석 요약"""
        if not fib_results:
            return "피보나치 분석 데이터 없음"
        
        # 주요 지지/저항 레벨 식별
        all_supports = []
        all_resistances = []
        
        for period_data in fib_results.values():
            if 'nearest_support' in period_data:
                all_supports.append(period_data['nearest_support'])
            if 'nearest_resistance' in period_data:
                all_resistances.append(period_data['nearest_resistance'])
        
        if all_supports and all_resistances:
            avg_support = np.mean(all_supports)
            avg_resistance = np.mean(all_resistances)
            return f"주요 지지선: ${avg_support:.2f}, 저항선: ${avg_resistance:.2f}"
        else:
            return "피보나치 레벨 분석 완료"
    
    def _detect_chart_patterns_enhanced(self, data: pd.DataFrame) -> Dict[str, Any]:
        """향상된 차트 패턴 인식"""
        try:
            patterns = {}
            
            # 헤드앤숄더 패턴 (개선됨)
            patterns['head_and_shoulders'] = self._detect_head_and_shoulders_enhanced(data)
            
            # 삼각형 패턴 (개선됨)
            patterns['triangle'] = self._detect_triangle_pattern_enhanced(data)
            
            # 더블 탑/바텀 (개선됨)
            patterns['double_pattern'] = self._detect_double_pattern_enhanced(data)
            
            # 플래그/페넌트 패턴
            patterns['flag_pennant'] = self._detect_flag_pennant(data)
            
            # 웨지 패턴
            patterns['wedge'] = self._detect_wedge_pattern(data)
            
            # 채널 패턴
            patterns['channel'] = self._detect_channel_pattern(data)
            
            # 감지된 패턴 수 표시
            detected_count = sum(1 for p in patterns.values() if p.get('detected', False))
            if detected_count > 0:
                st.success(f"✅ {detected_count}개 차트 패턴 감지")
            
            return patterns
            
        except Exception as e:
            logger.warning(f"차트 패턴 인식 실패: {e}")
            return {}
    
    def _detect_head_and_shoulders_enhanced(self, data: pd.DataFrame) -> Dict[str, Any]:
        """향상된 헤드앤숄더 패턴 감지"""
        try:
            prices = data['Close'].values
            
            if len(prices) < 20:
                return {'detected': False, 'reason': '데이터 부족'}
            
            # 최근 20일 데이터로 패턴 분석
            recent_prices = prices[-20:]
            
            # 고점 감지
            peaks, _ = find_peaks(
                recent_prices, 
                distance=3,
                prominence=np.std(recent_prices) * 0.3
            )
            
            if len(peaks) >= 3:
                # 최근 3개 고점 분석
                recent_peaks = peaks[-3:]
                peak_heights = [recent_prices[peak] for peak in recent_peaks]
                
                # 헤드앤숄더 조건 확인
                left_shoulder = peak_heights[0]
                head = peak_heights[1]
                right_shoulder = peak_heights[2]
                
                # 조건 확인
                head_highest = head > left_shoulder and head > right_shoulder
                shoulder_similar = abs(left_shoulder - right_shoulder) / max(left_shoulder, right_shoulder) < 0.05
                head_prominence = (head - max(left_shoulder, right_shoulder)) / head > 0.03
                
                if head_highest and shoulder_similar and head_prominence:
                    # 목선 계산
                    neckline_level = min(recent_prices[recent_peaks[0]:recent_peaks[1]].min(),
                                       recent_prices[recent_peaks[1]:recent_peaks[2]].min())
                    
                    # 목표가 계산
                    head_to_neckline = head - neckline_level
                    target_price = neckline_level - head_to_neckline
                    
                    confidence = 0.7 + (0.1 if shoulder_similar else 0) + (0.1 if head_prominence > 0.05 else 0)
                    
                    return {
                        'detected': True,
                        'type': 'Head and Shoulders',
                        'confidence': min(confidence, 0.9),
                        'target_price': target_price,
                        'neckline': neckline_level,
                        'pattern_completion': 85
                    }
            
            return {'detected': False, 'reason': '패턴 조건 불충족'}
            
        except Exception as e:
            return {'detected': False, 'reason': f'분석 오류: {str(e)}'}
    
    def _detect_triangle_pattern_enhanced(self, data: pd.DataFrame) -> Dict[str, Any]:
        """향상된 삼각형 패턴 감지"""
        try:
            prices = data['Close'].values[-15:]  # 최근 15일
            
            if len(prices) < 10:
                return {'detected': False, 'reason': '데이터 부족'}
            
            # 고점과 저점 감지
            peaks, _ = find_peaks(prices, distance=2, prominence=np.std(prices)*0.2)
            troughs, _ = find_peaks(-prices, distance=2, prominence=np.std(prices)*0.2)
            
            if len(peaks) >= 2 and len(troughs) >= 2:
                # 추세선 기울기 계산
                peak_slope = np.polyfit(peaks[-2:], [prices[p] for p in peaks[-2:]], 1)[0]
                trough_slope = np.polyfit(troughs[-2:], [prices[t] for t in troughs[-2:]], 1)[0]
                
                # 삼각형 패턴 유형 판별
                if peak_slope < -0.05 and trough_slope > 0.05:
                    pattern_type = 'Symmetrical Triangle'
                    confidence = 0.75
                elif peak_slope < -0.05 and abs(trough_slope) < 0.02:
                    pattern_type = 'Descending Triangle'
                    confidence = 0.70
                elif abs(peak_slope) < 0.02 and trough_slope > 0.05:
                    pattern_type = 'Ascending Triangle'
                    confidence = 0.70
                else:
                    return {'detected': False, 'reason': '삼각형 패턴 미형성'}
                
                return {
                    'detected': True,
                    'type': pattern_type,
                    'confidence': confidence,
                    'pattern_completion': 70
                }
            
            return {'detected': False, 'reason': '충분한 전환점 없음'}
            
        except Exception as e:
            return {'detected': False, 'reason': f'분석 오류: {str(e)}'}
    
    def _detect_double_pattern_enhanced(self, data: pd.DataFrame) -> Dict[str, Any]:
        """향상된 더블 탑/바텀 패턴 감지"""
        try:
            prices = data['Close'].values[-20:]  # 최근 20일
            
            if len(prices) < 15:
                return {'detected': False, 'reason': '데이터 부족'}
            
            # 고점 감지
            peaks, _ = find_peaks(prices, distance=5, prominence=np.std(prices)*0.3)
            
            if len(peaks) >= 2:
                last_two_peaks = peaks[-2:]
                peak_heights = [prices[p] for p in last_two_peaks]
                
                # 두 고점이 비슷한 높이인지 확인
                height_similarity = abs(peak_heights[0] - peak_heights[1]) / max(peak_heights) < 0.03
                time_gap = last_two_peaks[1] - last_two_peaks[0]
                
                if height_similarity and time_gap >= 4:
                    middle_low = np.min(prices[last_two_peaks[0]:last_two_peaks[1]])
                    pattern_height = max(peak_heights) - middle_low
                    target_price = middle_low - pattern_height
                    
                    return {
                        'detected': True,
                        'type': 'Double Top',
                        'confidence': 0.75,
                        'target_price': target_price,
                        'pattern_completion': 80
                    }
            
            # 저점 감지 (더블 바텀)
            troughs, _ = find_peaks(-prices, distance=5, prominence=np.std(prices)*0.3)
            
            if len(troughs) >= 2:
                last_two_troughs = troughs[-2:]
                trough_depths = [prices[t] for t in last_two_troughs]
                
                depth_similarity = abs(trough_depths[0] - trough_depths[1]) / min(trough_depths) < 0.03
                time_gap = last_two_troughs[1] - last_two_troughs[0]
                
                if depth_similarity and time_gap >= 4:
                    middle_high = np.max(prices[last_two_troughs[0]:last_two_troughs[1]])
                    pattern_height = middle_high - min(trough_depths)
                    target_price = middle_high + pattern_height
                    
                    return {
                        'detected': True,
                        'type': 'Double Bottom',
                        'confidence': 0.75,
                        'target_price': target_price,
                        'pattern_completion': 80
                    }
            
            return {'detected': False, 'reason': '더블 패턴 조건 불충족'}
            
        except Exception as e:
            return {'detected': False, 'reason': f'분석 오류: {str(e)}'}
    
    def _detect_wedge_pattern(self, data: pd.DataFrame) -> Dict[str, Any]:
        """웨지 패턴 감지"""
        try:
            prices = data['Close'].values[-15:]
            
            if len(prices) < 10:
                return {'detected': False}
            
            peaks, _ = find_peaks(prices, distance=2)
            troughs, _ = find_peaks(-prices, distance=2)
            
            if len(peaks) >= 2 and len(troughs) >= 2:
                peak_slope = np.polyfit(peaks[-2:], [prices[p] for p in peaks[-2:]], 1)[0]
                trough_slope = np.polyfit(troughs[-2:], [prices[t] for t in troughs[-2:]], 1)[0]
                
                if peak_slope < 0 and trough_slope < 0 and peak_slope < trough_slope:
                    return {
                        'detected': True,
                        'type': 'Falling Wedge',
                        'confidence': 0.65,
                        'pattern_completion': 60
                    }
                elif peak_slope > 0 and trough_slope > 0 and peak_slope > trough_slope:
                    return {
                        'detected': True,
                        'type': 'Rising Wedge',
                        'confidence': 0.65,
                        'pattern_completion': 60
                    }
            
            return {'detected': False}
            
        except:
            return {'detected': False}
    
    def _detect_channel_pattern(self, data: pd.DataFrame) -> Dict[str, Any]:
        """채널 패턴 감지"""
        try:
            prices = data['Close'].values[-20:]
            
            if len(prices) < 15:
                return {'detected': False}
            
            peaks, _ = find_peaks(prices, distance=3)
            troughs, _ = find_peaks(-prices, distance=3)
            
            if len(peaks) >= 2 and len(troughs) >= 2:
                peak_slope = np.polyfit(peaks[-2:], [prices[p] for p in peaks[-2:]], 1)[0]
                trough_slope = np.polyfit(troughs[-2:], [prices[t] for t in troughs[-2:]], 1)[0]
                
                slope_diff = abs(peak_slope - trough_slope)
                
                if slope_diff < 0.05:  # 평행선 조건
                    if abs(peak_slope) < 0.02:
                        channel_type = 'Horizontal Channel'
                    elif peak_slope > 0:
                        channel_type = 'Rising Channel'
                    else:
                        channel_type = 'Falling Channel'
                    
                    return {
                        'detected': True,
                        'type': channel_type,
                        'confidence': 0.70,
                        'pattern_completion': 75
                    }
            
            return {'detected': False}
            
        except:
            return {'detected': False}
    
    def _detect_flag_pennant(self, data: pd.DataFrame) -> Dict[str, Any]:
        """플래그/페넌트 패턴 감지"""
        try:
            prices = data['Close'].values[-8:]
            volumes = data['Volume'].values[-8:] if 'Volume' in data.columns else None
            
            if len(prices) < 6:
                return {'detected': False}
            
            early_volatility = np.std(prices[:4])
            late_volatility = np.std(prices[-4:])
            
            if late_volatility < early_volatility * 0.7:
                volume_decreasing = True
                if volumes is not None:
                    early_volume = np.mean(volumes[:4])
                    late_volume = np.mean(volumes[-4:])
                    volume_decreasing = late_volume < early_volume
                
                if volume_decreasing:
                    return {
                        'detected': True,
                        'type': 'Flag/Pennant',
                        'confidence': 0.60,
                        'pattern_completion': 70
                    }
            
            return {'detected': False}
            
        except:
            return {'detected': False}
    
    def _calculate_advanced_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """고급 기술적 지표 계산"""
        try:
            indicators = {}
            
            # Ichimoku Cloud
            indicators['ichimoku'] = self._calculate_ichimoku(data)
            
            # Parabolic SAR
            indicators['parabolic_sar'] = self._calculate_parabolic_sar(data)
            
            # Average True Range (ATR)
            indicators['atr'] = self._calculate_atr(data)
            
            # Commodity Channel Index (CCI)
            indicators['cci'] = self._calculate_cci(data)
            
            return indicators
            
        except Exception as e:
            logger.warning(f"고급 지표 계산 실패: {e}")
            return {}
    
    def _calculate_ichimoku(self, data: pd.DataFrame) -> Dict[str, float]:
        """Ichimoku Cloud 계산"""
        try:
            tenkan_sen = (data['High'].rolling(9).max() + data['Low'].rolling(9).min()) / 2
            kijun_sen = (data['High'].rolling(26).max() + data['Low'].rolling(26).min()) / 2
            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
            senkou_span_b = ((data['High'].rolling(52).max() + data['Low'].rolling(52).min()) / 2).shift(26)
            
            return {
                'tenkan_sen': tenkan_sen.iloc[-1] if not tenkan_sen.empty else 0,
                'kijun_sen': kijun_sen.iloc[-1] if not kijun_sen.empty else 0,
                'senkou_span_a': senkou_span_a.iloc[-1] if not senkou_span_a.empty else 0,
                'senkou_span_b': senkou_span_b.iloc[-1] if not senkou_span_b.empty else 0
            }
        except:
            return {}
    
    def _calculate_parabolic_sar(self, data: pd.DataFrame) -> float:
        """Parabolic SAR 계산"""
        try:
            high = data['High'].values
            low = data['Low'].values
            close = data['Close'].values
            
            if len(close) < 5:
                return close[-1]
            
            if close[-1] > close[-5]:
                sar = min(low[-5:])
            else:
                sar = max(high[-5:])
            
            return sar
        except:
            return 0
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Average True Range 계산"""
        try:
            high = data['High']
            low = data['Low']
            close = data['Close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(period).mean()
            
            return atr.iloc[-1] if not atr.empty else 0
        except:
            return 0
    
    def _calculate_cci(self, data: pd.DataFrame, period: int = 20) -> float:
        """Commodity Channel Index 계산"""
        try:
            typical_price = (data['High'] + data['Low'] + data['Close']) / 3
            sma = typical_price.rolling(period).mean()
            mean_deviation = typical_price.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
            
            cci = (typical_price - sma) / (0.015 * mean_deviation)
            
            return cci.iloc[-1] if not cci.empty else 0
        except:
            return 0
    
    def create_advanced_chart(self, data: pd.DataFrame, analysis_results: Dict[str, Any]) -> go.Figure:
        """고급 기술적 분석 차트 생성"""
        try:
            fig = make_subplots(
                rows=4, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=(
                    'Price & Chart Patterns', 
                    'Elliott Wave Analysis', 
                    'Fibonacci Levels',
                    'Pattern Detection'
                ),
                row_heights=[0.5, 0.2, 0.2, 0.1]
            )
            
            # 메인 가격 차트 (캔들스틱)
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Price',
                    increasing_line_color='green',
                    decreasing_line_color='red'
                ),
                row=1, col=1
            )
            
            # 피보나치 레벨 추가
            fib_data = analysis_results.get('fibonacci_levels', {}).get('primary', {})
            if fib_data and 'fibonacci_levels' in fib_data:
                for level_name, level_value in fib_data['fibonacci_levels'].items():
                    level_pct = level_name.split('_')[1]
                    fig.add_hline(
                        y=level_value,
                        line_dash="dash",
                        line_color="gold",
                        annotation_text=f"Fib {level_pct}",
                        annotation_position="bottom right",
                        row=1, col=1
                    )
            
            # 차트 패턴 표시
            patterns = analysis_results.get('chart_patterns', {})
            pattern_y_position = data['Close'].iloc[-1]
            
            for pattern_name, pattern_data in patterns.items():
                if pattern_data.get('detected'):
                    pattern_type = pattern_data.get('type', 'Unknown')
                    confidence = pattern_data.get('confidence', 0)
                    
                    fig.add_annotation(
                        x=data.index[-1],
                        y=pattern_y_position,
                        text=f"{pattern_type}<br>({confidence:.0%})",
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor="blue",
                        bgcolor="lightblue",
                        bordercolor="blue",
                        row=1, col=1
                    )
                    pattern_y_position *= 1.02
            
            # 엘리어트 파동 정보
            elliott = analysis_results.get('elliott_wave', {})
            if elliott and elliott.get('wave_count', 0) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=[data.index[-1]],
                        y=[elliott.get('completion_percentage', 0)],
                        mode='markers+text',
                        text=[f"Wave: {elliott.get('current_wave', 'N/A')}<br>완성도: {elliott.get('completion_percentage', 0):.0f}%"],
                        name='Elliott Wave',
                        marker=dict(size=10, color='purple')
                    ),
                    row=2, col=1
                )
                
                next_target = elliott.get('next_target', 0)
                if next_target > 0:
                    fig.add_hline(
                        y=next_target,
                        line_dash="dot",
                        line_color="purple",
                        annotation_text=f"목표: ${next_target:.2f}",
                        row=1, col=1
                    )
            
            fig.update_layout(
                title="Advanced Technical Analysis Dashboard",
                xaxis_rangeslider_visible=False,
                height=900,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"고급 차트 생성 실패: {e}")
            # 기본 차트 반환
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Price'
            ))
            fig.update_layout(title="Basic Price Chart", height=400)
            return fig
