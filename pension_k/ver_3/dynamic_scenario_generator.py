# dynamic_scenario_generator.py
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import logging
from datetime import datetime, timedelta
import random

logger = logging.getLogger(__name__)

class DynamicScenarioGenerator:
    def __init__(self):
        self.market_regimes = ['bull', 'bear', 'sideways', 'volatile']
        self.economic_factors = ['interest_rates', 'inflation', 'gdp_growth', 'unemployment']
        
    def generate_dynamic_scenarios(self, 
                                 ticker: str,
                                 current_data: Dict[str, Any],
                                 technical_analysis: Dict[str, Any],
                                 sentiment_data: Dict[str, Any],
                                 social_data: List[Dict[str, Any]],
                                 analyst_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """동적 시나리오 생성"""
        
        try:
            # 시장 환경 분석
            market_regime = self._identify_market_regime(current_data, technical_analysis)
            
            # 다중 요인 분석
            factor_analysis = self._analyze_multiple_factors(
                sentiment_data, social_data, analyst_data
            )
            
            # 시나리오 생성
            scenarios = self._create_dynamic_scenarios(
                ticker, current_data, market_regime, factor_analysis
            )
            
            # 확률 조정
            adjusted_scenarios = self._adjust_scenario_probabilities(
                scenarios, technical_analysis, factor_analysis
            )
            
            # 시간별 시나리오 전개
            time_based_scenarios = self._generate_time_based_scenarios(
                adjusted_scenarios, current_data
            )
            
            return {
                'scenarios': adjusted_scenarios,
                'market_regime': market_regime,
                'factor_analysis': factor_analysis,
                'time_based_scenarios': time_based_scenarios,
                'scenario_triggers': self._identify_scenario_triggers(factor_analysis),
                'method': 'dynamic_scenario_generation'
            }
            
        except Exception as e:
            logger.error(f"동적 시나리오 생성 오류: {e}")
            return {}
    
    def _identify_market_regime(self, current_data: Dict[str, Any], 
                              technical_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """시장 환경 식별"""
        
        # 변동성 분석
        volatility = technical_analysis.get('volatility', 0.15)
        
        # 추세 분석
        trend_strength = technical_analysis.get('trend_strength', 0)
        overall_signal = technical_analysis.get('overall_signal', 'HOLD')
        
        # 시장 환경 결정
        if volatility > 0.3:
            regime = 'volatile'
        elif trend_strength > 0.1 and overall_signal == 'BUY':
            regime = 'bull'
        elif trend_strength < -0.1 and overall_signal == 'SELL':
            regime = 'bear'
        else:
            regime = 'sideways'
        
        return {
            'regime': regime,
            'volatility_level': 'high' if volatility > 0.25 else 'medium' if volatility > 0.15 else 'low',
            'trend_strength': trend_strength,
            'confidence': 0.75
        }
    
    def _analyze_multiple_factors(self, 
                                sentiment_data: Dict[str, Any],
                                social_data: List[Dict[str, Any]],
                                analyst_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """다중 요인 분석"""
        
        # 뉴스 감정 점수
        news_sentiment = sentiment_data.get('sentiment_score', 0)
        
        # 소셜미디어 감정 평균
        social_sentiment = 0
        if social_data:
            social_scores = [item.get('sentiment_score', 0) for item in social_data]
            social_sentiment = np.mean(social_scores)
        
        # 애널리스트 평가 평균
        analyst_sentiment = 0
        if analyst_data:
            rating_scores = []
            for report in analyst_data:
                rating = report.get('rating', 'Hold')
                if rating == 'Buy':
                    rating_scores.append(1)
                elif rating == 'Sell':
                    rating_scores.append(-1)
                else:
                    rating_scores.append(0)
            analyst_sentiment = np.mean(rating_scores) if rating_scores else 0
        
        # 종합 감정 점수
        combined_sentiment = (news_sentiment * 0.4 + 
                            social_sentiment * 0.3 + 
                            analyst_sentiment * 0.3)
        
        # 경제 요인 시뮬레이션
        economic_factors = {
            'interest_rate_trend': random.choice(['rising', 'falling', 'stable']),
            'inflation_pressure': random.choice(['high', 'medium', 'low']),
            'economic_growth': random.choice(['strong', 'moderate', 'weak'])
        }
        
        return {
            'combined_sentiment': combined_sentiment,
            'news_sentiment': news_sentiment,
            'social_sentiment': social_sentiment,
            'analyst_sentiment': analyst_sentiment,
            'economic_factors': economic_factors,
            'sentiment_consistency': self._calculate_sentiment_consistency([
                news_sentiment, social_sentiment, analyst_sentiment
            ])
        }
    
    def _calculate_sentiment_consistency(self, sentiments: List[float]) -> float:
        """감정 일관성 계산"""
        if not sentiments:
            return 0
        
        # 표준편차로 일관성 측정 (낮을수록 일관성 높음)
        std_dev = np.std(sentiments)
        consistency = max(0, 1 - std_dev)
        return consistency
    
    def _create_dynamic_scenarios(self, 
                                ticker: str,
                                current_data: Dict[str, Any],
                                market_regime: Dict[str, Any],
                                factor_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """동적 시나리오 생성"""
        
        current_price = current_data.get('current_price', 100)
        regime = market_regime.get('regime', 'sideways')
        combined_sentiment = factor_analysis.get('combined_sentiment', 0)
        
        # 기본 시나리오 템플릿
        base_scenarios = {
            'bull': {
                'probability': 0.25,
                'return_range': (0.15, 0.35),
                'volatility_factor': 1.2,
                'time_horizon': '6-12 months'
            },
            'base': {
                'probability': 0.50,
                'return_range': (-0.05, 0.15),
                'volatility_factor': 1.0,
                'time_horizon': '3-6 months'
            },
            'bear': {
                'probability': 0.25,
                'return_range': (-0.25, -0.05),
                'volatility_factor': 1.5,
                'time_horizon': '3-9 months'
            }
        }
        
        # 시장 환경에 따른 조정
        regime_adjustments = {
            'bull': {'bull': 0.15, 'base': 0.05, 'bear': -0.10},
            'bear': {'bull': -0.10, 'base': -0.05, 'bear': 0.15},
            'volatile': {'bull': 0.05, 'base': -0.10, 'bear': 0.05},
            'sideways': {'bull': -0.05, 'base': 0.10, 'bear': -0.05}
        }
        
        # 감정 분석에 따른 조정
        sentiment_adjustment = combined_sentiment * 0.1
        
        # 시나리오 조정 적용
        adjusted_scenarios = {}
        for scenario_name, scenario_data in base_scenarios.items():
            regime_adj = regime_adjustments.get(regime, {}).get(scenario_name, 0)
            
            new_probability = scenario_data['probability'] + regime_adj + sentiment_adjustment
            new_probability = max(0.05, min(0.80, new_probability))  # 5%-80% 범위 제한
            
            # 수익률 범위 조정
            return_adj = sentiment_adjustment if scenario_name == 'bull' else -sentiment_adjustment if scenario_name == 'bear' else 0
            adjusted_return_range = (
                scenario_data['return_range'][0] + return_adj,
                scenario_data['return_range'][1] + return_adj
            )
            
            adjusted_scenarios[scenario_name] = {
                'probability': new_probability,
                'return_range': adjusted_return_range,
                'price_target': current_price * (1 + np.mean(adjusted_return_range)),
                'volatility_factor': scenario_data['volatility_factor'],
                'time_horizon': scenario_data['time_horizon'],
                'key_drivers': self._identify_scenario_drivers(scenario_name, factor_analysis),
                'confidence': self._calculate_scenario_confidence(factor_analysis)
            }
        
        # 확률 정규화
        total_prob = sum(s['probability'] for s in adjusted_scenarios.values())
        for scenario in adjusted_scenarios.values():
            scenario['probability'] /= total_prob
        
        return adjusted_scenarios
    
    def _identify_scenario_drivers(self, scenario_name: str, 
                                 factor_analysis: Dict[str, Any]) -> List[str]:
        """시나리오 주요 동인 식별"""
        
        drivers = []
        
        # 감정 분석 기반 동인
        combined_sentiment = factor_analysis.get('combined_sentiment', 0)
        if abs(combined_sentiment) > 0.2:
            if combined_sentiment > 0:
                drivers.append("긍정적 시장 감정")
            else:
                drivers.append("부정적 시장 감정")
        
        # 애널리스트 합의 기반 동인
        analyst_sentiment = factor_analysis.get('analyst_sentiment', 0)
        if abs(analyst_sentiment) > 0.3:
            if analyst_sentiment > 0:
                drivers.append("애널리스트 매수 추천")
            else:
                drivers.append("애널리스트 매도 추천")
        
        # 경제 요인 기반 동인
        economic_factors = factor_analysis.get('economic_factors', {})
        if economic_factors.get('interest_rate_trend') == 'falling':
            drivers.append("금리 하락 기대")
        elif economic_factors.get('interest_rate_trend') == 'rising':
            drivers.append("금리 상승 우려")
        
        # 시나리오별 기본 동인
        scenario_drivers = {
            'bull': ["실적 개선", "시장 확장", "기술 혁신"],
            'base': ["현상 유지", "안정적 성장", "시장 평형"],
            'bear': ["경기 둔화", "리스크 증가", "시장 조정"]
        }
        
        drivers.extend(scenario_drivers.get(scenario_name, []))
        
        return drivers[:4]  # 최대 4개 동인
    
    def _calculate_scenario_confidence(self, factor_analysis: Dict[str, Any]) -> float:
        """시나리오 신뢰도 계산"""
        
        # 감정 일관성이 높을수록 신뢰도 증가
        consistency = factor_analysis.get('sentiment_consistency', 0.5)
        
        # 데이터 소스 다양성
        data_sources = 0
        if factor_analysis.get('news_sentiment') != 0:
            data_sources += 1
        if factor_analysis.get('social_sentiment') != 0:
            data_sources += 1
        if factor_analysis.get('analyst_sentiment') != 0:
            data_sources += 1
        
        source_diversity = data_sources / 3
        
        # 종합 신뢰도
        confidence = (consistency * 0.6) + (source_diversity * 0.4)
        
        return min(0.95, max(0.50, confidence))
    
    def _adjust_scenario_probabilities(self, 
                                     scenarios: Dict[str, Any],
                                     technical_analysis: Dict[str, Any],
                                     factor_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """기술적 분석 기반 확률 조정"""
        
        # 기술적 신호 강도
        overall_signal = technical_analysis.get('overall_signal', 'HOLD')
        trend_strength = technical_analysis.get('trend_strength', 0)
        
        # 신호 기반 조정
        if overall_signal == 'BUY' and trend_strength > 0.1:
            scenarios['bull']['probability'] *= 1.2
            scenarios['bear']['probability'] *= 0.8
        elif overall_signal == 'SELL' and trend_strength < -0.1:
            scenarios['bear']['probability'] *= 1.2
            scenarios['bull']['probability'] *= 0.8
        
        # 확률 재정규화
        total_prob = sum(s['probability'] for s in scenarios.values())
        for scenario in scenarios.values():
            scenario['probability'] /= total_prob
        
        return scenarios
    
    def _generate_time_based_scenarios(self, 
                                     scenarios: Dict[str, Any],
                                     current_data: Dict[str, Any]) -> Dict[str, Any]:
        """시간별 시나리오 전개"""
        
        current_price = current_data.get('current_price', 100)
        
        time_scenarios = {
            '1_month': {},
            '3_months': {},
            '6_months': {},
            '12_months': {}
        }
        
        for time_period in time_scenarios.keys():
            months = int(time_period.split('_')[0])
            time_factor = months / 12  # 연간 기준으로 정규화
            
            for scenario_name, scenario_data in scenarios.items():
                return_range = scenario_data['return_range']
                adjusted_return = (
                    return_range[0] * time_factor,
                    return_range[1] * time_factor
                )
                
                time_scenarios[time_period][scenario_name] = {
                    'probability': scenario_data['probability'],
                    'price_range': (
                        current_price * (1 + adjusted_return[0]),
                        current_price * (1 + adjusted_return[1])
                    ),
                    'expected_price': current_price * (1 + np.mean(adjusted_return))
                }
        
        return time_scenarios
    
    def _identify_scenario_triggers(self, factor_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """시나리오 트리거 식별"""
        
        triggers = []
        
        # 감정 변화 트리거
        combined_sentiment = factor_analysis.get('combined_sentiment', 0)
        if abs(combined_sentiment) > 0.3:
            triggers.append({
                'type': 'sentiment_shift',
                'description': f"시장 감정 {'급격한 개선' if combined_sentiment > 0 else '급격한 악화'}",
                'probability': 0.25,
                'impact': 'high'
            })
        
        # 경제 지표 트리거
        economic_factors = factor_analysis.get('economic_factors', {})
        if economic_factors.get('interest_rate_trend') == 'rising':
            triggers.append({
                'type': 'interest_rate_hike',
                'description': "금리 인상으로 인한 시장 변동성 증가",
                'probability': 0.30,
                'impact': 'medium'
            })
        
        # 실적 발표 트리거
        triggers.append({
            'type': 'earnings_announcement',
            'description': "분기 실적 발표에 따른 주가 변동",
            'probability': 0.80,
            'impact': 'high'
        })
        
        return triggers
