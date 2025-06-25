import numpy as np
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime, timedelta
import asyncio
from langchain_naver import ChatClovaX

class AdvancedScenarioAnalyzer:
    def __init__(self):
        self.llm = ChatClovaX(model="HCX-005", temperature=0.3, max_tokens=2000)
    
    def analyze_market_regime(self, hist_data: pd.DataFrame) -> Dict:
        """현재 시장 체제 분석"""
        
        # 변동성 계산
        returns = hist_data['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # 연환산 변동성
        
        # 변동성 체제 분류
        if volatility < 0.15:
            vol_regime = 'low'
            vol_level = '낮음'
        elif volatility < 0.25:
            vol_regime = 'medium'
            vol_level = '보통'
        else:
            vol_regime = 'high'
            vol_level = '높음'
        
        # 트렌드 분석
        ma_20 = hist_data['Close'].rolling(20).mean().iloc[-1]
        ma_50 = hist_data['Close'].rolling(50).mean().iloc[-1]
        current_price = hist_data['Close'].iloc[-1]
        
        if current_price > ma_20 > ma_50:
            trend = 'bull'
            trend_direction = '상승'
        elif current_price < ma_20 < ma_50:
            trend = 'bear'
            trend_direction = '하락'
        else:
            trend = 'sideways'
            trend_direction = '횡보'
        
        # 모멘텀 분석
        momentum_5d = (current_price / hist_data['Close'].iloc[-6] - 1) * 100
        momentum_20d = (current_price / hist_data['Close'].iloc[-21] - 1) * 100
        
        return {
            'regime': vol_regime,
            'volatility_level': vol_level,
            'volatility_value': volatility,
            'trend': trend,
            'trend_direction': trend_direction,
            'momentum_5d': momentum_5d,
            'momentum_20d': momentum_20d,
            'confidence': min(0.95, 0.7 + abs(momentum_20d) * 0.01)
        }
    
    async def generate_adaptive_scenarios(self, ticker: str, hist_data: pd.DataFrame, 
                                        market_regime: Dict, sentiment_data: Dict = None) -> Dict:
        """시장 상황에 따른 적응형 시나리오 생성"""
        
        current_price = hist_data['Close'].iloc[-1]
        scenarios = []
        
        if market_regime['regime'] == 'high':
            # 고변동성 시나리오
            scenarios = [
                {
                    'name': '극단적 상승',
                    'probability': 0.15,
                    'price_change': 0.25,
                    'description': '고변동성 시장에서 급격한 상승',
                    'risk_level': '높음',
                    'key_drivers': '시장 충격, 호재 발표',
                    'strategy': '단기 수익 실현 고려'
                },
                {
                    'name': '극단적 하락',
                    'probability': 0.15,
                    'price_change': -0.25,
                    'description': '고변동성 시장에서 급격한 하락',
                    'risk_level': '매우 높음',
                    'key_drivers': '시장 패닉, 악재 발표',
                    'strategy': '손절매 기준 엄격 적용'
                },
                {
                    'name': '변동성 횡보',
                    'probability': 0.70,
                    'price_change': 0.05,
                    'description': '높은 변동성 속 횡보',
                    'risk_level': '보통',
                    'key_drivers': '불확실성, 혼재된 신호',
                    'strategy': '레인지 트레이딩 전략'
                }
            ]
        elif market_regime['trend'] == 'bull':
            # 강세장 시나리오
            scenarios = [
                {
                    'name': '강세 지속',
                    'probability': 0.60,
                    'price_change': 0.15,
                    'description': '강세장 모멘텀 지속',
                    'risk_level': '낮음',
                    'key_drivers': '긍정적 실적, 시장 낙관',
                    'strategy': '추가 매수 고려'
                },
                {
                    'name': '건전한 조정',
                    'probability': 0.30,
                    'price_change': -0.08,
                    'description': '과열 구간 조정',
                    'risk_level': '보통',
                    'key_drivers': '차익 실현, 기술적 조정',
                    'strategy': '조정 시 매수 기회'
                },
                {
                    'name': '급락 리스크',
                    'probability': 0.10,
                    'price_change': -0.20,
                    'description': '예상치 못한 급락',
                    'risk_level': '높음',
                    'key_drivers': '외부 충격, 악재',
                    'strategy': '리스크 관리 강화'
                }
            ]
        elif market_regime['trend'] == 'bear':
            # 약세장 시나리오
            scenarios = [
                {
                    'name': '하락 지속',
                    'probability': 0.50,
                    'price_change': -0.12,
                    'description': '약세장 지속',
                    'risk_level': '높음',
                    'key_drivers': '부정적 전망, 매도 압력',
                    'strategy': '현금 비중 확대'
                },
                {
                    'name': '바닥 반등',
                    'probability': 0.35,
                    'price_change': 0.10,
                    'description': '과매도 구간 반등',
                    'risk_level': '보통',
                    'key_drivers': '밸류에이션 매력, 저점 매수',
                    'strategy': '선별적 매수'
                },
                {
                    'name': '추가 하락',
                    'probability': 0.15,
                    'price_change': -0.25,
                    'description': '추가 하락 압력',
                    'risk_level': '매우 높음',
                    'key_drivers': '시스템 리스크, 유동성 위기',
                    'strategy': '방어적 포지션'
                }
            ]
        else:
            # 기본 시나리오 (횡보)
            scenarios = [
                {
                    'name': '상승 돌파',
                    'probability': 0.35,
                    'price_change': 0.10,
                    'description': '횡보 구간 상향 돌파',
                    'risk_level': '보통',
                    'key_drivers': '긍정적 재료, 매수 신호',
                    'strategy': '돌파 확인 후 매수'
                },
                {
                    'name': '하락 이탈',
                    'probability': 0.35,
                    'price_change': -0.10,
                    'description': '횡보 구간 하향 이탈',
                    'risk_level': '보통',
                    'key_drivers': '부정적 재료, 매도 신호',
                    'strategy': '손절매 준비'
                },
                {
                    'name': '횡보 지속',
                    'probability': 0.30,
                    'price_change': 0.02,
                    'description': '박스권 횡보 지속',
                    'risk_level': '낮음',
                    'key_drivers': '균형 상태, 관망세',
                    'strategy': '레인지 트레이딩'
                }
            ]
        
        # 감정 분석 결과로 시나리오 조정
        if sentiment_data:
            scenarios = self._adjust_scenarios_with_sentiment(scenarios, sentiment_data)
        
        # 시나리오별 목표 가격 계산
        for scenario in scenarios:
            scenario['target_price'] = current_price * (1 + scenario['price_change'])
            scenario['expected_return'] = scenario['price_change'] * 100
        
        # AI 해석 생성
        ai_interpretation = await self._generate_ai_scenario_interpretation(ticker, scenarios, market_regime)
        
        return {
            'detailed_scenarios': scenarios,
            'scenario_probabilities': {s['name']: s['probability'] for s in scenarios},
            'market_regime': market_regime,
            'ai_interpretation': ai_interpretation,
            'confidence': market_regime.get('confidence', 0.8)
        }
    
    def _adjust_scenarios_with_sentiment(self, scenarios: List[Dict], sentiment_data: Dict) -> List[Dict]:
        """감정 분석 결과로 시나리오 확률 조정"""
        
        sentiment_score = sentiment_data.get('composite', {}).get('score', 0)
        
        # 긍정적 감정일 때 상승 시나리오 확률 증가
        if sentiment_score > 0.2:
            for scenario in scenarios:
                if scenario['price_change'] > 0:
                    scenario['probability'] *= 1.2
                else:
                    scenario['probability'] *= 0.8
        # 부정적 감정일 때 하락 시나리오 확률 증가
        elif sentiment_score < -0.2:
            for scenario in scenarios:
                if scenario['price_change'] < 0:
                    scenario['probability'] *= 1.2
                else:
                    scenario['probability'] *= 0.8
        
        # 확률 정규화
        total_prob = sum(s['probability'] for s in scenarios)
        for scenario in scenarios:
            scenario['probability'] /= total_prob
        
        return scenarios
    
    async def _generate_ai_scenario_interpretation(self, ticker: str, scenarios: List[Dict], market_regime: Dict) -> str:
        """AI 기반 시나리오 해석"""
        
        prompt = f"""
{ticker} 주식의 시나리오 분석 결과를 전문가 관점에서 해석해주세요.

현재 시장 상황:
- 시장 체제: {market_regime.get('regime', 'N/A')}
- 트렌드: {market_regime.get('trend_direction', 'N/A')}
- 변동성: {market_regime.get('volatility_level', 'N/A')}

주요 시나리오:
"""
        
        for scenario in scenarios:
            prompt += f"""
- {scenario['name']}: 확률 {scenario['probability']:.1%}, 예상 수익률 {scenario['expected_return']:+.1f}%
  위험도: {scenario['risk_level']}, 전략: {scenario['strategy']}
"""
        
        prompt += """
다음 관점에서 분석해주세요:
1. 각 시나리오의 실현 가능성과 근거
2. 주요 위험 요소와 기회 요소  
3. 투자자가 모니터링해야 할 핵심 지표
4. 시나리오별 대응 전략
"""
        
        try:
            response = await self.llm.ainvoke(prompt)
            return response.content
        except Exception as e:
            return f"시나리오 해석 생성 중 오류: {str(e)}"
