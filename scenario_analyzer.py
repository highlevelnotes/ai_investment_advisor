# scenario_analyzer.py
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime, timedelta
import yfinance as yf
from scipy import stats
from langchain_naver import ChatClovaX
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

class AdvancedScenarioAnalyzer:
    def __init__(self):
        try:
            self.llm = ChatClovaX(
                model="HCX-005",
                temperature=0.4,
                max_tokens=400
            )
            self.llm_available = True
        except:
            self.llm_available = False
            logger.warning("HyperCLOVA X 모델 로드 실패")
    
    def analyze_investment_scenarios(self, ticker: str, 
                                   current_data: Dict[str, Any],
                                   technical_analysis: Dict[str, Any],
                                   sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """실제 데이터 기반 투자 시나리오 분석"""
        
        try:
            # 1. 과거 데이터 기반 변동성 계산
            volatility_data = self._calculate_volatility_metrics(ticker)
            
            # 2. 몬테카르로 시뮬레이션
            monte_carlo_results = self._run_monte_carlo_simulation(
                current_data, volatility_data
            )
            
            # 3. 기술적/감정적 요인 반영
            adjusted_scenarios = self._adjust_scenarios_with_factors(
                monte_carlo_results, technical_analysis, sentiment_data
            )
            
            # 4. AI 기반 시나리오 해석
            ai_interpretation = self._generate_ai_scenario_interpretation(
                ticker, adjusted_scenarios, current_data
            )
            
            # 5. 리스크 시나리오 분석
            risk_scenarios = self._analyze_risk_scenarios(
                ticker, current_data, volatility_data
            )
            
            return {
                'scenarios': adjusted_scenarios,
                'ai_interpretation': ai_interpretation,
                'risk_scenarios': risk_scenarios,
                'volatility_metrics': volatility_data,
                'monte_carlo_summary': self._summarize_monte_carlo(monte_carlo_results),
                'confidence_intervals': self._calculate_confidence_intervals(monte_carlo_results),
                'method': 'advanced_scenario_analysis'
            }
            
        except Exception as e:
            logger.error(f"시나리오 분석 오류: {e}")
            return self._get_fallback_scenarios(ticker, current_data)
    
    def _calculate_volatility_metrics(self, ticker: str) -> Dict[str, float]:
        """과거 데이터 기반 변동성 메트릭 계산"""
        try:
            # 1년간 데이터 수집
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")
            
            if hist.empty:
                return self._get_default_volatility()
            
            # 일일 수익률 계산
            returns = hist['Close'].pct_change().dropna()
            
            # 다양한 변동성 메트릭 계산
            daily_vol = returns.std()
            annual_vol = daily_vol * np.sqrt(252)
            
            # VaR 계산 (95%, 99% 신뢰구간)
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            
            # 최대 낙폭 계산
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # 상승/하락 비대칭성
            upside_vol = returns[returns > 0].std() * np.sqrt(252)
            downside_vol = returns[returns < 0].std() * np.sqrt(252)
            
            return {
                'daily_volatility': daily_vol,
                'annual_volatility': annual_vol,
                'var_95': var_95,
                'var_99': var_99,
                'max_drawdown': max_drawdown,
                'upside_volatility': upside_vol,
                'downside_volatility': downside_vol,
                'skewness': stats.skew(returns),
                'kurtosis': stats.kurtosis(returns)
            }
            
        except Exception as e:
            logger.warning(f"변동성 계산 실패: {e}")
            return self._get_default_volatility()
    
    def _run_monte_carlo_simulation(self, current_data: Dict[str, Any], 
                                  volatility_data: Dict[str, float], 
                                  num_simulations: int = 1000,
                                  time_horizon: int = 252) -> np.ndarray:
        """몬테카르로 시뮬레이션 실행"""
        
        current_price = current_data.get('current_price', 100)
        annual_vol = volatility_data.get('annual_volatility', 0.25)
        
        # 기대 수익률 추정 (CAPM 기반 간소화)
        risk_free_rate = 0.03  # 3% 무위험 수익률
        market_premium = 0.08  # 8% 시장 위험 프리미엄
        beta = 1.0  # 기본 베타값
        expected_return = risk_free_rate + beta * market_premium
        
        # 몬테카르로 시뮬레이션
        dt = 1/252  # 일일 시간 단위
        simulations = []
        
        for _ in range(num_simulations):
            prices = [current_price]
            
            for _ in range(time_horizon):
                # 기하 브라운 운동 모델
                random_shock = np.random.normal(0, 1)
                price_change = expected_return * dt + annual_vol * np.sqrt(dt) * random_shock
                new_price = prices[-1] * np.exp(price_change)
                prices.append(new_price)
            
            simulations.append(prices[-1])  # 최종 가격만 저장
        
        return np.array(simulations)
    
    def _adjust_scenarios_with_factors(self, monte_carlo_results: np.ndarray,
                                     technical_analysis: Dict[str, Any],
                                     sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """기술적/감정적 요인을 반영한 시나리오 조정"""
        
        base_scenarios = self._create_base_scenarios(monte_carlo_results)
        
        # 기술적 분석 조정 팩터
        tech_factor = self._calculate_technical_factor(technical_analysis)
        
        # 감정 분석 조정 팩터
        sentiment_factor = self._calculate_sentiment_factor(sentiment_data)
        
        # 종합 조정 팩터
        combined_factor = (tech_factor * 0.6) + (sentiment_factor * 0.4)
        
        # 시나리오별 조정
        adjusted_scenarios = {}
        
        for scenario_name, scenario_data in base_scenarios.items():
            adjustment = self._get_scenario_adjustment(scenario_name, combined_factor)
            
            adjusted_scenarios[scenario_name] = {
                'probability': scenario_data['probability'],
                'return_range': (
                    scenario_data['return_range'][0] + adjustment,
                    scenario_data['return_range'][1] + adjustment
                ),
                'price_target': scenario_data['price_target'] * (1 + adjustment),
                'key_factors': self._identify_key_factors(scenario_name, tech_factor, sentiment_factor),
                'adjustment_factor': adjustment
            }
        
        return adjusted_scenarios
    
    def _create_base_scenarios(self, monte_carlo_results: np.ndarray) -> Dict[str, Any]:
        """몬테카르로 결과를 기반으로 기본 시나리오 생성"""
        
        # 백분위수 계산
        p10 = np.percentile(monte_carlo_results, 10)
        p25 = np.percentile(monte_carlo_results, 25)
        p50 = np.percentile(monte_carlo_results, 50)
        p75 = np.percentile(monte_carlo_results, 75)
        p90 = np.percentile(monte_carlo_results, 90)
        
        current_price = np.mean(monte_carlo_results) / 1.08  # 역산으로 현재가 추정
        
        return {
            'bull_case': {
                'probability': 0.20,
                'return_range': ((p75/current_price - 1), (p90/current_price - 1)),
                'price_target': p80 := np.percentile(monte_carlo_results, 80),
                'description': '강세 시나리오'
            },
            'base_case': {
                'probability': 0.50,
                'return_range': ((p25/current_price - 1), (p75/current_price - 1)),
                'price_target': p50,
                'description': '기본 시나리오'
            },
            'bear_case': {
                'probability': 0.30,
                'return_range': ((p10/current_price - 1), (p25/current_price - 1)),
                'price_target': p20 := np.percentile(monte_carlo_results, 20),
                'description': '약세 시나리오'
            }
        }
    
    def _calculate_technical_factor(self, technical_analysis: Dict[str, Any]) -> float:
        """기술적 분석 요인 계산"""
        if not technical_analysis:
            return 0.0
        
        factor = 0.0
        
        # 전체 신호
        overall_signal = technical_analysis.get('overall_signal', 'HOLD')
        if overall_signal == 'BUY':
            factor += 0.15
        elif overall_signal == 'SELL':
            factor -= 0.15
        
        # 개별 지표들
        signals = technical_analysis.get('signals', {})
        
        buy_signals = sum(1 for signal in signals.values() if signal == 'BUY')
        sell_signals = sum(1 for signal in signals.values() if signal == 'SELL')
        total_signals = len(signals)
        
        if total_signals > 0:
            signal_ratio = (buy_signals - sell_signals) / total_signals
            factor += signal_ratio * 0.1
        
        # 트렌드 강도
        trend_strength = technical_analysis.get('trend_strength', 0)
        factor += trend_strength * 0.05
        
        return max(-0.25, min(0.25, factor))  # -25% ~ +25% 제한
    
    def _calculate_sentiment_factor(self, sentiment_data: Dict[str, Any]) -> float:
        """감정 분석 요인 계산"""
        if not sentiment_data:
            return 0.0
        
        sentiment_score = sentiment_data.get('sentiment_score', 0)
        confidence = sentiment_data.get('confidence', 0.5)
        
        # 신뢰도에 따른 가중치 적용
        weighted_sentiment = sentiment_score * confidence
        
        # -20% ~ +20% 범위로 조정
        return max(-0.20, min(0.20, weighted_sentiment * 0.2))
    
    def _get_scenario_adjustment(self, scenario_name: str, combined_factor: float) -> float:
        """시나리오별 조정값 계산"""
        adjustments = {
            'bull_case': combined_factor * 1.2,  # 강세 시나리오는 더 민감하게
            'base_case': combined_factor * 0.8,  # 기본 시나리오는 보수적으로
            'bear_case': combined_factor * 1.0   # 약세 시나리오는 기본 적용
        }
        
        return adjustments.get(scenario_name, combined_factor)
    
    def _identify_key_factors(self, scenario_name: str, tech_factor: float, sentiment_factor: float) -> List[str]:
        """시나리오별 핵심 요인 식별"""
        factors = []
        
        if abs(tech_factor) > 0.1:
            if tech_factor > 0:
                factors.append("긍정적 기술적 신호")
            else:
                factors.append("부정적 기술적 신호")
        
        if abs(sentiment_factor) > 0.1:
            if sentiment_factor > 0:
                factors.append("긍정적 시장 감정")
            else:
                factors.append("부정적 시장 감정")
        
        # 시나리오별 기본 요인
        scenario_factors = {
            'bull_case': ["실적 개선", "시장 확대", "긍정적 뉴스"],
            'base_case': ["현재 추세 유지", "안정적 성장", "시장 평균 성과"],
            'bear_case': ["시장 조정", "경기 둔화", "부정적 이슈"]
        }
        
        factors.extend(scenario_factors.get(scenario_name, []))
        
        return factors[:3]  # 최대 3개 요인
    
    async def _generate_ai_scenario_interpretation(self, ticker: str, 
                                                 scenarios: Dict[str, Any],
                                                 current_data: Dict[str, Any]) -> str:
        """AI 기반 시나리오 해석 생성"""
        
        if not self.llm_available:
            return self._generate_simple_interpretation(scenarios)
        
        current_price = current_data.get('current_price', 0)
        
        scenario_summary = ""
        for name, data in scenarios.items():
            prob = data['probability']
            target = data['price_target']
            return_pct = (target / current_price - 1) * 100 if current_price > 0 else 0
            
            scenario_summary += f"\n{name}: {prob:.0%} 확률, 목표가 ${target:.2f} ({return_pct:+.1f}%)"
        
        prompt = f"""
{ticker} 주식의 시나리오 분석 결과를 해석해주세요.

현재 상황:
- 현재가: ${current_price:.2f}

시나리오 분석 결과:{scenario_summary}

다음 관점에서 분석해주세요:
1. 각 시나리오의 실현 가능성
2. 주요 위험 요소와 기회 요소
3. 투자자가 주의해야 할 핵심 포인트
4. 시장 상황에 따른 대응 전략

간결하고 실용적인 조언을 제공해주세요.
"""
        
        messages = [
            SystemMessage(content="당신은 전문 투자 분석가입니다. 시나리오 분석을 바탕으로 실용적인 투자 인사이트를 제공해주세요."),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            return response.content
        except Exception as e:
            logger.warning(f"AI 해석 생성 실패: {e}")
            return self._generate_simple_interpretation(scenarios)
    
    def _analyze_risk_scenarios(self, ticker: str, current_data: Dict[str, Any], 
                              volatility_data: Dict[str, float]) -> Dict[str, Any]:
        """리스크 시나리오 분석"""
        
        current_price = current_data.get('current_price', 100)
        annual_vol = volatility_data.get('annual_volatility', 0.25)
        max_drawdown = volatility_data.get('max_drawdown', -0.20)
        
        # 스트레스 테스트 시나리오
        stress_scenarios = {
            'market_crash': {
                'name': '시장 급락 시나리오',
                'probability': 0.05,
                'price_impact': current_price * (1 + max_drawdown * 1.5),
                'description': '2008년, 2020년 수준의 시장 급락',
                'duration': '3-6개월',
                'recovery_time': '12-18개월'
            },
            'sector_rotation': {
                'name': '섹터 로테이션 시나리오',
                'probability': 0.15,
                'price_impact': current_price * (1 - annual_vol * 0.8),
                'description': '섹터별 자금 이동으로 인한 조정',
                'duration': '1-3개월',
                'recovery_time': '6-12개월'
            },
            'company_specific': {
                'name': '기업 특화 리스크',
                'probability': 0.10,
                'price_impact': current_price * (1 - annual_vol * 1.2),
                'description': '실적 부진, 경영진 이슈 등',
                'duration': '즉시-1개월',
                'recovery_time': '3-12개월'
            }
        }
        
        return stress_scenarios
    
    def _calculate_confidence_intervals(self, monte_carlo_results: np.ndarray) -> Dict[str, float]:
        """신뢰구간 계산"""
        return {
            '90%_lower': np.percentile(monte_carlo_results, 5),
            '90%_upper': np.percentile(monte_carlo_results, 95),
            '95%_lower': np.percentile(monte_carlo_results, 2.5),
            '95%_upper': np.percentile(monte_carlo_results, 97.5),
            '99%_lower': np.percentile(monte_carlo_results, 0.5),
            '99%_upper': np.percentile(monte_carlo_results, 99.5)
        }
    
    def _summarize_monte_carlo(self, results: np.ndarray) -> Dict[str, Any]:
        """몬테카르로 결과 요약"""
        return {
            'mean_price': np.mean(results),
            'median_price': np.median(results),
            'std_dev': np.std(results),
            'min_price': np.min(results),
            'max_price': np.max(results),
            'positive_returns_prob': np.mean(results > np.mean(results)) * 100
        }
    
    def _generate_simple_interpretation(self, scenarios: Dict[str, Any]) -> str:
        """간단한 시나리오 해석 (AI 실패 시 대체)"""
        bull_prob = scenarios.get('bull_case', {}).get('probability', 0)
        bear_prob = scenarios.get('bear_case', {}).get('probability', 0)
        
        if bull_prob > bear_prob:
            outlook = "전반적으로 긍정적인 전망을 보이고 있습니다."
        elif bear_prob > bull_prob:
            outlook = "신중한 접근이 필요한 상황입니다."
        else:
            outlook = "중립적인 시장 상황으로 판단됩니다."
        
        return f"""
시나리오 분석 결과, {outlook}

주요 포인트:
- 강세 시나리오 확률: {bull_prob:.0%}
- 약세 시나리오 확률: {bear_prob:.0%}
- 변동성을 고려한 리스크 관리가 중요합니다.
- 시장 상황 변화에 따른 유연한 대응이 필요합니다.
"""
    
    def _get_default_volatility(self) -> Dict[str, float]:
        """기본 변동성 데이터"""
        return {
            'daily_volatility': 0.02,
            'annual_volatility': 0.25,
            'var_95': -0.03,
            'var_99': -0.05,
            'max_drawdown': -0.20,
            'upside_volatility': 0.22,
            'downside_volatility': 0.28,
            'skewness': -0.5,
            'kurtosis': 3.0
        }
    
    def _get_fallback_scenarios(self, ticker: str, current_data: Dict[str, Any]) -> Dict[str, Any]:
        """대체 시나리오 (오류 시)"""
        current_price = current_data.get('current_price', 100)
        
        return {
            'scenarios': {
                'bull_case': {
                    'probability': 0.25,
                    'return_range': (0.10, 0.25),
                    'price_target': current_price * 1.20,
                    'key_factors': ['시장 회복', '실적 개선'],
                    'adjustment_factor': 0.0
                },
                'base_case': {
                    'probability': 0.50,
                    'return_range': (-0.05, 0.10),
                    'price_target': current_price * 1.05,
                    'key_factors': ['현상 유지', '안정적 성장'],
                    'adjustment_factor': 0.0
                },
                'bear_case': {
                    'probability': 0.25,
                    'return_range': (-0.20, -0.05),
                    'price_target': current_price * 0.85,
                    'key_factors': ['시장 조정', '리스크 증가'],
                    'adjustment_factor': 0.0
                }
            },
            'ai_interpretation': '기본 시나리오 분석이 적용되었습니다.',
            'method': 'fallback_scenarios'
        }
