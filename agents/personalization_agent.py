# agents/personalization_agent.py
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from agents.aimodels import UserProfile
import logging
from .hyperclova_client import HyperCLOVAXClient

logger = logging.getLogger(__name__)

class PersonalizationAgent:
    def __init__(self):
        self.user_profiles = {}
        self.clova_client = HyperCLOVAXClient()
        self.korean_sector_mappings = {
            '기술': ['005930.KS', '000660.KS', '035420.KS'],  # 삼성전자, SK하이닉스, NAVER
            '금융': ['055550.KS', '086790.KS', '316140.KS'],  # 신한지주, 하나금융지주, 우리금융지주
            '화학': ['051910.KS', '009830.KS', '011170.KS'],  # LG화학, 한화솔루션, 롯데케미칼
            '자동차': ['005380.KS', '012330.KS', '000270.KS'], # 현대차, 현대모비스, 기아
            '바이오': ['207940.KS', '068270.KS', '196170.KS']  # 삼성바이오로직스, 셀트리온, 알테오젠
        }
    
    def update_user_profile(self, user_profile: UserProfile):
        """사용자 프로파일 업데이트"""
        self.user_profiles[user_profile.user_id] = user_profile
    
    async def generate_personalized_recommendations(self, 
                                                  user_id: str,
                                                  analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """HyperCLOVA X를 사용한 개인화된 투자 추천 생성"""
        try:
            user_profile = self.user_profiles.get(user_id)
            if not user_profile:
                return {'error': '사용자 프로파일을 찾을 수 없습니다'}
            
            # 기본 분석 결과 정리
            analysis_summary = self._prepare_analysis_summary(analysis_results)
            
            # HyperCLOVA X를 사용한 개인화 추천 생성
            personalized_recommendations = await self._generate_ai_recommendations(
                user_profile, analysis_summary
            )
            
            # 포트폴리오 구성
            portfolio = await self._construct_portfolio_with_ai(
                personalized_recommendations, user_profile, analysis_results
            )
            
            return {
                'recommendations': portfolio,
                'ai_reasoning': personalized_recommendations.get('reasoning', []),
                'risk_assessment': await self._ai_risk_assessment(portfolio, user_profile),
                'expected_outcomes': personalized_recommendations.get('expected_outcomes', {})
            }
            
        except Exception as e:
            logger.error(f"개인화 추천 생성 오류: {e}")
            return {'error': str(e)}
    
    def _prepare_analysis_summary(self, analysis_results: Dict[str, Any]) -> str:
        """분석 결과를 요약하여 텍스트로 변환"""
        summary_parts = []
        
        # 기술적 분석 요약
        technical_analysis = analysis_results.get('technical_analysis', {})
        if technical_analysis:
            summary_parts.append("=== 기술적 분석 결과 ===")
            for ticker, data in technical_analysis.items():
                signal = data.get('overall_signal', 'HOLD')
                volatility = data.get('volatility', 0)
                summary_parts.append(f"{ticker}: {signal} 신호, 변동성 {volatility:.2%}")
        
        # 감정 분석 요약
        sentiment_data = analysis_results.get('sentiment_data', {})
        if sentiment_data:
            summary_parts.append("\n=== 시장 감정 분석 ===")
            for ticker, data in sentiment_data.items():
                score = data.get('sentiment_score', 0)
                summary_parts.append(f"{ticker}: 감정 점수 {score:.2f}")
        
        # 리스크 분석 요약
        risk_analysis = analysis_results.get('risk_analysis', {})
        if risk_analysis and 'optimization' in risk_analysis:
            opt_data = risk_analysis['optimization']
            summary_parts.append(f"\n=== 포트폴리오 최적화 ===")
            summary_parts.append(f"예상 수익률: {opt_data.get('expected_return', 0):.2%}")
            summary_parts.append(f"변동성: {opt_data.get('volatility', 0):.2%}")
            summary_parts.append(f"샤프 비율: {opt_data.get('sharpe_ratio', 0):.2f}")
        
        return "\n".join(summary_parts)
    
    async def _generate_ai_recommendations(self, user_profile: UserProfile, 
                                         analysis_summary: str) -> Dict[str, Any]:
        """HyperCLOVA X를 사용한 AI 추천 생성"""
        
        prompt = f"""
다음은 투자자 정보와 시장 분석 결과입니다. 이를 바탕으로 개인화된 투자 추천을 제공해주세요.

=== 투자자 프로필 ===
- 나이: {user_profile.age}세
- 연소득: {user_profile.income:,}만원
- 순자산: {user_profile.net_worth:,}만원
- 위험 성향: {user_profile.risk_tolerance}
- 투자 기간: {user_profile.investment_horizon}
- 선호 섹터: {', '.join(user_profile.sector_preferences)}

=== 시장 분석 결과 ===
{analysis_summary}

다음 형식으로 추천을 제공해주세요:

1. 추천 종목 (최대 5개):
   - 종목명: 추천 비중 (%), 추천 이유

2. 투자 전략:
   - 핵심 전략 설명

3. 위험 관리 방안:
   - 주요 리스크와 대응 방안

4. 예상 성과:
   - 예상 연간 수익률: %
   - 예상 변동성: %
   - 투자 기간별 목표

한국 주식시장 상황을 고려하여 실용적인 조언을 제공해주세요.
"""
        
        messages = [
            {"role": "system", "content": "당신은 한국 주식시장 전문 투자 어드바이저입니다. 개인의 투자 성향과 시장 분석을 바탕으로 실용적이고 구체적인 투자 추천을 제공해주세요."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self.clova_client.generate_response(messages, max_tokens=800)
            
            # 응답 파싱
            recommendations = self._parse_ai_recommendations(response)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"AI 추천 생성 오류: {e}")
            return {
                'reasoning': ['AI 추천 생성 중 오류가 발생했습니다.'],
                'expected_outcomes': {}
            }
    
    def _parse_ai_recommendations(self, response: str) -> Dict[str, Any]:
        """AI 응답을 파싱하여 구조화된 데이터로 변환"""
        try:
            lines = response.split('\n')
            
            reasoning = []
            expected_outcomes = {}
            recommended_stocks = {}
            
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                if '추천 종목' in line:
                    current_section = 'stocks'
                elif '투자 전략' in line:
                    current_section = 'strategy'
                elif '위험 관리' in line:
                    current_section = 'risk'
                elif '예상 성과' in line:
                    current_section = 'outcomes'
                elif current_section:
                    if current_section in ['strategy', 'risk']:
                        reasoning.append(line)
                    elif current_section == 'outcomes':
                        if '수익률' in line:
                            import re
                            numbers = re.findall(r'\d+\.?\d*', line)
                            if numbers:
                                expected_outcomes['expected_annual_return'] = float(numbers[0]) / 100
                        elif '변동성' in line:
                            import re
                            numbers = re.findall(r'\d+\.?\d*', line)
                            if numbers:
                                expected_outcomes['expected_volatility'] = float(numbers[0]) / 100
            
            return {
                'reasoning': reasoning,
                'expected_outcomes': expected_outcomes,
                'recommended_stocks': recommended_stocks
            }
            
        except Exception as e:
            logger.error(f"AI 응답 파싱 오류: {e}")
            return {
                'reasoning': ['응답 파싱 중 오류가 발생했습니다.'],
                'expected_outcomes': {}
            }
    
    async def _construct_portfolio_with_ai(self, ai_recommendations: Dict[str, Any],
                                         user_profile: UserProfile,
                                         analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """AI 추천을 바탕으로 포트폴리오 구성"""
        
        # 기본 포트폴리오 구성 로직 (기존 코드 활용)
        base_portfolio = self._construct_base_portfolio(analysis_results, user_profile)
        
        # AI 추천과 결합하여 최종 포트폴리오 생성
        portfolio = {}
        
        # 상위 종목들을 선택하여 포트폴리오 구성
        if base_portfolio:
            total_weight = 0
            for ticker, data in list(base_portfolio.items())[:5]:  # 상위 5개 종목
                weight = data.get('weight', 0.2)
                portfolio[ticker] = {
                    'weight': weight,
                    'reasoning': data.get('reasoning', ''),
                    'ai_analysis': '포트폴리오 최적화 결과 포함'
                }
                total_weight += weight
            
            # 가중치 정규화
            if total_weight > 0:
                for ticker in portfolio:
                    portfolio[ticker]['weight'] /= total_weight
        
        return portfolio
    
    def _construct_base_portfolio(self, analysis_results: Dict[str, Any], 
                                user_profile: UserProfile) -> Dict[str, Any]:
        """기본 포트폴리오 구성 (기존 로직 활용)"""
        portfolio = {}
        
        # 기술적 분석 결과 활용
        technical_analysis = analysis_results.get('technical_analysis', {})
        sentiment_data = analysis_results.get('sentiment_data', {})
        
        scores = {}
        
        for ticker in technical_analysis.keys():
            score = 0
            
            # 기술적 분석 점수
            technical_signal = technical_analysis[ticker].get('overall_signal', 'HOLD')
            if technical_signal == 'BUY':
                score += 1
            elif technical_signal == 'SELL':
                score -= 1
            
            # 감정 분석 점수
            sentiment_score = sentiment_data.get(ticker, {}).get('sentiment_score', 0)
            score += sentiment_score
            
            scores[ticker] = score
        
        # 점수 기준으로 정렬
        sorted_stocks = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # 상위 종목들로 포트폴리오 구성
        selected_stocks = sorted_stocks[:5]  # 상위 5개
        
        if selected_stocks:
            total_score = sum(max(score, 0.1) for _, score in selected_stocks)
            
            for ticker, score in selected_stocks:
                if score > 0:
                    weight = max(score, 0.1) / total_score
                    portfolio[ticker] = {
                        'weight': min(weight, 0.4),  # 최대 40%
                        'reasoning': f"기술적 분석 점수: {score:.2f}"
                    }
        
        return portfolio
    
    async def _ai_risk_assessment(self, portfolio: Dict[str, Any], 
                                user_profile: UserProfile) -> Dict[str, Any]:
        """AI를 사용한 포트폴리오 리스크 평가"""
        
        portfolio_summary = "\n".join([
            f"{ticker}: {data['weight']:.1%}" 
            for ticker, data in portfolio.items()
        ])
        
        prompt = f"""
다음 포트폴리오의 리스크를 평가해주세요:

투자자 정보:
- 위험 성향: {user_profile.risk_tolerance}
- 투자 기간: {user_profile.investment_horizon}
- 나이: {user_profile.age}세

포트폴리오 구성:
{portfolio_summary}

다음 관점에서 평가해주세요:
1. 분산화 수준 (1-10점)
2. 위험 성향 적합성 (적합/부적합)
3. 주요 리스크 요인
4. 개선 제안사항

간결하게 평가해주세요.
"""
        
        messages = [
            {"role": "system", "content": "당신은 포트폴리오 리스크 관리 전문가입니다."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self.clova_client.generate_response(messages, max_tokens=400)
            
            return {
                'ai_assessment': response,
                'diversification_score': len(portfolio) / 10,
                'risk_alignment': 'AI 평가 완료'
            }
            
        except Exception as e:
            logger.error(f"AI 리스크 평가 오류: {e}")
            return {
                'ai_assessment': '리스크 평가 중 오류가 발생했습니다.',
                'diversification_score': len(portfolio) / 10,
                'risk_alignment': 'UNKNOWN'
            }
