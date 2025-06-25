# workflow.py
from langgraph.graph import StateGraph, END
from typing import Dict, Any
import asyncio
import logging
from datetime import datetime

from agents.aimodels import InvestmentState, UserProfile
from agents.data_collector import DataCollectorAgent
from agents.sentiment_analyzer import SentimentAnalyzerAgent
from agents.technical_analyzer import TechnicalAnalyzerAgent
from agents.risk_manager import RiskManagerAgent
from agents.personalization_agent import PersonalizationAgent
from agents.hyperclova_client import HyperCLOVAXClient

logger = logging.getLogger(__name__)

class InvestmentWorkflow:
    def __init__(self):
        self.data_collector = DataCollectorAgent()
        self.sentiment_analyzer = SentimentAnalyzerAgent()
        self.technical_analyzer = TechnicalAnalyzerAgent()
        self.risk_manager = RiskManagerAgent()
        self.personalization_agent = PersonalizationAgent()
        self.clova_client = HyperCLOVAXClient()
        
        self.workflow = self._setup_workflow()
    
    def _setup_workflow(self) -> StateGraph:
        """워크플로우 설정"""
        builder = StateGraph(InvestmentState)
        
        # 노드 추가 (상태 키와 충돌하지 않는 이름 사용)
        builder.add_node("data_collection_node", self._collect_data_node)
        builder.add_node("sentiment_analysis_node", self._analyze_sentiment_node)
        builder.add_node("technical_analysis_node", self._technical_analysis_node)  # 이름 변경
        builder.add_node("risk_analysis_node", self._risk_analysis_node)
        builder.add_node("recommendation_generation_node", self._generate_recommendations_node)
        builder.add_node("ai_summary_node", self._ai_summary_node)
        
        # 엣지 설정 (노드 이름 업데이트)
        builder.set_entry_point("data_collection_node")
        builder.add_edge("data_collection_node", "sentiment_analysis_node")
        builder.add_edge("sentiment_analysis_node", "technical_analysis_node")
        builder.add_edge("technical_analysis_node", "risk_analysis_node")
        builder.add_edge("risk_analysis_node", "recommendation_generation_node")
        builder.add_edge("recommendation_generation_node", "ai_summary_node")
        builder.add_edge("ai_summary_node", END)
        
        return builder.compile()

    
    async def _ai_summary_node(self, state: InvestmentState) -> InvestmentState:
        """HyperCLOVA X를 사용한 최종 분석 요약"""
        logger.info("AI 최종 요약 생성 시작")
        
        try:
            # 모든 분석 결과 통합
            summary_data = {
                'technical_analysis': state['technical_analysis'],
                'sentiment_data': state['sentiment_data'],
                'risk_analysis': state['risk_analysis'],
                'recommendations': state['recommendations']
            }
            
            # AI 요약 생성
            ai_summary = await self._generate_comprehensive_summary(summary_data, state['user_preferences'])
            
            # 기존 추천에 AI 요약 추가
            if 'recommendations' in state and state['recommendations']:
                state['recommendations']['ai_summary'] = ai_summary
            else:
                state['recommendations'] = {'ai_summary': ai_summary}
                
        except Exception as e:
            logger.error(f"AI 요약 생성 오류: {e}")
            
        return state
    
    async def _generate_comprehensive_summary(self, analysis_data: Dict[str, Any], 
                                            user_preferences: Dict[str, Any]) -> str:
        """종합적인 AI 분석 요약 생성"""
        
        # 분석 결과를 텍스트로 정리
        summary_text = self._format_analysis_for_summary(analysis_data)
        
        prompt = f"""
다음은 AI 주식 투자 분석 시스템의 종합 분석 결과입니다. 
투자자가 이해하기 쉽게 핵심 내용을 요약하고 실행 가능한 조언을 제공해주세요.

투자자 정보:
- 위험 성향: {user_preferences.get('risk_tolerance', '보통')}
- 투자 기간: {user_preferences.get('investment_horizon', '중기')}

분석 결과:
{summary_text}

다음 형식으로 요약해주세요:

📊 **시장 현황 요약**
- 주요 시장 동향과 투자 환경

💡 **핵심 투자 포인트**
- 가장 중요한 투자 기회와 주의사항

⚠️ **리스크 관리**
- 주요 위험 요소와 대응 방안

🎯 **실행 계획**
- 구체적인 투자 실행 단계

한국어로 친근하고 이해하기 쉽게 작성해주세요.
"""
        
        messages = [
            {"role": "system", "content": "당신은 한국의 개인 투자자를 위한 친근한 투자 어드바이저입니다. 복잡한 분석 결과를 쉽고 실용적으로 설명해주세요."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            summary = await self.clova_client.generate_response(messages, max_tokens=600)
            return summary
        except Exception as e:
            logger.error(f"AI 요약 생성 오류: {e}")
            return "분석 요약 생성 중 오류가 발생했습니다. 개별 분석 결과를 참고해주세요."
    
    def _format_analysis_for_summary(self, analysis_data: Dict[str, Any]) -> str:
        """분석 데이터를 요약용 텍스트로 포맷팅"""
        summary_parts = []
        
        # 기술적 분석 요약
        technical_analysis = analysis_data.get('technical_analysis', {})
        if technical_analysis:
            summary_parts.append("기술적 분석:")
            for ticker, data in technical_analysis.items():
                signal = data.get('overall_signal', 'HOLD')
                summary_parts.append(f"- {ticker}: {signal}")
        
        # 감정 분석 요약
        sentiment_data = analysis_data.get('sentiment_data', {})
        if sentiment_data:
            summary_parts.append("\n감정 분석:")
            for ticker, data in sentiment_data.items():
                score = data.get('sentiment_score', 0)
                sentiment = "긍정적" if score > 0.1 else "부정적" if score < -0.1 else "중립적"
                summary_parts.append(f"- {ticker}: {sentiment} ({score:.2f})")
        
        # 추천 결과 요약
        recommendations = analysis_data.get('recommendations', {})
        if recommendations and 'recommendations' in recommendations:
            summary_parts.append("\n포트폴리오 추천:")
            portfolio = recommendations['recommendations']
            for ticker, data in portfolio.items():
                weight = data.get('weight', 0)
                summary_parts.append(f"- {ticker}: {weight:.1%}")
        
        return "\n".join(summary_parts)
    
    # 기존 노드들은 동일하게 유지
    async def _collect_data_node(self, state: InvestmentState) -> InvestmentState:
        """데이터 수집 노드"""
        logger.info(f"데이터 수집 시작: {state['tickers']}")
        
        try:
            async with self.data_collector as collector:
                real_time_data = await collector.get_real_time_data(state['tickers'])
                historical_data = await collector.get_historical_data(state['tickers'])
                news_data = await collector.get_financial_news(state['tickers'])
                
                state['raw_data'] = {
                    'real_time': real_time_data,
                    'news': news_data
                }
                state['historical_data'] = historical_data
                state['timestamp'] = datetime.now().isoformat()
                
        except Exception as e:
            logger.error(f"데이터 수집 오류: {e}")
            state['raw_data'] = {}
            state['historical_data'] = {}
        
        return state
    
    async def _analyze_sentiment_node(self, state: InvestmentState) -> InvestmentState:
        """감정 분석 노드 (HyperCLOVA X 사용)"""
        logger.info("HyperCLOVA X 감정 분석 시작")
        
        try:
            news_data = state['raw_data'].get('news', [])
            sentiment_results = await self.sentiment_analyzer.analyze_news_sentiment(news_data)
            state['sentiment_data'] = sentiment_results
            
        except Exception as e:
            logger.error(f"감정 분석 오류: {e}")
            state['sentiment_data'] = {}
        
        return state
    
    async def _technical_analysis_node(self, state: InvestmentState) -> InvestmentState:
        """기술적 분석 노드"""
        logger.info("기술적 분석 시작")
        
        try:
            historical_data = state['historical_data']
            technical_results = self.technical_analyzer.analyze_technical_indicators(historical_data)
            state['technical_analysis'] = technical_results
            
        except Exception as e:
            logger.error(f"기술적 분석 오류: {e}")
            state['technical_analysis'] = {}
        
        return state
    
    async def _risk_analysis_node(self, state: InvestmentState) -> InvestmentState:
        """리스크 분석 노드"""
        logger.info("리스크 분석 시작")
        
        try:
            historical_data = state['historical_data']
            user_preferences = state['user_preferences']
            
            risk_results = self.risk_manager.analyze_portfolio_risk(
                historical_data, user_preferences
            )
            state['risk_analysis'] = risk_results
            
        except Exception as e:
            logger.error(f"리스크 분석 오류: {e}")
            state['risk_analysis'] = {}
        
        return state
    
    async def _generate_recommendations_node(self, state: InvestmentState) -> InvestmentState:
        """추천 생성 노드 (HyperCLOVA X 사용)"""
        logger.info("HyperCLOVA X 개인화 추천 생성 시작")
        
        try:
            user_id = state['user_id']
            
            analysis_results = {
                'technical_analysis': state['technical_analysis'],
                'sentiment_data': state['sentiment_data'],
                'risk_analysis': state['risk_analysis'],
                'raw_data': state['raw_data']
            }
            
            recommendations = await self.personalization_agent.generate_personalized_recommendations(
                user_id, analysis_results
            )
            state['recommendations'] = recommendations
            
        except Exception as e:
            logger.error(f"추천 생성 오류: {e}")
            state['recommendations'] = {}
        
        return state
    
    async def run_analysis(self, user_id: str, tickers: list, 
                          user_preferences: dict) -> Dict[str, Any]:
        """전체 분석 실행"""
        initial_state = {
            'user_id': user_id,
            'tickers': tickers,
            'user_preferences': user_preferences,
            'raw_data': {},
            'historical_data': {},
            'sentiment_data': {},
            'technical_analysis': {},
            'fundamental_analysis': {},
            'risk_analysis': {},
            'portfolio_optimization': {},
            'recommendations': {},
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            result = await self.workflow.ainvoke(initial_state)
            return result
        except Exception as e:
            logger.error(f"워크플로우 실행 오류: {e}")
            return {'error': str(e)}
