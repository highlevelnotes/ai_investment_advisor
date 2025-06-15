# ai_analyzer.py
import json
from typing import Dict, Any, Optional
from langchain_naver import ChatClovaX
from config import Config

class AIAnalyzer:
    def __init__(self):
        """LangChain HyperClova X 초기화"""
        self.api_key = Config.HYPERCLOVA_X_API_KEY
        self.model_name = Config.HYPERCLOVA_MODEL
        self.max_tokens = Config.HYPERCLOVA_MAX_TOKENS
        
        if self.api_key:
            try:
                self.client = ChatClovaX(
                    api_key=self.api_key,
                    model=self.model_name,
                    max_tokens=self.max_tokens,
                    temperature=0.7
                )
                self.available = True
            except Exception as e:
                print(f"HyperClova X 초기화 실패: {e}")
                self.available = False
        else:
            self.available = False
            print("HyperClova X API 키가 설정되지 않았습니다.")
    
    def analyze_market_situation(self, macro_data: Dict, etf_data: Dict) -> str:
        """시장 상황 분석"""
        if not self.available:
            return self._get_sample_market_analysis()
        
        try:
            # 매크로 데이터 요약
            macro_summary = self._format_macro_data(macro_data)
            etf_summary = self._format_etf_data(etf_data)
            
            prompt = f"""
            현재 한국 경제 상황과 ETF 시장 데이터를 분석해주세요.

            **매크로 경제 지표:**
            {macro_summary}

            **ETF 시장 현황:**
            {etf_summary}

            다음 관점에서 분석해주세요:
            1. 현재 경제 상황 종합 평가
            2. ETF 시장의 주요 트렌드
            3. 향후 3-6개월 시장 전망
            4. 퇴직연금 투자자가 주의해야 할 리스크 요인

            분석 결과를 명확하고 이해하기 쉽게 설명해주세요.
            """
            
            response = self.client.invoke(prompt)
            return response.content
            
        except Exception as e:
            print(f"시장 분석 중 오류: {e}")
            return self._get_sample_market_analysis()
    
    def generate_portfolio_recommendation(self, macro_data: Dict, etf_data: Dict, user_profile: Dict) -> str:
        """개인 맞춤형 포트폴리오 추천"""
        if not self.available:
            return self._get_sample_portfolio_recommendation(user_profile)
        
        try:
            macro_summary = self._format_macro_data(macro_data)
            etf_summary = self._format_etf_data(etf_data)
            profile_summary = self._format_user_profile(user_profile)
            
            prompt = f"""
            다음 정보를 바탕으로 개인 맞춤형 ETF 퇴직연금 포트폴리오를 추천해주세요.

            **사용자 프로필:**
            {profile_summary}

            **현재 경제 상황:**
            {macro_summary}

            **ETF 시장 데이터:**
            {etf_summary}

            다음 사항을 포함하여 추천해주세요:
            1. 추천 포트폴리오 구성 (카테고리별 비중)
            2. 구체적인 ETF 종목 추천 (3-5개)
            3. 추천 근거 및 투자 논리
            4. 예상 수익률과 리스크 수준
            5. 리밸런싱 전략
            6. 주의사항 및 모니터링 포인트

            안정성을 중시하는 퇴직연금 특성을 고려하여 추천해주세요.
            """
            
            response = self.client.invoke(prompt)
            return response.content
            
        except Exception as e:
            print(f"포트폴리오 추천 중 오류: {e}")
            return self._get_sample_portfolio_recommendation(user_profile)
    
    def analyze_portfolio_performance(self, portfolio_data: Dict, benchmark_data: Dict) -> str:
        """포트폴리오 성과 분석"""
        if not self.available:
            return self._get_sample_performance_analysis()
        
        try:
            performance_summary = self._format_performance_data(portfolio_data, benchmark_data)
            
            prompt = f"""
            다음 포트폴리오 성과 데이터를 분석해주세요.

            **성과 데이터:**
            {performance_summary}

            다음 관점에서 분석해주세요:
            1. 절대 수익률 평가
            2. 벤치마크 대비 상대 성과
            3. 리스크 조정 수익률 (샤프 비율 등)
            4. 최대 낙폭(MDD) 분석
            5. 성과 기여 요인 분석
            6. 개선 방안 제시

            퇴직연금 투자자 관점에서 이해하기 쉽게 설명해주세요.
            """
            
            response = self.client.invoke(prompt)
            return response.content
            
        except Exception as e:
            print(f"성과 분석 중 오류: {e}")
            return self._get_sample_performance_analysis()
    
    def _format_macro_data(self, macro_data: Dict) -> str:
        """매크로 데이터 포맷팅"""
        formatted = []
        for indicator, data in macro_data.items():
            trend_symbol = "📈" if data['trend'] == 'up' else "📉"
            formatted.append(f"- {indicator}: {data['current']:.2f} {trend_symbol} (전월: {data['previous']:.2f})")
        return "\n".join(formatted)
    
    def _format_etf_data(self, etf_data: Dict) -> str:
        """ETF 데이터 포맷팅"""
        formatted = []
        for category, etfs in etf_data.items():
            formatted.append(f"\n**{category}:**")
            for name, data in list(etfs.items())[:2]:  # 상위 2개만 표시
                returns_pct = data['returns'].mean() * 252 * 100  # 연환산 수익률
                formatted.append(f"  - {name}: {data['price']:.0f}원 (연환산 수익률: {returns_pct:.1f}%)")
        return "\n".join(formatted)
    
    def _format_user_profile(self, user_profile: Dict) -> str:
        """사용자 프로필 포맷팅"""
        return f"""
        - 나이: {user_profile.get('age', 30)}세
        - 투자성향: {user_profile.get('risk_tolerance', '위험중립형')}
        - 투자기간: {user_profile.get('investment_period', 20)}년
        - 현재 자산: {user_profile.get('current_assets', 0):,}원
        - 월 납입액: {user_profile.get('monthly_contribution', 0):,}원
        """
    
    def _format_performance_data(self, portfolio_data: Dict, benchmark_data: Dict) -> str:
        """성과 데이터 포맷팅"""
        return f"""
        - 포트폴리오 수익률: {portfolio_data.get('return', 0):.2f}%
        - 벤치마크 수익률: {benchmark_data.get('return', 0):.2f}%
        - 포트폴리오 변동성: {portfolio_data.get('volatility', 0):.2f}%
        - 샤프 비율: {portfolio_data.get('sharpe_ratio', 0):.2f}
        - 최대 낙폭: {portfolio_data.get('max_drawdown', 0):.2f}%
        """
    
    def _get_sample_market_analysis(self) -> str:
        """샘플 시장 분석"""
        return """
        ## 🏛️ 현재 경제 상황 종합 평가
        
        한국 경제는 전반적으로 안정적인 성장세를 유지하고 있습니다. GDP 성장률이 3.2%로 전월 대비 상승하며 견조한 경제 회복세를 보이고 있습니다. 다만 글로벌 인플레이션 압력과 지정학적 리스크가 여전히 주요 변수로 작용하고 있습니다.
        
        ## 📊 ETF 시장의 주요 트렌드
        
        - **국내주식형**: KODEX 200을 중심으로 안정적인 성과 유지
        - **해외주식형**: 미국 시장 ETF의 강세 지속
        - **채권형**: 금리 상승 국면에서 단기채권 선호 현상
        - **섹터/테마**: 2차전지, 바이오 등 성장주 테마 주목
        
        ## 🔮 향후 3-6개월 시장 전망
        
        중앙은행의 통화정책 정상화 과정에서 시장 변동성이 확대될 가능성이 있습니다. 다만 기업 실적 개선과 구조적 성장 동력은 긍정적 요인으로 작용할 것으로 예상됩니다.
        
        ## ⚠️ 주요 리스크 요인
        
        1. 글로벌 금리 상승에 따른 자금 유출 우려
        2. 지정학적 긴장 고조로 인한 시장 불안
        3. 원자재 가격 변동성 확대
        4. 국내 부동산 시장 조정 영향
        """
    
    def _get_sample_portfolio_recommendation(self, user_profile: Dict) -> str:
        """샘플 포트폴리오 추천"""
        age = user_profile.get('age', 30)
        risk_tolerance = user_profile.get('risk_tolerance', '위험중립형')
        
        if age < 40:
            stock_ratio = 60
            bond_ratio = 30
            alternative_ratio = 10
        elif age < 55:
            stock_ratio = 50
            bond_ratio = 40
            alternative_ratio = 10
        else:
            stock_ratio = 40
            bond_ratio = 50
            alternative_ratio = 10
        
        return f"""
        ## 🎯 개인 맞춤형 포트폴리오 추천
        
        **추천 자산배분:**
        - 주식형 ETF: {stock_ratio}%
        - 채권형 ETF: {bond_ratio}%
        - 대안투자 ETF: {alternative_ratio}%
        
        ## 📋 구체적 ETF 종목 추천
        
        1. **KODEX 200 (069500)** - 25%
           - 국내 대표 지수 추종으로 안정성 확보
        
        2. **KODEX 미국S&P500선물(H) (138230)** - 20%
           - 글로벌 분산투자 효과
        
        3. **KODEX 국고채10년 (148070)** - 25%
           - 안정적인 채권 수익 확보
        
        4. **KODEX 2차전지산업 (117700)** - 15%
           - 성장 테마 투자
        
        5. **KODEX 골드선물(H) (132030)** - 15%
           - 인플레이션 헤지
        
        ## 💡 투자 논리 및 근거
        
        현재 {age}세, {risk_tolerance} 성향을 고려하여 안정성과 성장성의 균형을 추구하는 포트폴리오를 구성했습니다. 국내외 분산투자를 통해 리스크를 관리하면서도 장기 성장 가능성을 확보했습니다.
        
        ## 📈 예상 성과
        
        - 연평균 기대수익률: 6-8%
        - 예상 변동성: 12-15%
        - 샤프 비율: 0.4-0.6
        
        ## 🔄 리밸런싱 전략
        
        분기별로 목표 비중에서 ±5% 이상 이탈시 리밸런싱을 실시하며, 시장 상황에 따라 전술적 조정을 수행합니다.
        """
    
    def _get_sample_performance_analysis(self) -> str:
        """샘플 성과 분석"""
        return """
        ## 📊 포트폴리오 성과 분석 결과
        
        ### 절대 수익률 평가
        현재 포트폴리오는 연환산 7.2%의 양호한 수익률을 기록하고 있습니다. 이는 퇴직연금 평균 수익률 4.8%를 상회하는 우수한 성과입니다.
        
        ### 벤치마크 대비 성과
        KOSPI 200 대비 +1.8%의 초과수익을 달성했으며, 이는 효과적인 자산배분과 종목 선택의 결과입니다.
        
        ### 리스크 조정 수익률
        샤프 비율 0.52로 위험 대비 수익률이 양호한 수준입니다. 변동성 관리가 효과적으로 이루어지고 있습니다.
        
        ### 최대 낙폭 분석
        MDD -8.3%로 시장 평균 대비 낮은 수준의 손실 위험을 보이고 있어 안정성이 확보되었습니다.
        
        ### 개선 방안
        1. 해외 분산투자 비중 확대 검토
        2. 리밸런싱 주기 최적화
        3. 대안투자 비중 조정 고려
        """
