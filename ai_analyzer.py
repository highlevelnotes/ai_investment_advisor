# ai_analyzer.py
import json
import re
from typing import Dict, Any, Optional
from langchain_naver import ChatClovaX
from config import Config, LIFECYCLE_ALLOCATION, RISK_ALLOCATION
import numpy as np

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
                    temperature=0.3  # 포트폴리오 구성에서는 일관성을 위해 낮은 temperature
                )
                self.available = True
            except Exception as e:
                print(f"HyperClova X 초기화 실패: {e}")
                self.available = False
        else:
            self.available = False
            print("HyperClova X API 키가 설정되지 않았습니다.")
    
    def generate_ai_portfolio_allocation(self, macro_data: Dict, etf_data: Dict, user_profile: Dict) -> Dict:
        """AI 기반 포트폴리오 자산배분 생성"""
        if not self.available:
            return self._get_default_allocation(user_profile)
        
        try:
            macro_summary = self._format_macro_data(macro_data)
            etf_summary = self._format_etf_performance(etf_data)
            profile_summary = self._format_user_profile(user_profile)
            
            prompt = f"""
            현재 경제 상황과 ETF 성과 데이터를 분석하여 최적의 자산배분을 제안해주세요.

            **사용자 프로필:**
            {profile_summary}

            **현재 경제 상황:**
            {macro_summary}

            **ETF 성과 현황:**
            {etf_summary}

            **투자 가능한 자산군:**
            - 국내주식형: 안정적인 대형주 중심, 성장주 포함
            - 국내채권형: 국고채, 회사채, 단기채권
            - 국내섹터/테마: 2차전지, 바이오, 신재생에너지 등
            - 국내대안투자: 금, 원유, 국내 리츠

            다음 조건을 고려하여 자산배분을 제안해주세요:
            1. 현재 경제 상황 (인플레이션, 금리, 성장률 등)
            2. 사용자의 나이와 위험성향
            3. ETF별 최근 성과와 전망
            4. 포트폴리오 분산효과

            **응답 형식 (JSON):**
            {{
                "allocation": {{
                    "국내주식": 0.XX,
                    "국내채권": 0.XX,
                    "국내섹터": 0.XX,
                    "국내대안": 0.XX
                }},
                "reasoning": "자산배분 근거 설명",
                "market_outlook": "시장 전망",
                "risk_assessment": "리스크 평가",
                "adjustment_factors": ["조정 요인1", "조정 요인2"]
            }}

            비중의 합은 반드시 1.0이 되어야 하며, 각 자산군 최소 0.05(5%) 이상 배분해주세요.
            """
            
            response = self.client.invoke(prompt)
            return self._parse_allocation_response(response.content, user_profile)
            
        except Exception as e:
            print(f"AI 포트폴리오 배분 생성 중 오류: {e}")
            return self._get_default_allocation(user_profile)
    
    def generate_specific_etf_selection(self, allocation: Dict, etf_data: Dict, macro_data: Dict) -> Dict:
        """AI 기반 구체적 ETF 종목 선택"""
        if not self.available:
            return self._get_default_etf_selection(allocation, etf_data)
        
        try:
            etf_details = self._format_etf_details(etf_data)
            macro_summary = self._format_macro_data(macro_data)
            
            prompt = f"""
            주어진 자산배분에 따라 구체적인 ETF 종목을 선택하고 비중을 정해주세요.

            **목표 자산배분:**
            {json.dumps(allocation, ensure_ascii=False, indent=2)}

            **현재 경제 상황:**
            {macro_summary}

            **선택 가능한 ETF 상세 정보:**
            {etf_details}

            다음 기준으로 ETF를 선택해주세요:
            1. 현재 경제 상황에 적합한 종목
            2. 최근 성과와 안정성
            3. 유동성과 거래량
            4. 분산투자 효과

            **응답 형식 (JSON):**
            {{
                "selected_etfs": {{
                    "ETF명1": 0.XX,
                    "ETF명2": 0.XX,
                    ...
                }},
                "selection_reasoning": {{
                    "ETF명1": "선택 근거",
                    "ETF명2": "선택 근거",
                    ...
                }},
                "portfolio_strategy": "전체 포트폴리오 전략",
                "rebalancing_trigger": ["리밸런싱 신호1", "리밸런싱 신호2"]
            }}

            선택된 ETF 비중의 합은 반드시 1.0이 되어야 합니다.
            """
            
            response = self.client.invoke(prompt)
            return self._parse_etf_selection_response(response.content, allocation, etf_data)
            
        except Exception as e:
            print(f"AI ETF 선택 중 오류: {e}")
            return self._get_default_etf_selection(allocation, etf_data)
    
    def analyze_market_situation(self, macro_data: Dict, etf_data: Dict) -> str:
        """시장 상황 분석"""
        if not self.available:
            return self._get_sample_market_analysis()
        
        try:
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
            5. 포트폴리오 구성에 미치는 영향

            분석 결과를 명확하고 이해하기 쉽게 설명해주세요.
            """
            
            response = self.client.invoke(prompt)
            return response.content
            
        except Exception as e:
            print(f"시장 분석 중 오류: {e}")
            return self._get_sample_market_analysis()
    
    def _parse_allocation_response(self, response_text: str, user_profile: Dict) -> Dict:
        """AI 응답에서 자산배분 파싱"""
        try:
            # JSON 부분 추출
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed_data = json.loads(json_str)
                
                allocation = parsed_data.get('allocation', {})
                
                # 비중 정규화
                total_weight = sum(allocation.values())
                if total_weight > 0:
                    allocation = {k: v/total_weight for k, v in allocation.items()}
                
                # 최소 비중 보장
                for asset_class in ['국내주식', '국내채권', '국내섹터', '국내대안']:
                    if asset_class not in allocation:
                        allocation[asset_class] = 0.05
                
                # 재정규화
                total_weight = sum(allocation.values())
                allocation = {k: v/total_weight for k, v in allocation.items()}
                
                return {
                    'allocation': allocation,
                    'reasoning': parsed_data.get('reasoning', ''),
                    'market_outlook': parsed_data.get('market_outlook', ''),
                    'risk_assessment': parsed_data.get('risk_assessment', ''),
                    'adjustment_factors': parsed_data.get('adjustment_factors', [])
                }
            else:
                raise ValueError("JSON 형식을 찾을 수 없습니다.")
                
        except Exception as e:
            print(f"AI 응답 파싱 실패: {e}")
            return self._get_default_allocation(user_profile)
    
    def _parse_etf_selection_response(self, response_text: str, allocation: Dict, etf_data: Dict) -> Dict:
        """AI 응답에서 ETF 선택 파싱"""
        try:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                parsed_data = json.loads(json_str)
                
                selected_etfs = parsed_data.get('selected_etfs', {})
                
                # ETF 이름 검증 및 정규화
                validated_etfs = {}
                all_etf_names = []
                for category_etfs in etf_data.values():
                    all_etf_names.extend(category_etfs.keys())
                
                for etf_name, weight in selected_etfs.items():
                    # 정확한 이름 매칭 또는 유사한 이름 찾기
                    matched_name = self._find_matching_etf_name(etf_name, all_etf_names)
                    if matched_name:
                        validated_etfs[matched_name] = weight
                
                # 비중 정규화
                total_weight = sum(validated_etfs.values())
                if total_weight > 0:
                    validated_etfs = {k: v/total_weight for k, v in validated_etfs.items()}
                
                return {
                    'weights': validated_etfs,
                    'selection_reasoning': parsed_data.get('selection_reasoning', {}),
                    'portfolio_strategy': parsed_data.get('portfolio_strategy', ''),
                    'rebalancing_trigger': parsed_data.get('rebalancing_trigger', [])
                }
            else:
                raise ValueError("JSON 형식을 찾을 수 없습니다.")
                
        except Exception as e:
            print(f"ETF 선택 파싱 실패: {e}")
            return self._get_default_etf_selection(allocation, etf_data)
    
    def _find_matching_etf_name(self, ai_name: str, available_names: list) -> str:
        """AI가 제안한 ETF 이름과 실제 ETF 이름 매칭"""
        # 정확한 매칭
        if ai_name in available_names:
            return ai_name
        
        # 부분 매칭
        for name in available_names:
            if ai_name in name or name in ai_name:
                return name
        
        # 키워드 매칭
        ai_keywords = ai_name.replace('KODEX ', '').replace('TIGER ', '').split()
        for name in available_names:
            for keyword in ai_keywords:
                if keyword in name:
                    return name
        
        return None
    
    def _get_default_allocation(self, user_profile: Dict) -> Dict:
        """기본 자산배분 반환"""
        age = user_profile.get('age', 30)
        risk_tolerance = user_profile.get('risk_tolerance', '위험중립형')
        
        # 생애주기 기반 기본 배분
        if age < 40:
            lifecycle_stage = '청년층'
        elif age < 55:
            lifecycle_stage = '중년층'
        else:
            lifecycle_stage = '장년층'
        
        base_allocation = LIFECYCLE_ALLOCATION[lifecycle_stage]
        risk_allocation = RISK_ALLOCATION[risk_tolerance]
        
        # 가중평균
        final_allocation = {}
        for asset_class in base_allocation.keys():
            final_allocation[asset_class] = (
                base_allocation[asset_class] * 0.7 + 
                risk_allocation[asset_class] * 0.3
            )
        
        return {
            'allocation': final_allocation,
            'reasoning': f'{lifecycle_stage} 및 {risk_tolerance} 성향 기반 기본 배분',
            'market_outlook': '중립적 시장 전망',
            'risk_assessment': '보통 수준의 리스크',
            'adjustment_factors': ['생애주기', '위험성향']
        }
    
    def _get_default_etf_selection(self, allocation: Dict, etf_data: Dict) -> Dict:
        """기본 ETF 선택"""
        weights = {}
        
        # 각 카테고리별로 대표 ETF 선택
        category_mapping = {
            '국내주식': '국내주식형',
            '국내채권': '국내채권형',
            '국내섹터': '국내섹터/테마',
            '국내대안': '국내대안투자'
        }
        
        for asset_class, target_weight in allocation.items():
            if asset_class in category_mapping:
                category = category_mapping[asset_class]
                if category in etf_data:
                    category_etfs = list(etf_data[category].keys())
                    if category_etfs:
                        # 첫 번째 ETF에 전체 비중 할당 (단순화)
                        weights[category_etfs[0]] = target_weight
        
        return {
            'weights': weights,
            'selection_reasoning': {},
            'portfolio_strategy': '기본 분산투자 전략',
            'rebalancing_trigger': ['시장 변동성 확대', '경제지표 변화']
        }
    
    def _format_macro_data(self, macro_data: Dict) -> str:
        """매크로 데이터 포맷팅"""
        formatted = []
        for indicator, data in macro_data.items():
            trend_symbol = "📈" if data['trend'] == 'up' else "📉"
            change = data['current'] - data['previous']
            formatted.append(f"- {indicator}: {data['current']:.2f}{data.get('unit', '')} {trend_symbol} (변화: {change:+.2f})")
        return "\n".join(formatted)
    
    def _format_etf_performance(self, etf_data: Dict) -> str:
        """ETF 성과 데이터 포맷팅"""
        formatted = []
        for category, etfs in etf_data.items():
            formatted.append(f"\n**{category}:**")
            for name, data in list(etfs.items())[:3]:  # 상위 3개만 표시
                if 'returns' in data and not data['returns'].empty:
                    annual_return = data['returns'].mean() * 252 * 100
                    annual_vol = data['returns'].std() * np.sqrt(252) * 100
                else:
                    annual_return = 0
                    annual_vol = 0
                formatted.append(f"  - {name}: 수익률 {annual_return:.1f}%, 변동성 {annual_vol:.1f}%")
        return "\n".join(formatted)
    
    def _format_etf_details(self, etf_data: Dict) -> str:
        """ETF 상세 정보 포맷팅"""
        formatted = []
        for category, etfs in etf_data.items():
            formatted.append(f"\n**{category}:**")
            for name, data in etfs.items():
                if 'returns' in data and not data['returns'].empty:
                    annual_return = data['returns'].mean() * 252 * 100
                    annual_vol = data['returns'].std() * np.sqrt(252) * 100
                    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
                else:
                    annual_return = 0
                    annual_vol = 0
                    sharpe = 0
                
                formatted.append(f"  - {name}: 수익률 {annual_return:.1f}%, 변동성 {annual_vol:.1f}%, 샤프비율 {sharpe:.2f}")
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
    
    def _format_etf_data(self, etf_data: Dict) -> str:
        """ETF 데이터 포맷팅"""
        formatted = []
        for category, etfs in etf_data.items():
            formatted.append(f"\n**{category}:**")
            for name, data in list(etfs.items())[:2]:
                returns_pct = data['returns'].mean() * 252 * 100 if 'returns' in data and not data['returns'].empty else 0
                formatted.append(f"  - {name}: {data['price']:.0f}원 (연환산 수익률: {returns_pct:.1f}%)")
        return "\n".join(formatted)
    
    def _get_sample_market_analysis(self) -> str:
        """샘플 시장 분석"""
        return """
        ## 🏛️ 현재 경제 상황 종합 평가
        
        한국 경제는 전반적으로 안정적인 성장세를 유지하고 있습니다. GDP 성장률이 3.2%로 전월 대비 상승하며 견조한 경제 회복세를 보이고 있습니다.
        
        ## 📊 ETF 시장의 주요 트렌드
        
        - **국내주식형**: KODEX 200을 중심으로 안정적인 성과 유지
        - **국내채권형**: 금리 상승 국면에서 단기채권 선호 현상
        - **국내섹터/테마**: 2차전지, 바이오 등 성장주 테마 주목
        - **국내대안투자**: 금과 국내 리츠의 분산투자 효과 부각
        
        ## 🔮 향후 3-6개월 시장 전망
        
        중앙은행의 통화정책 정상화 과정에서 시장 변동성이 확대될 가능성이 있습니다. 다만 기업 실적 개선과 구조적 성장 동력은 긍정적 요인으로 작용할 것으로 예상됩니다.
        """
