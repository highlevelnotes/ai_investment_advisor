# ai_analyzer.py
import json
import re
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
                    temperature=0.7,
                    top_p=0.8
                )
                self.available = True
                print("✅ HyperClova X 초기화 성공")
            except Exception as e:
                print(f"❌ HyperClova X 초기화 실패: {e}")
                self.available = False
        else:
            self.available = False
            print("❌ HyperClova X API 키가 없습니다")
    
    def comprehensive_market_analysis(self, macro_data: Dict, etf_data: Dict, user_profile: Dict) -> Dict:
        """종합적인 시장 분석 및 포트폴리오 추천 (통합 기능)"""
        if not self.available:
            return self._get_comprehensive_default_analysis(macro_data, etf_data, user_profile)
        
        try:
            # 매크로 데이터 요약
            macro_summary = self._format_macro_data(macro_data)
            etf_summary = self._format_etf_performance(etf_data)
            profile_summary = self._format_user_profile(user_profile)
            
            # 사용 가능한 ETF 목록
            available_etfs = []
            for category, etfs in etf_data.items():
                for etf_name in list(etfs.keys())[:3]:  # 각 카테고리에서 3개씩
                    available_etfs.append(f"{etf_name} ({category})")
            
            prompt = f"""
당신은 전문 퇴직연금 포트폴리오 매니저입니다. 다음 정보를 종합 분석하여 완전한 투자 솔루션을 제공해주세요.

**사용자 프로필:**
{profile_summary}

**현재 경제 상황:**
{macro_summary}

**ETF 시장 현황:**
{etf_summary}

**투자 가능한 ETF:**
{chr(10).join(available_etfs[:15])}

다음 순서로 종합 분석해주세요:

1. **매크로 경제 분석**: 현재 경제 상황과 향후 전망 (3-4줄)
2. **ETF 시장 동향**: 주요 트렌드와 기회 요인 (3-4줄)  
3. **투자 전략**: 사용자 맞춤형 투자 방향성 (3-4줄)
4. **리스크 요인**: 주의해야 할 위험 요소들 (2-3줄)

그리고 마지막에 구체적인 포트폴리오를 다음 JSON 형식으로 제시해주세요:

{{"portfolio": {{"ETF명1": 비중, "ETF명2": 비중, ...}}, "allocation_reasoning": "배분 근거", "expected_return": "예상수익률", "risk_level": "리스크수준"}}

모든 ETF 비중의 합은 1.0이 되어야 하며, 최소 3개 이상의 ETF를 선택해주세요.
"""
            
            response = self.client.invoke(prompt)
            return self._parse_comprehensive_response(response.content, etf_data, user_profile)
            
        except Exception as e:
            print(f"❌ 종합 시장 분석 실패: {e}")
            return self._get_comprehensive_default_analysis(macro_data, etf_data, user_profile)
    
    def _parse_comprehensive_response(self, response_text: str, etf_data: Dict, user_profile: Dict) -> Dict:
        """AI 응답 파싱"""
        try:
            # 텍스트 분석 부분 추출
            sections = {
                'macro_analysis': '',
                'market_trends': '',
                'investment_strategy': '',
                'risk_factors': ''
            }
            
            # 간단한 텍스트 파싱 (실제로는 더 정교한 파싱 필요)
            lines = response_text.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                if '매크로' in line or '경제' in line:
                    current_section = 'macro_analysis'
                elif 'ETF' in line or '시장' in line:
                    current_section = 'market_trends'
                elif '투자' in line or '전략' in line:
                    current_section = 'investment_strategy'
                elif '리스크' in line or '위험' in line:
                    current_section = 'risk_factors'
                elif current_section and line:
                    sections[current_section] += line + ' '
            
            # JSON 포트폴리오 부분 추출
            json_match = re.search(r'\{[^{}]*"portfolio"[^{}]*\{[^{}]*\}[^{}]*\}', response_text, re.DOTALL)
            portfolio_data = {}
            
            if json_match:
                try:
                    portfolio_json = json.loads(json_match.group())
                    portfolio_data = portfolio_json
                except:
                    pass
            
            # 포트폴리오가 없으면 기본 생성
            if not portfolio_data.get('portfolio'):
                portfolio_data = self._generate_default_portfolio(etf_data, user_profile)
            
            # 포트폴리오 가중치 검증 및 정규화
            weights = portfolio_data.get('portfolio', {})
            validated_weights = self._validate_and_normalize_weights(weights, etf_data)
            
            return {
                'analysis': {
                    'macro_analysis': sections['macro_analysis'].strip() or "현재 경제 상황은 안정적인 성장세를 보이고 있습니다.",
                    'market_trends': sections['market_trends'].strip() or "ETF 시장에서 국내 주식형과 채권형의 균형잡힌 접근이 필요합니다.",
                    'investment_strategy': sections['investment_strategy'].strip() or "분산투자를 통한 안정적인 수익 추구가 바람직합니다.",
                    'risk_factors': sections['risk_factors'].strip() or "시장 변동성과 금리 변화에 주의가 필요합니다."
                },
                'portfolio': {
                    'weights': validated_weights,
                    'allocation_reasoning': portfolio_data.get('allocation_reasoning', 'AI 기반 최적 배분'),
                    'expected_return': portfolio_data.get('expected_return', '6-8%'),
                    'risk_level': portfolio_data.get('risk_level', '중간')
                },
                'source': 'ai_generated'
            }
            
        except Exception as e:
            print(f"❌ AI 응답 파싱 실패: {e}")
            return self._get_comprehensive_default_analysis({}, etf_data, user_profile)
    
    def _validate_and_normalize_weights(self, weights: Dict, etf_data: Dict) -> Dict:
        """가중치 검증 및 정규화"""
        if not weights:
            return {}
        
        # 실제 존재하는 ETF만 필터링
        valid_weights = {}
        all_etf_names = []
        
        for category, etfs in etf_data.items():
            all_etf_names.extend(etfs.keys())
        
        for etf_name, weight in weights.items():
            # 정확한 매칭 또는 유사한 이름 찾기
            matched_name = self._find_matching_etf_name(etf_name, all_etf_names)
            if matched_name and isinstance(weight, (int, float)) and 0 <= weight <= 1:
                valid_weights[matched_name] = float(weight)
        
        # 가중치 정규화
        total_weight = sum(valid_weights.values())
        if total_weight > 0 and len(valid_weights) >= 3:
            valid_weights = {k: v/total_weight for k, v in valid_weights.items()}
            return valid_weights
        
        return {}
    
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
                if keyword in name and len(keyword) > 2:
                    return name
        
        return None
    
    def _generate_default_portfolio(self, etf_data: Dict, user_profile: Dict) -> Dict:
        """기본 포트폴리오 생성"""
        age = user_profile.get('age', 30)
        risk_tolerance = user_profile.get('risk_tolerance', '위험중립형')
        
        # 나이와 위험성향에 따른 기본 배분
        if age < 40:
            if risk_tolerance in ['적극투자형', '위험중립형']:
                base_allocation = {'국내주식형': 0.5, '국내채권형': 0.3, '국내섹터/테마': 0.15, '국내대안투자': 0.05}
            else:
                base_allocation = {'국내주식형': 0.4, '국내채권형': 0.45, '국내섹터/테마': 0.1, '국내대안투자': 0.05}
        elif age < 55:
            base_allocation = {'국내주식형': 0.4, '국내채권형': 0.4, '국내섹터/테마': 0.1, '국내대안투자': 0.1}
        else:
            base_allocation = {'국내주식형': 0.3, '국내채권형': 0.55, '국내섹터/테마': 0.05, '국내대안투자': 0.1}
        
        # 실제 ETF 선택
        portfolio = {}
        for category, target_weight in base_allocation.items():
            if category in etf_data and etf_data[category]:
                etf_list = list(etf_data[category].keys())
                if len(etf_list) >= 2:
                    portfolio[etf_list[0]] = target_weight * 0.6
                    portfolio[etf_list[1]] = target_weight * 0.4
                else:
                    portfolio[etf_list[0]] = target_weight
        
        return {
            'portfolio': portfolio,
            'allocation_reasoning': f'{age}세 {risk_tolerance} 투자자 맞춤 기본 배분',
            'expected_return': '5-7%',
            'risk_level': '중간'
        }
    
    def _get_comprehensive_default_analysis(self, macro_data: Dict, etf_data: Dict, user_profile: Dict) -> Dict:
        """기본 종합 분석"""
        age = user_profile.get('age', 30)
        risk_tolerance = user_profile.get('risk_tolerance', '위험중립형')
        
        # 기본 포트폴리오 생성
        default_portfolio = self._generate_default_portfolio(etf_data, user_profile)
        
        return {
            'analysis': {
                'macro_analysis': f"현재 한국 경제는 GDP 성장률 3.2%, 인플레이션 2.1% 수준으로 안정적인 성장세를 보이고 있습니다. 기준금리 3.5%는 통화정책의 중립적 기조를 반영하며, 전반적으로 투자하기에 양호한 환경입니다.",
                'market_trends': f"국내 ETF 시장에서는 2차전지, 반도체 등 성장 섹터의 관심이 높아지고 있으며, 금리 안정화에 따른 채권 ETF의 매력도도 증가하고 있습니다. 국내 리츠와 금 ETF를 통한 분산투자 효과도 주목받고 있습니다.",
                'investment_strategy': f"{age}세 {risk_tolerance} 투자자에게는 국내 ETF 중심의 분산투자가 적합합니다. 안정성과 성장성의 균형을 추구하며, 정기적인 리밸런싱을 통해 목표 수익률 달성을 추구하는 것이 바람직합니다.",
                'risk_factors': f"글로벌 경제 불확실성과 금리 변동, 지정학적 리스크가 주요 위험 요인입니다. 특히 환율 변동과 원자재 가격 변화에 따른 영향을 주의 깊게 모니터링해야 합니다."
            },
            'portfolio': {
                'weights': default_portfolio['portfolio'],
                'allocation_reasoning': default_portfolio['allocation_reasoning'],
                'expected_return': default_portfolio['expected_return'],
                'risk_level': default_portfolio['risk_level']
            },
            'source': 'default_analysis'
        }
    
    def _format_macro_data(self, macro_data: Dict) -> str:
        """매크로 데이터 포맷팅"""
        if not macro_data:
            return "GDP: 3.2%, 인플레이션: 2.1%, 기준금리: 3.5%"
        
        formatted = []
        for indicator, data in macro_data.items():
            if isinstance(data, dict) and 'current' in data:
                trend_symbol = "📈" if data.get('trend') == 'up' else "📉"
                formatted.append(f"{indicator}: {data['current']:.1f}% {trend_symbol}")
        
        return ", ".join(formatted) if formatted else "경제지표 데이터 없음"
    
    def _format_etf_performance(self, etf_data: Dict) -> str:
        """ETF 성과 데이터 포맷팅"""
        if not etf_data:
            return "ETF 데이터 없음"
        
        formatted = []
        for category, etfs in etf_data.items():
            etf_count = len(etfs)
            formatted.append(f"{category}: {etf_count}개 ETF")
        
        return ", ".join(formatted)
    
    def _format_user_profile(self, user_profile: Dict) -> str:
        """사용자 프로필 포맷팅"""
        return f"""
나이: {user_profile.get('age', 30)}세
투자성향: {user_profile.get('risk_tolerance', '위험중립형')}
투자기간: {user_profile.get('investment_period', 20)}년
현재 자산: {user_profile.get('current_assets', 0):,}원
월 납입액: {user_profile.get('monthly_contribution', 0):,}원
"""
