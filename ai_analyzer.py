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
        """종합적인 시장 분석 및 다중 ETF 포트폴리오 추천"""
        if not self.available:
            return self._get_comprehensive_default_analysis(macro_data, etf_data, user_profile)
        
        try:
            # 매크로 데이터 요약
            macro_summary = self._format_macro_data(macro_data)
            etf_summary = self._format_detailed_etf_data(etf_data)
            profile_summary = self._format_user_profile(user_profile)
            
            prompt = f"""
당신은 전문 퇴직연금 포트폴리오 매니저입니다. 다음 정보를 종합 분석하여 각 자산군 내에서 여러 ETF를 조합한 완전한 투자 솔루션을 제공해주세요.

**사용자 프로필:**
{profile_summary}

**현재 경제 상황:**
{macro_summary}

**ETF 시장 현황:**
{etf_summary}

다음 순서로 종합 분석해주세요:

1. **매크로 경제 분석**: 현재 경제 상황과 향후 전망 (3-4줄)
2. **ETF 시장 동향**: 주요 트렌드와 기회 요인 (3-4줄)  
3. **투자 전략**: 사용자 맞춤형 투자 방향성 (3-4줄)
4. **리스크 요인**: 주의해야 할 위험 요소들 (2-3줄)

그리고 마지막에 **각 자산군별로 여러 ETF를 조합한** 구체적인 포트폴리오를 다음 JSON 형식으로 제시해주세요:

{{"portfolio": {{"ETF명1": 비중, "ETF명2": 비중, "ETF명3": 비중, ...}}, "allocation_reasoning": "배분 근거", "diversification_strategy": "분산투자 전략"}}

**중요 요구사항:**
- 각 자산군(국내주식형, 국내채권형, 국내섹터/테마, 국내대안투자)에서 최소 2-3개 ETF 선택
- 총 8-12개 ETF로 구성
- 모든 ETF 비중의 합은 1.0이 되어야 함
- 각 ETF의 최소 비중은 0.05(5%) 이상
- 동일 자산군 내에서도 서로 다른 특성의 ETF 조합 (예: 대형주+중소형주, 단기채권+장기채권)
"""
            
            response = self.client.invoke(prompt)
            return self._parse_multi_etf_response(response.content, etf_data, user_profile)
            
        except Exception as e:
            print(f"❌ 종합 시장 분석 실패: {e}")
            return self._get_comprehensive_default_analysis(macro_data, etf_data, user_profile)
    
    def _parse_multi_etf_response(self, response_text: str, etf_data: Dict, user_profile: Dict) -> Dict:
        """다중 ETF 응답 파싱 - 개선된 버전"""
        try:
            # 텍스트 분석 부분 추출 - 더 정확한 파싱
            sections = {
                'macro_analysis': '',
                'market_trends': '',
                'investment_strategy': '',
                'risk_factors': ''
            }
            
            lines = response_text.split('\n')
            current_section = None
            content_buffer = []
            
            for line in lines:
                line = line.strip()
                
                # 섹션 헤더 감지 (더 정확한 패턴 매칭)
                if any(keyword in line.lower() for keyword in ['매크로', '경제 분석', '경제상황', '경제 상황']):
                    if current_section and content_buffer:
                        sections[current_section] = ' '.join(content_buffer)
                    current_section = 'macro_analysis'
                    content_buffer = []
                elif any(keyword in line.lower() for keyword in ['etf', '시장 동향', '시장동향', '시장 트렌드']):
                    if current_section and content_buffer:
                        sections[current_section] = ' '.join(content_buffer)
                    current_section = 'market_trends'
                    content_buffer = []
                elif any(keyword in line.lower() for keyword in ['투자 전략', '투자전략', '투자 방향', '투자방향']):
                    if current_section and content_buffer:
                        sections[current_section] = ' '.join(content_buffer)
                    current_section = 'investment_strategy'
                    content_buffer = []
                elif any(keyword in line.lower() for keyword in ['리스크', '위험', '리스크 요인', '위험 요인']):
                    if current_section and content_buffer:
                        sections[current_section] = ' '.join(content_buffer)
                    current_section = 'risk_factors'
                    content_buffer = []
                elif current_section and line and not line.startswith('{') and not line.startswith('**'):
                    # 내용 수집 (JSON이나 마크다운 헤더가 아닌 경우)
                    content_buffer.append(line)
            
            # 마지막 섹션 처리
            if current_section and content_buffer:
                sections[current_section] = ' '.join(content_buffer)
            
            # 빈 섹션에 대한 기본값 설정
            if not sections['macro_analysis'].strip():
                sections['macro_analysis'] = self._generate_default_macro_analysis()
            
            if not sections['market_trends'].strip():
                sections['market_trends'] = self._generate_default_market_trends()
            
            if not sections['investment_strategy'].strip():
                sections['investment_strategy'] = self._generate_default_investment_strategy(user_profile)
            
            if not sections['risk_factors'].strip():
                sections['risk_factors'] = self._generate_default_risk_factors()
            
            # JSON 포트폴리오 부분 추출
            json_match = re.search(r'\{[^{}]*"portfolio"[^{}]*\{[^{}]*\}[^{}]*\}', response_text, re.DOTALL)
            portfolio_data = {}
            
            if json_match:
                try:
                    portfolio_json = json.loads(json_match.group())
                    portfolio_data = portfolio_json
                except:
                    pass
            
            # 포트폴리오가 없거나 부족하면 다중 ETF 기본 생성
            if not portfolio_data.get('portfolio') or len(portfolio_data.get('portfolio', {})) < 6:
                portfolio_data = self._generate_multi_etf_portfolio(etf_data, user_profile)
            
            # 포트폴리오 가중치 검증 및 정규화
            weights = portfolio_data.get('portfolio', {})
            validated_weights = self._validate_and_normalize_multi_weights(weights, etf_data)
            
            return {
                'analysis': {
                    'macro_analysis': sections['macro_analysis'],
                    'market_trends': sections['market_trends'],
                    'investment_strategy': sections['investment_strategy'],
                    'risk_factors': sections['risk_factors']
                },
                'portfolio': {
                    'weights': validated_weights,
                    'allocation_reasoning': portfolio_data.get('allocation_reasoning', '다중 ETF 기반 정교한 분산투자'),
                    'diversification_strategy': portfolio_data.get('diversification_strategy', '자산군 내외 이중 분산투자'),
                    'etf_count': len(validated_weights),
                    'category_distribution': self._analyze_category_distribution(validated_weights, etf_data)
                },
                'source': 'ai_generated_multi_etf'
            }
            
        except Exception as e:
            print(f"❌ AI 응답 파싱 실패: {e}")
            return self._get_comprehensive_default_analysis({}, etf_data, user_profile)

    def _generate_default_macro_analysis(self) -> str:
        """기본 매크로 분석"""
        return "현재 한국 경제는 GDP 성장률 3.2%, 인플레이션 2.1% 수준으로 안정적인 성장세를 보이고 있습니다. 기준금리 3.5%는 통화정책의 중립적 기조를 반영하며, 전반적으로 투자하기에 양호한 환경입니다. 글로벌 경제 불확실성이 있지만 국내 경제의 펀더멘털은 견고한 상태를 유지하고 있습니다."

    def _generate_default_market_trends(self) -> str:
        """기본 ETF 시장 동향"""
        return "국내 ETF 시장에서는 2차전지, 반도체 등 성장 섹터의 관심이 높아지고 있으며, 금리 안정화에 따른 채권 ETF의 매력도도 증가하고 있습니다. 특히 각 자산군 내에서도 다양한 ETF를 조합하는 정교한 분산투자 전략이 주목받고 있으며, 국내 리츠와 금 ETF를 통한 대안투자 수요도 늘어나고 있습니다."

    def _generate_default_investment_strategy(self, user_profile: Dict) -> str:
        """기본 투자 전략"""
        age = user_profile.get('age', 30)
        risk_tolerance = user_profile.get('risk_tolerance', '위험중립형')
        
        return f"{age}세 {risk_tolerance} 투자자에게는 8-12개 ETF를 활용한 다중 분산투자가 적합합니다. 각 자산군 내에서 2-3개 ETF를 조합하여 이중 분산효과를 추구하며, 정기적인 리밸런싱을 통해 목표 수익률 달성을 추구하는 것이 바람직합니다. 국내 ETF만을 활용하여 환율 리스크를 제거하면서도 충분한 분산투자 효과를 얻을 수 있습니다."

    def _generate_default_risk_factors(self) -> str:
        """기본 리스크 요인"""
        return "글로벌 경제 불확실성과 금리 변동, 지정학적 리스크가 주요 위험 요인입니다. 특히 개별 ETF 간 상관관계 변화와 시장 집중도 리스크에 주의가 필요하며, 정기적인 리밸런싱을 통한 분산효과 유지가 중요합니다."

    def comprehensive_market_analysis(self, macro_data: Dict, etf_data: Dict, user_profile: Dict) -> Dict:
        """종합적인 시장 분석 및 다중 ETF 포트폴리오 추천 - 개선된 프롬프트"""
        if not self.available:
            return self._get_comprehensive_default_analysis(macro_data, etf_data, user_profile)
        
        try:
            # 매크로 데이터 요약
            macro_summary = self._format_macro_data(macro_data)
            etf_summary = self._format_detailed_etf_data(etf_data)
            profile_summary = self._format_user_profile(user_profile)
            
            prompt = f"""
    당신은 전문 퇴직연금 포트폴리오 매니저입니다. 다음 정보를 종합 분석하여 각 자산군 내에서 여러 ETF를 조합한 완전한 투자 솔루션을 제공해주세요.

    **사용자 프로필:**
    {profile_summary}

    **현재 경제 상황:**
    {macro_summary}

    **ETF 시장 현황:**
    {etf_summary}

    반드시 다음 4개 섹션으로 나누어 분석해주세요:

    **1. 매크로 경제 분석**
    현재 경제 상황과 향후 전망을 3-4문장으로 설명해주세요.

    **2. ETF 시장 동향**  
    국내 ETF 시장의 주요 트렌드와 기회 요인을 3-4문장으로 설명해주세요.

    **3. 투자 전략**
    사용자 맞춤형 투자 방향성과 전략을 3-4문장으로 설명해주세요.

    **4. 리스크 요인**
    주의해야 할 위험 요소들을 2-3문장으로 설명해주세요.

    그리고 마지막에 **각 자산군별로 여러 ETF를 조합한** 구체적인 포트폴리오를 다음 JSON 형식으로 제시해주세요:

    {{"portfolio": {{"ETF명1": 비중, "ETF명2": 비중, "ETF명3": 비중, ...}}, "allocation_reasoning": "배분 근거", "diversification_strategy": "분산투자 전략"}}

    **중요 요구사항:**
    - 각 자산군(국내주식형, 국내채권형, 국내섹터/테마, 국내대안투자)에서 최소 2-3개 ETF 선택
    - 총 8-12개 ETF로 구성
    - 모든 ETF 비중의 합은 1.0이 되어야 함
    - 각 ETF의 최소 비중은 0.05(5%) 이상

    각 섹션을 명확히 구분하여 답변해주세요.
    """
            
            response = self.client.invoke(prompt)
            return self._parse_multi_etf_response(response.content, etf_data, user_profile)
            
        except Exception as e:
            print(f"❌ 종합 시장 분석 실패: {e}")
            return self._get_comprehensive_default_analysis(macro_data, etf_data, user_profile)
    
    def _validate_and_normalize_multi_weights(self, weights: Dict, etf_data: Dict) -> Dict:
        """다중 ETF 가중치 검증 및 정규화"""
        if not weights:
            return {}
        
        # 실제 존재하는 ETF만 필터링
        valid_weights = {}
        all_etf_names = []
        
        for category, etfs in etf_data.items():
            all_etf_names.extend(etfs.keys())
        
        for etf_name, weight in weights.items():
            matched_name = self._find_matching_etf_name(etf_name, all_etf_names)
            if matched_name and isinstance(weight, (int, float)) and 0 <= weight <= 1:
                valid_weights[matched_name] = float(weight)
        
        # 최소 비중 확인 (5% 이상)
        filtered_weights = {k: v for k, v in valid_weights.items() if v >= 0.05}
        
        # 가중치 정규화
        total_weight = sum(filtered_weights.values())
        if total_weight > 0 and len(filtered_weights) >= 6:  # 최소 6개 ETF
            filtered_weights = {k: v/total_weight for k, v in filtered_weights.items()}
            return filtered_weights
        
        return {}
    
    def _generate_multi_etf_portfolio(self, etf_data: Dict, user_profile: Dict) -> Dict:
        """다중 ETF 기본 포트폴리오 생성"""
        age = user_profile.get('age', 30)
        risk_tolerance = user_profile.get('risk_tolerance', '위험중립형')
        
        # 나이와 위험성향에 따른 자산군별 기본 배분
        if age < 40:
            if risk_tolerance in ['적극투자형', '위험중립형']:
                base_allocation = {
                    '국내주식형': 0.50,
                    '국내채권형': 0.25,
                    '국내섹터/테마': 0.15,
                    '국내대안투자': 0.10
                }
            else:
                base_allocation = {
                    '국내주식형': 0.35,
                    '국내채권형': 0.45,
                    '국내섹터/테마': 0.10,
                    '국내대안투자': 0.10
                }
        elif age < 55:
            base_allocation = {
                '국내주식형': 0.40,
                '국내채권형': 0.35,
                '국내섹터/테마': 0.15,
                '국내대안투자': 0.10
            }
        else:
            base_allocation = {
                '국내주식형': 0.25,
                '국내채권형': 0.55,
                '국내섹터/테마': 0.10,
                '국내대안투자': 0.10
            }
        
        # 각 자산군에서 다중 ETF 선택
        portfolio = {}
        
        for category, target_weight in base_allocation.items():
            if category in etf_data and etf_data[category]:
                etf_list = list(etf_data[category].keys())
                
                if len(etf_list) >= 3:
                    # 3개 ETF로 분산
                    portfolio[etf_list[0]] = target_weight * 0.5   # 50%
                    portfolio[etf_list[1]] = target_weight * 0.3   # 30%
                    portfolio[etf_list[2]] = target_weight * 0.2   # 20%
                elif len(etf_list) >= 2:
                    # 2개 ETF로 분산
                    portfolio[etf_list[0]] = target_weight * 0.6   # 60%
                    portfolio[etf_list[1]] = target_weight * 0.4   # 40%
                else:
                    # 1개만 있는 경우
                    portfolio[etf_list[0]] = target_weight
        
        return {
            'portfolio': portfolio,
            'allocation_reasoning': f'{age}세 {risk_tolerance} 투자자를 위한 다중 ETF 분산투자 전략',
            'diversification_strategy': '각 자산군 내 2-3개 ETF 조합으로 이중 분산효과 추구'
        }
    
    def _analyze_category_distribution(self, weights: Dict, etf_data: Dict) -> Dict:
        """카테고리별 분포 분석"""
        category_weights = {}
        etf_by_category = {}
        
        # ETF를 카테고리별로 분류
        for category, etfs in etf_data.items():
            category_weights[category] = 0
            etf_by_category[category] = []
            
            for etf_name in etfs.keys():
                if etf_name in weights:
                    category_weights[category] += weights[etf_name]
                    etf_by_category[category].append({
                        'name': etf_name,
                        'weight': weights[etf_name]
                    })
        
        return {
            'category_weights': category_weights,
            'etfs_by_category': etf_by_category
        }
    
    def _format_detailed_etf_data(self, etf_data: Dict) -> str:
        """상세한 ETF 데이터 포맷팅"""
        if not etf_data:
            return "ETF 데이터 없음"
        
        formatted = []
        for category, etfs in etf_data.items():
            formatted.append(f"\n**{category}** ({len(etfs)}개 ETF):")
            for i, (etf_name, data) in enumerate(etfs.items()):
                if i < 5:  # 각 카테고리에서 5개까지만 표시
                    if 'returns' in data and data['returns'] is not None and len(data['returns']) > 0:
                        annual_return = data['returns'].mean() * 252 * 100
                        formatted.append(f"  - {etf_name}: 연환산 {annual_return:.1f}%")
                    else:
                        formatted.append(f"  - {etf_name}: 데이터 수집 중")
        
        return "\n".join(formatted)
    
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
    
    def _get_comprehensive_default_analysis(self, macro_data: Dict, etf_data: Dict, user_profile: Dict) -> Dict:
        """기본 종합 분석 (다중 ETF)"""
        age = user_profile.get('age', 30)
        risk_tolerance = user_profile.get('risk_tolerance', '위험중립형')
        
        # 다중 ETF 기본 포트폴리오 생성
        default_portfolio = self._generate_multi_etf_portfolio(etf_data, user_profile)
        
        return {
            'analysis': {
                'macro_analysis': "현재 한국 경제는 GDP 성장률 3.2%, 인플레이션 2.1% 수준으로 안정적인 성장세를 보이고 있습니다. 각 자산군 내에서도 세밀한 분산투자가 필요한 시점입니다.",
                'market_trends': "ETF 시장에서는 단순한 자산군별 배분을 넘어서 각 자산군 내에서도 다양한 ETF를 조합하는 정교한 분산투자 전략이 주목받고 있습니다.",
                'investment_strategy': f"{age}세 {risk_tolerance} 투자자에게는 8-12개 ETF를 활용한 다중 분산투자가 적합합니다. 자산군 내외 이중 분산효과를 통해 리스크를 최소화하면서 안정적인 수익을 추구합니다.",
                'risk_factors': "개별 ETF 간 상관관계 변화와 시장 집중도 리스크에 주의가 필요하며, 정기적인 리밸런싱을 통한 분산효과 유지가 중요합니다."
            },
            'portfolio': {
                'weights': default_portfolio['portfolio'],
                'allocation_reasoning': default_portfolio['allocation_reasoning'],
                'diversification_strategy': default_portfolio['diversification_strategy'],
                'etf_count': len(default_portfolio['portfolio']),
                'category_distribution': self._analyze_category_distribution(default_portfolio['portfolio'], etf_data)
            },
            'source': 'default_multi_etf_analysis'
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
    
    def _format_user_profile(self, user_profile: Dict) -> str:
        """사용자 프로필 포맷팅"""
        return f"""
나이: {user_profile.get('age', 30)}세
투자성향: {user_profile.get('risk_tolerance', '위험중립형')}
투자기간: {user_profile.get('investment_period', 20)}년
현재 자산: {user_profile.get('current_assets', 0):,}원
월 납입액: {user_profile.get('monthly_contribution', 0):,}원
"""
