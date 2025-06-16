# ai_analyzer.py - 완전 재작성
import json
import re
from typing import Dict, Any, Optional
from langchain_naver import ChatClovaX
from config import Config

class AIAnalyzer:
    def __init__(self):
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
    
    def analyze_market_situation(self, macro_data: Dict, etf_data: Dict) -> str:
        """시장 상황 분석 - 단순화된 버전"""
        if not self.available:
            return self._get_detailed_market_analysis(macro_data, etf_data)
        
        try:
            # 매우 간단한 프롬프트로 시작
            prompt = f"""
현재 한국 경제 상황을 분석해주세요.

GDP 성장률: {macro_data.get('GDP', {}).get('current', 3.0)}%
인플레이션: {macro_data.get('CPI', {}).get('current', 2.0)}%
기준금리: {macro_data.get('INTEREST_RATE', {}).get('current', 3.5)}%

다음 4가지로 나누어 분석해주세요:
1. 경제 상황 요약
2. 투자 환경 평가  
3. 주요 기회 요인
4. 리스크 요인

각 항목을 2-3줄로 간단히 설명해주세요.
"""
            
            response = self.client.invoke(prompt)
            
            if response and response.content and len(response.content.strip()) > 50:
                return response.content
            else:
                print("⚠️ AI 응답이 부족합니다. 기본 분석을 제공합니다.")
                return self._get_detailed_market_analysis(macro_data, etf_data)
                
        except Exception as e:
            print(f"❌ AI 시장 분석 실패: {e}")
            return self._get_detailed_market_analysis(macro_data, etf_data)
    
    def generate_portfolio_strategy(self, macro_data: Dict, etf_data: Dict, user_profile: Dict) -> str:
        """포트폴리오 전략 생성"""
        if not self.available:
            return self._get_detailed_portfolio_strategy(user_profile, macro_data)
        
        try:
            age = user_profile.get('age', 30)
            risk_tolerance = user_profile.get('risk_tolerance', '위험중립형')
            
            prompt = f"""
{age}세, {risk_tolerance} 투자자를 위한 국내 ETF 포트폴리오 전략을 제시해주세요.

현재 경제 상황:
- GDP: {macro_data.get('GDP', {}).get('current', 3.0)}%
- 인플레이션: {macro_data.get('CPI', {}).get('current', 2.0)}%

다음 순서로 전략을 제시해주세요:
1. 투자 방향성
2. 자산배분 전략
3. 추천 ETF 종목 3개
4. 리밸런싱 방법

실용적이고 구체적으로 설명해주세요.
"""
            
            response = self.client.invoke(prompt)
            
            if response and response.content and len(response.content.strip()) > 100:
                return response.content
            else:
                return self._get_detailed_portfolio_strategy(user_profile, macro_data)
                
        except Exception as e:
            print(f"❌ AI 포트폴리오 전략 생성 실패: {e}")
            return self._get_detailed_portfolio_strategy(user_profile, macro_data)
    
    def generate_ai_portfolio_weights(self, macro_data: Dict, etf_data: Dict, user_profile: Dict) -> Dict:
        """AI 기반 포트폴리오 가중치 생성"""
        if not self.available:
            return self._get_smart_default_weights(user_profile, macro_data, etf_data)
        
        try:
            age = user_profile.get('age', 30)
            risk_tolerance = user_profile.get('risk_tolerance', '위험중립형')
            
            # ETF 목록 생성
            etf_list = []
            for category, etfs in etf_data.items():
                for name in list(etfs.keys())[:3]:  # 각 카테고리에서 3개씩
                    etf_list.append(name)
            
            prompt = f"""
{age}세, {risk_tolerance} 투자자를 위한 ETF 포트폴리오 비중을 정해주세요.

투자 가능한 ETF:
{', '.join(etf_list[:10])}

경제 상황을 고려하여 각 ETF의 투자 비중을 %로 제시해주세요.
모든 비중의 합은 100%가 되어야 합니다.

다음 형식으로만 답변해주세요:
ETF명: 비중%
ETF명: 비중%
...
"""
            
            response = self.client.invoke(prompt)
            
            if response and response.content:
                weights = self._parse_weights_from_response(response.content, etf_list)
                if weights:
                    return weights
            
            return self._get_smart_default_weights(user_profile, macro_data, etf_data)
            
        except Exception as e:
            print(f"❌ AI 포트폴리오 가중치 생성 실패: {e}")
            return self._get_smart_default_weights(user_profile, macro_data, etf_data)
    
    def _parse_weights_from_response(self, response_text: str, etf_list: list) -> Dict:
        """AI 응답에서 가중치 파싱"""
        weights = {}
        lines = response_text.split('\n')
        
        for line in lines:
            for etf_name in etf_list:
                if etf_name in line:
                    # 숫자 추출
                    numbers = re.findall(r'\d+\.?\d*', line)
                    if numbers:
                        try:
                            weight = float(numbers[0]) / 100.0
                            if 0 <= weight <= 1:
                                weights[etf_name] = weight
                        except:
                            continue
        
        # 가중치 정규화
        total_weight = sum(weights.values())
        if total_weight > 0 and len(weights) >= 3:
            weights = {k: v/total_weight for k, v in weights.items()}
            return weights
        
        return {}
    
    def _get_detailed_market_analysis(self, macro_data: Dict, etf_data: Dict) -> str:
        """상세한 기본 시장 분석"""
        gdp = macro_data.get('GDP', {}).get('current', 3.0)
        inflation = macro_data.get('CPI', {}).get('current', 2.0)
        interest_rate = macro_data.get('INTEREST_RATE', {}).get('current', 3.5)
        
        return f"""
## 📊 현재 시장 상황 분석

### 1. 경제 상황 요약
한국 경제는 GDP 성장률 {gdp}%로 {self._get_growth_assessment(gdp)} 성장세를 보이고 있습니다. 
인플레이션 {inflation}%는 {self._get_inflation_assessment(inflation)} 수준이며, 
기준금리 {interest_rate}%는 통화정책의 {self._get_rate_assessment(interest_rate)} 기조를 반영합니다.

### 2. 투자 환경 평가
현재 투자 환경은 {self._get_investment_environment(gdp, inflation, interest_rate)}입니다.
ETF 시장에서는 국내주식형과 채권형 ETF 간의 균형잡힌 접근이 필요한 시점입니다.

### 3. 주요 기회 요인
- 국내 2차전지 및 반도체 산업의 구조적 성장 지속
- 금리 안정화에 따른 채권 ETF 매력도 증가
- 국내 리츠 시장의 안정적 배당 수익 기대

### 4. 리스크 요인
- 글로벌 경제 불확실성에 따른 변동성 확대 가능성
- 금리 변동에 따른 자산군별 상대적 매력도 변화
- 지정학적 리스크가 국내 시장에 미치는 영향
"""
    
    def _get_detailed_portfolio_strategy(self, user_profile: Dict, macro_data: Dict) -> str:
        """상세한 기본 포트폴리오 전략"""
        age = user_profile.get('age', 30)
        risk_tolerance = user_profile.get('risk_tolerance', '위험중립형')
        
        if age < 40:
            stock_ratio = "50-60%"
            bond_ratio = "25-35%"
            strategy_focus = "성장 중심"
        elif age < 55:
            stock_ratio = "40-50%"
            bond_ratio = "35-45%"
            strategy_focus = "균형 중심"
        else:
            stock_ratio = "30-40%"
            bond_ratio = "45-55%"
            strategy_focus = "안정 중심"
        
        return f"""
## 🎯 맞춤형 포트폴리오 전략

### 1. 투자 방향성
{age}세 {risk_tolerance} 투자자에게는 **{strategy_focus}** 전략이 적합합니다.
현재 경제 상황을 고려할 때, 국내 ETF 중심의 분산투자를 통해 
안정성과 수익성의 균형을 추구하는 것이 바람직합니다.

### 2. 자산배분 전략
- **국내주식형 ETF**: {stock_ratio} (KODEX 200, TIGER 200 중심)
- **국내채권형 ETF**: {bond_ratio} (국고채 10년, 단기채권 혼합)
- **섹터/테마 ETF**: 10-15% (2차전지, 바이오 등 성장 섹터)
- **대안투자 ETF**: 5-10% (금, 국내 리츠 등 분산효과)

### 3. 추천 ETF 종목
1. **KODEX 200 (069500)**: 국내 대표지수 추종, 안정성 확보
2. **KODEX 국고채10년 (148070)**: 금리 안정화 수혜, 안전자산 역할
3. **KODEX 2차전지산업 (117700)**: 국내 성장 산업, 장기 투자 매력

### 4. 리밸런싱 방법
- **주기**: 분기별 (3개월마다) 포트폴리오 점검
- **기준**: 목표 비중에서 ±5% 이상 이탈시 조정
- **시장 상황**: 급격한 변동성 확대시 임시 조정 고려
"""
    
    def _get_smart_default_weights(self, user_profile: Dict, macro_data: Dict, etf_data: Dict) -> Dict:
        """스마트 기본 가중치 (경제지표 반영)"""
        age = user_profile.get('age', 30)
        risk_tolerance = user_profile.get('risk_tolerance', '위험중립형')
        
        # 경제지표 기반 조정
        gdp = macro_data.get('GDP', {}).get('current', 3.0)
        inflation = macro_data.get('CPI', {}).get('current', 2.0)
        
        # 기본 비중 설정
        if age < 40:
            base_stock = 0.55
            base_bond = 0.30
        elif age < 55:
            base_stock = 0.45
            base_bond = 0.40
        else:
            base_stock = 0.35
            base_bond = 0.50
        
        # 경제지표 조정
        if gdp > 3.5:  # 고성장
            base_stock += 0.05
        elif gdp < 2.0:  # 저성장
            base_stock -= 0.05
            
        if inflation > 3.0:  # 고인플레이션
            base_bond -= 0.05
        
        # 위험성향 조정
        risk_adjustments = {
            '안정형': -0.1,
            '안정추구형': -0.05,
            '위험중립형': 0.0,
            '적극투자형': 0.1
        }
        base_stock += risk_adjustments.get(risk_tolerance, 0)
        
        # ETF별 가중치 배분
        weights = {}
        
        # 주요 ETF 선택
        main_etfs = [
            ('KODEX 200', base_stock * 0.6),
            ('TIGER 200', base_stock * 0.4),
            ('KODEX 국고채10년', base_bond * 0.6),
            ('KODEX 단기채권', base_bond * 0.4),
            ('KODEX 2차전지산업', 0.08),
            ('KODEX 골드선물(H)', 0.07)
        ]
        
        # 실제 존재하는 ETF만 선택
        for etf_name, target_weight in main_etfs:
            for category, etfs in etf_data.items():
                if etf_name in etfs:
                    weights[etf_name] = target_weight
                    break
        
        # 가중치 정규화
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        return weights
    
    def _get_growth_assessment(self, gdp):
        if gdp > 3.5: return "견조한"
        elif gdp > 2.5: return "안정적인"
        else: return "둔화된"
    
    def _get_inflation_assessment(self, inflation):
        if inflation > 3.0: return "높은"
        elif inflation > 1.5: return "적정"
        else: return "낮은"
    
    def _get_rate_assessment(self, rate):
        if rate > 4.0: return "긴축적"
        elif rate > 2.5: return "중립적"
        else: return "완화적"
    
    def _get_investment_environment(self, gdp, inflation, rate):
        if gdp > 3.0 and inflation < 3.0:
            return "양호한 투자 환경"
        elif gdp < 2.0 or inflation > 4.0:
            return "신중한 접근이 필요한 환경"
        else:
            return "혼재된 신호를 보이는 환경"
