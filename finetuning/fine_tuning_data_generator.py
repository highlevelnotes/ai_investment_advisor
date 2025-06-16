# fine_tuning_data_generator.py
import json
import pandas as pd
import numpy as np
from langchain_naver import ChatClovaX
from datetime import datetime, timedelta
import random
from config import Config

class FineTuningDataGenerator:
    def __init__(self):
        """파인튜닝 데이터 생성기 초기화"""
        self.client = ChatClovaX(
            api_key=Config.HYPERCLOVA_X_API_KEY,
            model='hcx-005',
            max_tokens=3000,
            temperature=0.7
        )
        
    def generate_market_scenarios(self, num_scenarios=200):
        """다양한 시장 시나리오 생성"""
        scenarios = []
        
        # 경제지표 범위 설정
        gdp_range = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        inflation_range = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
        interest_rate_range = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
        
        # 사용자 프로필 범위
        age_range = list(range(25, 66, 5))
        risk_types = ['안정형', '안정추구형', '위험중립형', '적극투자형']
        investment_periods = [5, 10, 15, 20, 25, 30]
        
        for i in range(num_scenarios):
            scenario = {
                'gdp': random.choice(gdp_range),
                'inflation': random.choice(inflation_range),
                'interest_rate': random.choice(interest_rate_range),
                'age': random.choice(age_range),
                'risk_tolerance': random.choice(risk_types),
                'investment_period': random.choice(investment_periods),
                'etf_performance_data': self._generate_etf_performance()
            }
            scenarios.append(scenario)
            
        return scenarios
    
    def _generate_etf_performance(self):
        """ETF 성과 데이터 생성"""
        etf_categories = {
            '국내주식형': ['KODEX 200', 'TIGER 200', 'KODEX 반도체'],
            '국내채권형': ['KODEX 국고채10년', 'TIGER 단기통안채'],
            '국내섹터': ['KODEX 2차전지산업', 'KODEX 바이오'],
            '국내대안': ['KODEX 골드선물(H)', 'TIGER 국내리츠']
        }
        
        performance_data = {}
        for category, etfs in etf_categories.items():
            category_data = {}
            for etf in etfs:
                category_data[etf] = {
                    'return': round(random.uniform(-10, 20), 2),
                    'volatility': round(random.uniform(5, 25), 2)
                }
            performance_data[category] = category_data
            
        return performance_data
    
    def generate_training_data(self, scenarios, prompt_template, data_type):
        """훈련 데이터 생성"""
        training_data = []
        
        for i, scenario in enumerate(scenarios):
            try:
                # 프롬프트 생성
                if data_type == 'market_analysis':
                    user_input = prompt_template.format(**scenario)
                elif data_type == 'risk_scenario':
                    scenario['scenario_description'] = self._generate_risk_scenario()
                    scenario['current_allocation'] = self._generate_current_allocation()
                    scenario['market_conditions'] = self._generate_market_conditions()
                    user_input = prompt_template.format(**scenario)
                elif data_type == 'lifecycle_strategy':
                    scenario['current_assets'] = random.randint(1000, 10000) * 10000
                    scenario['monthly_contribution'] = random.randint(30, 200) * 10000
                    scenario['retirement_age'] = random.choice([60, 62, 65, 67])
                    user_input = prompt_template.format(**scenario)
                
                # AI 응답 생성
                response = self.client.invoke(user_input)
                ai_output = response.content
                
                # 데이터 포맷팅
                training_sample = {
                    'C_ID': i // 10,  # 10개씩 그룹화
                    'T_ID': i % 10,
                    'Text': user_input,
                    'Completion': ai_output
                }
                
                training_data.append(training_sample)
                
                print(f"Generated sample {i+1}/{len(scenarios)}")
                
            except Exception as e:
                print(f"Error generating sample {i}: {e}")
                continue
                
        return training_data
    
    def _generate_risk_scenario(self):
        """리스크 시나리오 생성"""
        scenarios = [
            "금리 급등으로 인한 채권 가격 하락",
            "반도체 업종 조정으로 인한 국내주식 하락",
            "글로벌 인플레이션 심화",
            "지정학적 리스크 확대",
            "부동산 시장 급락",
            "원유가격 급등"
        ]
        return random.choice(scenarios)
    
    def _generate_current_allocation(self):
        """현재 자산배분 생성"""
        allocations = [
            "국내주식 50%, 국내채권 30%, 국내섹터 15%, 국내대안 5%",
            "국내주식 40%, 국내채권 40%, 국내섹터 10%, 국내대안 10%",
            "국내주식 60%, 국내채권 25%, 국내섹터 10%, 국내대안 5%"
        ]
        return random.choice(allocations)
    
    def _generate_market_conditions(self):
        """시장 상황 생성"""
        conditions = [
            "변동성 확대, 거래량 증가",
            "안정적 추세, 낮은 변동성",
            "불확실성 증가, 방향성 부재",
            "강세장 지속, 높은 거래량"
        ]
        return random.choice(conditions)
    
    def save_to_csv(self, training_data, filename):
        """CSV 형식으로 저장"""
        df = pd.DataFrame(training_data)
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"Data saved to {filename}")
        
    def save_to_jsonl(self, training_data, filename):
        """JSONL 형식으로 저장"""
        with open(filename, 'w', encoding='utf-8') as f:
            for item in training_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Data saved to {filename}")

# 데이터 생성 실행 코드
def generate_fine_tuning_dataset():
    """파인튜닝 데이터셋 생성 메인 함수"""
    generator = FineTuningDataGenerator()
    
    # 프롬프트 템플릿 정의
    market_analysis_prompt = """
당신은 전문 퇴직연금 포트폴리오 매니저입니다. 주어진 경제 상황과 사용자 프로필을 바탕으로 국내 ETF 중심의 포트폴리오를 추천해주세요.

**입력 조건:**
- 경제지표: GDP 성장률 {gdp}%, 인플레이션 {inflation}%, 기준금리 {interest_rate}%
- 사용자: {age}세, {risk_tolerance}, 투자기간 {investment_period}년
- ETF 성과: {etf_performance_data}

**출력 요구사항:**
1. 자산배분 비율 (국내주식/국내채권/국내섹터/국내대안)
2. 구체적 ETF 3-5개 추천
3. 추천 근거 (경제 상황 반영)
4. 예상 수익률과 리스크
5. 모니터링 포인트

JSON 형식으로 응답해주세요.
"""
    
    # 시나리오 생성
    print("Generating market scenarios...")
    scenarios = generator.generate_market_scenarios(500)
    
    # 훈련 데이터 생성
    print("Generating training data...")
    training_data = generator.generate_training_data(
        scenarios, 
        market_analysis_prompt, 
        'market_analysis'
    )
    
    # 데이터 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"pension_portfolio_training_data_{timestamp}.csv"
    jsonl_filename = f"pension_portfolio_training_data_{timestamp}.jsonl"
    
    generator.save_to_csv(training_data, csv_filename)
    generator.save_to_jsonl(training_data, jsonl_filename)
    
    return training_data, csv_filename, jsonl_filename

if __name__ == "__main__":
    training_data, csv_file, jsonl_file = generate_fine_tuning_dataset()
    print(f"Generated {len(training_data)} training samples")
    print(f"Files saved: {csv_file}, {jsonl_file}")


market_analysis_prompt = """
당신은 전문 퇴직연금 포트폴리오 매니저입니다. 주어진 경제 상황과 사용자 프로필을 바탕으로 국내 ETF 중심의 포트폴리오를 추천해주세요.

**입력 조건:**
- 경제지표: GDP 성장률 {gdp}%, 인플레이션 {inflation}%, 기준금리 {interest_rate}%
- 사용자: {age}세, {risk_tolerance}, 투자기간 {investment_period}년
- ETF 성과: {etf_performance_data}

**출력 요구사항:**
1. 자산배분 비율 (국내주식/국내채권/국내섹터/국내대안)
2. 구체적 ETF 3-5개 추천
3. 추천 근거 (경제 상황 반영)
4. 예상 수익률과 리스크
5. 모니터링 포인트

JSON 형식으로 응답해주세요.
"""

risk_scenario_prompt = """
퇴직연금 포트폴리오 리스크 관리 전문가로서 다음 시나리오에 대한 대응 전략을 제시해주세요.

**시나리오:** {scenario_description}
**현재 포트폴리오:** {current_allocation}
**시장 상황:** {market_conditions}

**분석 요청사항:**
1. 시나리오 발생 시 예상 포트폴리오 영향도
2. 리스크 완화를 위한 구체적 조치
3. 자산배분 조정 방안
4. 대안 투자 옵션
5. 회복 전략

전문적이고 실행 가능한 조언을 제공해주세요.
"""

lifecycle_strategy_prompt = """
생애주기 기반 퇴직연금 설계 전문가로서 고객 맞춤형 투자 전략을 수립해주세요.

**고객 정보:**
- 나이: {age}세
- 현재 자산: {current_assets}원
- 월 납입액: {monthly_contribution}원
- 은퇴 목표: {retirement_age}세
- 위험성향: {risk_tolerance}

**요청사항:**
1. 생애주기 단계 분석
2. 목표 수익률과 필요 수익률 계산
3. 단계별 자산배분 전략
4. 구체적 ETF 포트폴리오 구성
5. 정기 리뷰 및 조정 계획

실현 가능하고 구체적인 전략을 제시해주세요.
"""
