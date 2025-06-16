# fine_tuning_lifecycle_generator.py
import json
import pandas as pd
import numpy as np
from langchain_naver import ChatClovaX
from datetime import datetime, timedelta
import random
from config import Config

class LifecycleFineTuningDataGenerator:
    def __init__(self):
        """생애주기 파인튜닝 데이터 생성기 초기화"""
        self.client = ChatClovaX(
            api_key=Config.HYPERCLOVA_X_API_KEY,
            model='hcx-005',
            max_tokens=3000,
            temperature=0.7
        )
        
    def generate_lifecycle_scenarios(self, num_scenarios=500):
        """다양한 생애주기 시나리오 생성"""
        scenarios = []
        
        # 연령대별 시나리오 설정
        age_groups = {
            'young': list(range(25, 35)),      # 청년층
            'middle': list(range(35, 50)),     # 중년층
            'pre_retire': list(range(50, 60)), # 은퇴준비층
            'senior': list(range(60, 65))      # 시니어층
        }
        
        # 위험성향 분포
        risk_types = ['안정형', '안정추구형', '위험중립형', '적극투자형']
        
        # 은퇴 목표 연령
        retirement_ages = [60, 62, 65, 67, 70]
        
        for i in range(num_scenarios):
            # 연령대 선택 (가중치 적용)
            age_group = random.choices(
                list(age_groups.keys()), 
                weights=[0.3, 0.4, 0.2, 0.1]  # 중년층에 가중치
            )[0]
            
            age = random.choice(age_groups[age_group])
            
            # 연령대별 자산 규모 설정
            if age_group == 'young':
                current_assets = random.randint(500, 3000) * 10000  # 500만~3천만
                monthly_contribution = random.randint(30, 100) * 10000  # 30만~100만
            elif age_group == 'middle':
                current_assets = random.randint(3000, 8000) * 10000  # 3천만~8천만
                monthly_contribution = random.randint(50, 200) * 10000  # 50만~200만
            elif age_group == 'pre_retire':
                current_assets = random.randint(8000, 20000) * 10000  # 8천만~2억
                monthly_contribution = random.randint(100, 300) * 10000  # 100만~300만
            else:  # senior
                current_assets = random.randint(10000, 30000) * 10000  # 1억~3억
                monthly_contribution = random.randint(50, 150) * 10000  # 50만~150만
            
            # 은퇴 목표 연령 (현재 나이보다 큰 값)
            available_retirement_ages = [r for r in retirement_ages if r > age]
            retirement_age = random.choice(available_retirement_ages) if available_retirement_ages else 65
            
            scenario = {
                'age': age,
                'age_group': age_group,
                'current_assets': current_assets,
                'monthly_contribution': monthly_contribution,
                'retirement_age': retirement_age,
                'risk_tolerance': random.choice(risk_types),
                'investment_period': retirement_age - age,
                'life_stage': self._determine_life_stage(age, retirement_age),
                'financial_goals': self._generate_financial_goals(age_group),
                'market_assumptions': self._generate_market_assumptions()
            }
            scenarios.append(scenario)
            
        return scenarios
    
    def _determine_life_stage(self, age, retirement_age):
        """생애주기 단계 결정"""
        years_to_retirement = retirement_age - age
        
        if years_to_retirement > 20:
            return "자산축적기"
        elif years_to_retirement > 10:
            return "자산증식기"
        elif years_to_retirement > 5:
            return "은퇴준비기"
        else:
            return "은퇴직전기"
    
    def _generate_financial_goals(self, age_group):
        """연령대별 재정 목표 생성"""
        goals = {
            'young': [
                "내집마련 자금 준비",
                "결혼자금 마련",
                "퇴직연금 기반 구축",
                "비상자금 확보"
            ],
            'middle': [
                "자녀 교육비 준비",
                "주택 대출 상환",
                "은퇴자금 본격 축적",
                "부모님 부양 준비"
            ],
            'pre_retire': [
                "은퇴자금 목표 달성",
                "의료비 준비",
                "은퇴 후 소득원 확보",
                "상속 계획 수립"
            ],
            'senior': [
                "안정적 노후 소득 확보",
                "의료비 대비",
                "자산 보전",
                "상속세 절약"
            ]
        }
        return random.choice(goals[age_group])
    
    def _generate_market_assumptions(self):
        """시장 가정 생성"""
        return {
            'expected_return_stock': round(random.uniform(6.0, 9.0), 1),
            'expected_return_bond': round(random.uniform(3.0, 5.0), 1),
            'inflation_rate': round(random.uniform(2.0, 3.5), 1),
            'volatility_stock': round(random.uniform(15.0, 25.0), 1),
            'volatility_bond': round(random.uniform(3.0, 8.0), 1)
        }
    
    def generate_training_data(self, scenarios, prompt_template):
        """생애주기 훈련 데이터 생성"""
        training_data = []
        
        for i, scenario in enumerate(scenarios):
            try:
                # 프롬프트 생성
                user_input = prompt_template.format(**scenario)
                
                # AI 응답 생성
                response = self.client.invoke(user_input)
                ai_output = response.content
                
                # 훈련 데이터 포맷팅 (OpenAI 파인튜닝 형식)
                training_sample = {
                    'messages': [
                        {
                            'role': 'user',
                            'content': user_input
                        },
                        {
                            'role': 'assistant', 
                            'content': ai_output
                        }
                    ]
                }
                
                # CSV용 데이터 포맷팅
                csv_sample = {
                    'C_ID': i // 10,  # 10개씩 그룹화
                    'T_ID': i % 10,
                    'age': scenario['age'],
                    'age_group': scenario['age_group'],
                    'life_stage': scenario['life_stage'],
                    'current_assets': scenario['current_assets'],
                    'monthly_contribution': scenario['monthly_contribution'],
                    'retirement_age': scenario['retirement_age'],
                    'risk_tolerance': scenario['risk_tolerance'],
                    'financial_goals': scenario['financial_goals'],
                    'Text': user_input,
                    'Completion': ai_output
                }
                
                training_data.append({
                    'jsonl': training_sample,
                    'csv': csv_sample
                })
                
                print(f"Generated lifecycle sample {i+1}/{len(scenarios)}")
                
                # API 호출 제한을 위한 딜레이
                if i % 10 == 0 and i > 0:
                    import time
                    time.sleep(1)
                
            except Exception as e:
                print(f"Error generating sample {i}: {e}")
                continue
                
        return training_data
    
    def save_to_csv(self, training_data, filename):
        """CSV 형식으로 저장"""
        csv_data = [item['csv'] for item in training_data]
        df = pd.DataFrame(csv_data)
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"CSV data saved to {filename}")
        
    def save_to_jsonl(self, training_data, filename):
        """JSONL 형식으로 저장 (OpenAI 파인튜닝용)"""
        with open(filename, 'w', encoding='utf-8') as f:
            for item in training_data:
                f.write(json.dumps(item['jsonl'], ensure_ascii=False) + '\n')
        print(f"JSONL data saved to {filename}")
    
    def save_statistics(self, scenarios, filename):
        """데이터셋 통계 저장"""
        stats = {
            'total_samples': len(scenarios),
            'age_distribution': {},
            'life_stage_distribution': {},
            'risk_tolerance_distribution': {},
            'asset_range': {
                'min': min(s['current_assets'] for s in scenarios),
                'max': max(s['current_assets'] for s in scenarios),
                'avg': sum(s['current_assets'] for s in scenarios) / len(scenarios)
            }
        }
        
        # 연령대 분포
        for scenario in scenarios:
            age_group = scenario['age_group']
            stats['age_distribution'][age_group] = stats['age_distribution'].get(age_group, 0) + 1
        
        # 생애주기 분포
        for scenario in scenarios:
            life_stage = scenario['life_stage']
            stats['life_stage_distribution'][life_stage] = stats['life_stage_distribution'].get(life_stage, 0) + 1
        
        # 위험성향 분포
        for scenario in scenarios:
            risk = scenario['risk_tolerance']
            stats['risk_tolerance_distribution'][risk] = stats['risk_tolerance_distribution'].get(risk, 0) + 1
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"Statistics saved to {filename}")

def generate_lifecycle_fine_tuning_dataset():
    """생애주기 파인튜닝 데이터셋 생성 메인 함수"""
    generator = LifecycleFineTuningDataGenerator()
    
    # lifecycle_strategy_prompt 정의
    lifecycle_strategy_prompt = """생애주기 기반 퇴직연금 설계 전문가로서 고객 맞춤형 투자 전략을 수립해주세요.

**고객 정보:**
- 나이: {age}세
- 생애주기 단계: {life_stage}
- 현재 자산: {current_assets:,}원
- 월 납입액: {monthly_contribution:,}원
- 은퇴 목표: {retirement_age}세
- 위험성향: {risk_tolerance}
- 주요 재정목표: {financial_goals}
- 투자기간: {investment_period}년

**시장 가정:**
- 주식 기대수익률: {market_assumptions[expected_return_stock]}%
- 채권 기대수익률: {market_assumptions[expected_return_bond]}%
- 인플레이션율: {market_assumptions[inflation_rate]}%

**요청사항:**
1. 생애주기 단계 분석 및 특징
2. 목표 수익률과 필요 수익률 계산
3. 단계별 자산배분 전략 (5년 단위)
4. 구체적 ETF 포트폴리오 구성 (국내 ETF 중심)
5. 리스크 관리 방안
6. 정기 리뷰 및 조정 계획

실현 가능하고 구체적인 전략을 제시해주세요."""
    
    # 시나리오 생성
    print("Generating lifecycle scenarios...")
    scenarios = generator.generate_lifecycle_scenarios(300)  # 300개 샘플 생성
    
    # 훈련 데이터 생성
    print("Generating training data...")
    training_data = generator.generate_training_data(scenarios, lifecycle_strategy_prompt)
    
    # 파일 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"pension_lifecycle_training_data_{timestamp}.csv"
    jsonl_filename = f"pension_lifecycle_training_data_{timestamp}.jsonl"
    stats_filename = f"pension_lifecycle_statistics_{timestamp}.json"
    
    generator.save_to_csv(training_data, csv_filename)
    generator.save_to_jsonl(training_data, jsonl_filename)
    generator.save_statistics(scenarios, stats_filename)
    
    # 결과 요약
    print(f"\n{'='*60}")
    print("📊 생애주기 파인튜닝 데이터셋 생성 완료")
    print(f"{'='*60}")
    print(f"✅ 총 생성 샘플: {len(training_data)}개")
    print(f"📁 CSV 파일: {csv_filename}")
    print(f"📁 JSONL 파일: {jsonl_filename}")
    print(f"📊 통계 파일: {stats_filename}")
    
    # 간단한 통계 출력
    age_groups = {}
    for scenario in scenarios:
        age_group = scenario['age_group']
        age_groups[age_group] = age_groups.get(age_group, 0) + 1
    
    print(f"\n📈 연령대별 분포:")
    for age_group, count in age_groups.items():
        print(f"  {age_group}: {count}개 ({count/len(scenarios)*100:.1f}%)")
    
    return training_data, csv_filename, jsonl_filename

# Jupyter Notebook에서 실행할 수 있는 간단한 버전
def quick_lifecycle_test():
    """빠른 테스트용 함수 (5개 샘플만 생성)"""
    generator = LifecycleFineTuningDataGenerator()
    
    lifecycle_strategy_prompt = """생애주기 기반 퇴직연금 설계 전문가로서 고객 맞춤형 투자 전략을 수립해주세요.

**고객 정보:**
- 나이: {age}세 ({life_stage})
- 현재 자산: {current_assets:,}원
- 월 납입액: {monthly_contribution:,}원
- 은퇴 목표: {retirement_age}세
- 위험성향: {risk_tolerance}

**요청사항:**
1. 생애주기 단계별 투자 전략
2. 구체적 ETF 포트폴리오 구성
3. 목표 수익률 및 리스크 관리 방안

실현 가능한 전략을 제시해주세요."""
    
    scenarios = generator.generate_lifecycle_scenarios(5)
    training_data = generator.generate_training_data(scenarios, lifecycle_strategy_prompt)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"test_lifecycle_{timestamp}.csv"
    jsonl_filename = f"test_lifecycle_{timestamp}.jsonl"
    
    generator.save_to_csv(training_data, csv_filename)
    generator.save_to_jsonl(training_data, jsonl_filename)
    
    return training_data

if __name__ == "__main__":
    # 전체 데이터셋 생성
    training_data, csv_file, jsonl_file = generate_lifecycle_fine_tuning_dataset()
    print(f"\n🎉 생성 완료! 파일: {csv_file}, {jsonl_file}")
