import requests
import json
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

class HyperCLOVAXTuningTester:
    def __init__(self, model_id, api_key):
        """
        튜닝된 HyperCLOVA X 모델 테스터 초기화
        
        Args:
            model_id (str): 튜닝 완료된 모델 ID (예: vpcxuy1b)
            api_key (str): CLOVA Studio API 키
        """
        self.api_key = os.getenv('CLOVA_API_KEY')
        if not self.api_key:
            raise ValueError("CLOVA_API_KEY가 필요합니다.")
        
        self.model_id = model_id
        self.base_url = "https://clovastudio.stream.ntruss.com"
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

    def test_model(self, prompt, max_tokens=500, temperature=0.3):
        """
        튜닝된 모델로 텍스트 생성 테스트
        
        Args:
            prompt (str): 입력 프롬프트
            max_tokens (int): 최대 토큰 수
            temperature (float): 생성 다양성 조절 (0.0-1.0)
        
        Returns:
            str: 모델 응답 또는 오류 메시지
        """
        url = f"{self.base_url}/testapp/v1/chat-completions/{self.model_id}"
        
        payload = {
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "topP": 0.8,
            "topK": 0,
            "maxTokens": max_tokens,
            "temperature": temperature,
            "repeatPenalty": 1.0,
            "stopBefore": [],
            "includeAiFilters": True
        }
        
        try:
            print(f"모델 ID: {self.model_id}")
            print(f"요청 중...")
            
            response = requests.post(url, headers=self.headers, json=payload)
            
            print(f"응답 상태: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                if 'result' in result and 'message' in result['result']:
                    return result['result']['message']['content']
                else:
                    return f"응답 구조 오류: {result}"
            else:
                return f"API 오류 ({response.status_code}): {response.text}"
                
        except requests.exceptions.RequestException as e:
            return f"네트워크 오류: {e}"

    def batch_test(self, test_cases):
        """
        여러 테스트 케이스를 일괄 처리
        
        Args:
            test_cases (list): 테스트할 프롬프트 리스트
        
        Returns:
            list: 각 테스트 결과
        """
        results = []
        
        for i, prompt in enumerate(test_cases, 1):
            print(f"\n=== 테스트 {i}/{len(test_cases)} ===")
            print(f"입력: {prompt[:100]}...")
            
            result = self.test_model(prompt)
            results.append({
                'test_number': i,
                'input': prompt,
                'output': result,
                'timestamp': datetime.now().isoformat()
            })
            
            print(f"출력: {result}")
        
        return results

def create_etf_test_cases():
    """
    ETF 시계열 예측 테스트 케이스 생성
    """
    test_cases = [
        # 테스트 케이스 1: 기본 ETF 예측
        """다음은 310970 ETF의 정규화된 가격 시계열 데이터입니다.
과거 30일간의 데이터를 바탕으로 향후 10일의 가격을 예측하세요.

입력 데이터: 0.647059 0.500000 0.536765 0.463235 0.647059 0.794118 0.808824 0.838235 0.713235 1.000000 0.823529 0.757353 0.485294 0.544118 0.397059 0.338235 0.536765 0.654412 0.360294 0.397059 0.308824 0.448529 0.235294 0.191176 0.301471 0.058824 0.279412 0.426471 0.139706 0.000000

예상 출력:""",

        # 테스트 케이스 2: 다른 패턴의 데이터
        """다음은 069500 ETF의 정규화된 가격 시계열 데이터입니다.
과거 30일간의 데이터를 바탕으로 향후 10일의 가격을 예측하세요.

입력 데이터: 0.800000 0.750000 0.700000 0.650000 0.600000 0.550000 0.500000 0.450000 0.400000 0.350000 0.300000 0.250000 0.200000 0.150000 0.100000 0.050000 0.000000 0.100000 0.200000 0.300000 0.400000 0.500000 0.600000 0.700000 0.800000 0.900000 0.950000 0.975000 0.990000 1.000000

예상 출력:""",

        # 테스트 케이스 3: 상승 트렌드 데이터
        """다음은 232080 ETF의 정규화된 가격 시계열 데이터입니다.
과거 30일간의 데이터를 바탕으로 향후 10일의 가격을 예측하세요.

입력 데이터: 0.100000 0.150000 0.200000 0.250000 0.300000 0.350000 0.400000 0.450000 0.500000 0.550000 0.600000 0.650000 0.700000 0.750000 0.800000 0.850000 0.900000 0.920000 0.940000 0.960000 0.970000 0.980000 0.985000 0.990000 0.992000 0.994000 0.996000 0.998000 0.999000 1.000000

예상 출력:""",

        # 테스트 케이스 4: 변동성이 큰 데이터
        """다음은 114260 ETF의 정규화된 가격 시계열 데이터입니다.
과거 30일간의 데이터를 바탕으로 향후 10일의 가격을 예측하세요.

입력 데이터: 0.500000 0.800000 0.200000 0.900000 0.100000 0.700000 0.300000 0.600000 0.400000 1.000000 0.000000 0.750000 0.250000 0.850000 0.150000 0.950000 0.050000 0.650000 0.350000 0.550000 0.450000 0.950000 0.050000 0.800000 0.200000 0.700000 0.300000 0.900000 0.100000 0.600000

예상 출력:"""
    ]
    
    return test_cases

def main():
    """
    메인 테스트 실행 함수
    """
    print("=== HyperCLOVA X 튜닝 모델 테스트 ===\n")
    
    # 모델 정보 설정
    model_id = "vpcxuy1b"  # 실제 튜닝 완료된 모델 ID
    api_key = "YOUR_CLOVA_API_KEY"  # 실제 API 키로 교체
    
    # 테스터 초기화
    try:
        tester = HyperCLOVAXTuningTester(model_id, api_key)
        print(f"✅ 테스터 초기화 완료")
        print(f"모델 ID: {model_id}")
    except Exception as e:
        print(f"❌ 테스터 초기화 실패: {e}")
        return
    
    # 테스트 케이스 생성
    test_cases = create_etf_test_cases()
    print(f"📝 {len(test_cases)}개 테스트 케이스 준비 완료\n")
    
    # 일괄 테스트 실행
    results = tester.batch_test(test_cases)
    
    # 결과 저장
    output_file = f'tuning_test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n=== 테스트 완료 ===")
    print(f"결과 저장: {output_file}")
    print(f"총 테스트: {len(results)}개")
    
    # 성공/실패 통계
    success_count = sum(1 for r in results if not r['output'].startswith('API 오류') and not r['output'].startswith('네트워크 오류'))
    print(f"성공: {success_count}/{len(results)}개")
    
    return results

# 개별 테스트용 함수
def quick_test(model_id, api_key, prompt):
    """
    빠른 단일 테스트
    """
    tester = HyperCLOVAXTuningTester(model_id, api_key)
    result = tester.test_model(prompt)
    print(f"입력: {prompt}")
    print(f"출력: {result}")
    return result

if __name__ == '__main__':
    # 실제 API 키를 여기에 입력하세요
    API_KEY = "YOUR_ACTUAL_CLOVA_API_KEY"
    MODEL_ID = "tuning-6571-250628-172946-4dc8f"
    
    # 전체 테스트 실행
    # main()
    
    # 또는 빠른 단일 테스트
    test_prompt = """다음은 310970 ETF의 정규화된 가격 시계열 데이터입니다.
과거 30일간의 데이터를 바탕으로 향후 10일의 가격을 예측하세요.

입력 데이터: 0.647059 0.500000 0.536765 0.463235 0.647059 0.794118 0.808824 0.838235 0.713235 1.000000 0.823529 0.757353 0.485294 0.544118 0.397059 0.338235 0.536765 0.654412 0.360294 0.397059 0.308824 0.448529 0.235294 0.191176 0.301471 0.058824 0.279412 0.426471 0.139706 0.000000

예상 출력:"""
    
    quick_test(MODEL_ID, API_KEY, test_prompt)
