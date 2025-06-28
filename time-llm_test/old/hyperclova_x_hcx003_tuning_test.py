import os
import json
import time
import requests
from datetime import datetime
from dotenv import load_dotenv

class HyperCLOVAXTuningClient:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('CLOVA_API_KEY')
        
        if not self.api_key:
            raise ValueError("CLOVA_API_KEY가 설정되지 않았습니다.")
        
        self.base_url = "https://clovastudio.stream.ntruss.com"
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

    def test_api_connection(self):
        """API 연결 테스트"""
        url = f"{self.base_url}/testapp/v1/chat-completions/HCX-003"
        
        payload = {
            'messages': [
                {'role': 'user', 'content': '연결 테스트'}
            ],
            'maxTokens': 5,
            'temperature': 0.1
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            
            if response.status_code == 200:
                print("✅ API 연결 성공")
                return True
            else:
                print(f"❌ API 연결 실패: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ API 연결 오류: {e}")
            return False

    def load_jsonl_data(self, file_path):
        """JSONL 파일에서 튜닝 데이터 로드"""
        tuning_data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if 'messages' in data and isinstance(data['messages'], list):
                                # 메시지 유효성 검사
                                valid_messages = []
                                for msg in data['messages']:
                                    if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                                        valid_messages.append(msg)
                                
                                if len(valid_messages) >= 2:
                                    tuning_data.append({'messages': valid_messages})
                                    
                        except json.JSONDecodeError as e:
                            print(f"라인 {line_num} 파싱 오류: {e}")
                            continue
            
            print(f"✅ {len(tuning_data)}개 유효한 튜닝 샘플 로드 완료")
            return tuning_data
            
        except FileNotFoundError:
            print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
            return []

    def start_tuning_job_v1(self, tuning_data, job_name=None):
        """튜닝 작업 시작 - 방법 1: 표준 Bearer 토큰 방식"""
        if job_name is None:
            job_name = f"hcx003_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        url = f"{self.base_url}/testapp/v1/api-tools/tuning/tasks"
        
        payload = {
            "name": job_name,
            "baseModel": "HCX-003",
            "description": "ETF 시계열 예측 모델 튜닝",
            "dataset": tuning_data,
            "hyperParameters": {
                "epochs": 3,
                "learningRate": 0.00001,
                "batchSize": 4
            }
        }
        
        return self._make_tuning_request(url, payload, "표준 Bearer 방식")

    def start_tuning_job_v2(self, tuning_data, job_name=None):
        """튜닝 작업 시작 - 방법 2: v2 API Bearer 방식"""
        if job_name is None:
            job_name = f"hcx003_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        url = f"{self.base_url}/tuning/v2/tasks"
        
        payload = {
            "tuningType": "PEFT",
            "taskType": "GENERATION",
            "model": "HCX-003",
            "name": job_name,
            "description": "ETF 시계열 예측 모델 튜닝",
            "dataset": {
                "data": tuning_data
            },
            "hyperParameters": {
                "epochs": 3,
                "learningRate": 0.00001,
                "batchSize": 4
            }
        }
        
        return self._make_tuning_request(url, payload, "v2 Bearer 방식")

    def start_tuning_job_v3(self, tuning_data, job_name=None):
        """튜닝 작업 시작 - 방법 3: 단순화된 Bearer 방식"""
        if job_name is None:
            job_name = f"hcx003_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        url = f"{self.base_url}/testapp/v1/tuning"
        
        payload = {
            "model": "HCX-003",
            "training_data": tuning_data,
            "task_name": job_name,
            "epochs": 3,
            "learning_rate": 0.00001,
            "batch_size": 4
        }
        
        return self._make_tuning_request(url, payload, "단순화 Bearer 방식")

    def start_tuning_job_v4(self, tuning_data, job_name=None):
        """튜닝 작업 시작 - 방법 4: 직접 데이터 전송"""
        if job_name is None:
            job_name = f"hcx003_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        url = f"{self.base_url}/testapp/v1/api-tools/tuning"
        
        payload = {
            "name": job_name,
            "model": "HCX-003",
            "data": tuning_data,
            "config": {
                "epochs": 3,
                "learning_rate": 1e-5,
                "batch_size": 4
            }
        }
        
        return self._make_tuning_request(url, payload, "직접 데이터 전송")

    def _make_tuning_request(self, url, payload, method_name):
        """튜닝 요청 공통 처리"""
        try:
            print(f"\n{method_name} 시도 중...")
            print(f"URL: {url}")
            
            # 데이터 크기 확인
            data_field = payload.get('dataset', payload.get('training_data', payload.get('data', [])))
            if isinstance(data_field, dict):
                data_count = len(data_field.get('data', []))
            else:
                data_count = len(data_field) if isinstance(data_field, list) else 0
            
            print(f"데이터 샘플 수: {data_count}개")
            
            response = requests.post(url, headers=self.headers, json=payload)
            
            print(f"응답 상태 코드: {response.status_code}")
            print(f"응답 내용: {response.text}")
            
            if response.status_code in [200, 201]:
                result = response.json()
                task_id = (result.get('taskId') or 
                          result.get('id') or 
                          result.get('result', {}).get('taskId') or
                          result.get('task_id'))
                
                if task_id:
                    print(f"✅ {method_name} 성공! 작업 ID: {task_id}")
                    return task_id
                else:
                    print(f"⚠️ {method_name} 응답은 성공했지만 작업 ID를 찾을 수 없음")
                    return None
            else:
                print(f"❌ {method_name} 실패")
                return None
            
        except Exception as e:
            print(f"❌ {method_name} 오류: {e}")
            return None

    def check_tuning_status(self, task_id):
        """튜닝 상태 확인"""
        # 여러 가능한 상태 확인 엔드포인트 시도
        endpoints = [
            f"{self.base_url}/tuning/v2/tasks/{task_id}",
            f"{self.base_url}/testapp/v1/api-tools/tuning/tasks/{task_id}",
            f"{self.base_url}/testapp/v1/tuning/{task_id}"
        ]
        
        for endpoint in endpoints:
            try:
                response = requests.get(endpoint, headers=self.headers)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code != 404:
                    print(f"상태 확인 시도 ({endpoint}): {response.status_code}")
                    
            except Exception as e:
                continue
        
        print(f"❌ 모든 상태 확인 엔드포인트 실패")
        return {}

    def wait_for_completion(self, task_id, interval=300, max_wait_time=7200):
        """튜닝 완료 대기 (최대 2시간)"""
        print(f"튜닝 완료 대기 중... (5분마다 상태 확인, 최대 2시간)")
        
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            status_info = self.check_tuning_status(task_id)
            
            if not status_info:
                print("상태 정보를 가져올 수 없습니다. 5분 후 재시도...")
                time.sleep(interval)
                continue
            
            status = status_info.get('status', 'UNKNOWN')
            progress = status_info.get('progress', 0)
            
            print(f"튜닝 상태: {status} 진행률: {progress}%")
            
            if status in ['COMPLETED', 'SUCCESS', 'FINISHED']:
                model_id = (status_info.get('tunedModelId') or 
                           status_info.get('modelId') or
                           status_info.get('model_id'))
                print(f"✅ 튜닝 완료! 모델 ID: {model_id}")
                return model_id
                
            elif status in ['FAILED', 'ERROR', 'CANCELLED']:
                error_msg = status_info.get('errorMessage', '알 수 없는 오류')
                print(f"❌ 튜닝 실패: {error_msg}")
                return None
            
            time.sleep(interval)
        
        print("❌ 최대 대기 시간 초과")
        return None

    def test_tuned_model(self, model_id, prompt):
        """튜닝된 모델 테스트"""
        url = f"{self.base_url}/testapp/v1/chat-completions/{model_id}"
        
        payload = {
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "topP": 0.8,
            "topK": 0,
            "maxTokens": 500,
            "temperature": 0.3,
            "repeatPenalty": 1.0,
            "stopBefore": [],
            "includeAiFilters": True
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                if 'result' in result and 'message' in result['result']:
                    return result['result']['message']['content']
                else:
                    return f"응답 구조 오류: {result}"
            else:
                return f"API 오류 ({response.status_code}): {response.text}"
                
        except Exception as e:
            return f"네트워크 오류: {e}"


def main():
    """메인 실행 함수"""
    print("=== HyperCLOVA X HCX-003 튜닝 (Bearer 토큰 방식) ===\n")
    
    try:
        # 클라이언트 초기화
        client = HyperCLOVAXTuningClient()
        
        # 1단계: API 연결 테스트
        print("1단계: API 연결 테스트 중...")
        if not client.test_api_connection():
            print("❌ API 연결 실패. 프로그램을 종료합니다.")
            return
        
        # 2단계: 튜닝 데이터 로드
        print("\n2단계: 튜닝 데이터 로드 중...")
        jsonl_file_path = 'data/hyperclova_tuning/fixed_tuning_data_20250628_122248.jsonl'
        tuning_data = client.load_jsonl_data(jsonl_file_path)
        
        if not tuning_data:
            print("❌ 튜닝 데이터가 없습니다. 프로그램을 종료합니다.")
            return
        
        if len(tuning_data) < 10:
            print(f"⚠️ 튜닝 데이터가 부족합니다. ({len(tuning_data)}개)")
            print("최소 10개 이상의 샘플이 권장됩니다.")
        
        # 3단계: 여러 방법으로 튜닝 시도
        print("\n3단계: 튜닝 작업 시작 중...")
        
        task_id = None
        
        # 방법 1 시도
        task_id = client.start_tuning_job_v1(tuning_data)
        
        # 방법 1 실패시 방법 2 시도
        if not task_id:
            task_id = client.start_tuning_job_v2(tuning_data)
        
        # 방법 2 실패시 방법 3 시도
        if not task_id:
            task_id = client.start_tuning_job_v3(tuning_data)
        
        # 방법 3 실패시 방법 4 시도
        if not task_id:
            task_id = client.start_tuning_job_v4(tuning_data)
        
        if not task_id:
            print("\n❌ 모든 튜닝 방법이 실패했습니다.")
            print("가능한 원인:")
            print("1. HCX-003 모델이 현재 튜닝을 지원하지 않을 수 있음")
            print("2. 계정에 튜닝 권한이 없을 수 있음")
            print("3. 데이터 형식이 요구사항에 맞지 않을 수 있음")
            print("4. 최소 데이터 요구량을 충족하지 않을 수 있음")
            return
        
        # 4단계: 튜닝 완료 대기
        print(f"\n4단계: 튜닝 완료 대기 중... (작업 ID: {task_id})")
        model_id = client.wait_for_completion(task_id)
        
        if not model_id:
            print("❌ 튜닝이 완료되지 않았습니다.")
            return
        
        # 5단계: 튜닝된 모델 테스트
        print(f"\n5단계: 튜닝된 모델 테스트 중... (모델 ID: {model_id})")
        
        test_prompt = """다음은 310970 ETF의 정규화된 가격 시계열 데이터입니다.
과거 30일간의 데이터를 바탕으로 향후 10일의 가격을 예측하세요.

입력 데이터: 0.647059 0.500000 0.536765 0.463235 0.647059 0.794118 0.808824 0.838235 0.713235 1.000000 0.823529 0.757353 0.485294 0.544118 0.397059 0.338235 0.536765 0.654412 0.360294 0.397059 0.308824 0.448529 0.235294 0.191176 0.301471 0.058824 0.279412 0.426471 0.139706 0.000000

예상 출력:"""
        
        response = client.test_tuned_model(model_id, test_prompt)
        print(f"튜닝된 모델 응답:\n{response}")
        
        # 결과 저장
        result_info = {
            "task_id": task_id,
            "model_id": model_id,
            "tuning_date": datetime.now().isoformat(),
            "test_result": response,
            "data_samples": len(tuning_data),
            "api_method": "Bearer Token"
        }
        
        with open('hcx003_tuning_result.json', 'w', encoding='utf-8') as f:
            json.dump(result_info, f, ensure_ascii=False, indent=2)
        
        print(f"\n=== 튜닝 완료 ===")
        print(f"튜닝된 모델 ID: {model_id}")
        print(f"결과 저장: hcx003_tuning_result.json")
        
    except Exception as e:
        print(f"❌ 예상치 못한 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
