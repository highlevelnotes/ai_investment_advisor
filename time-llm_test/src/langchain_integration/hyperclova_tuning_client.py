# src/langchain_integration/hyperclova_tuning_client.py
import os
import json
import requests
import time
import logging
from typing import Dict, List, Optional
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class LangChainHyperClovaTuningClient:
    """LangChain 기반 HyperClova X 튜닝 클라이언트 (수정된 API 경로)"""
    
    def __init__(self):
        self.api_key = os.getenv("NCP_CLOVASTUDIO_API_KEY")
        self.apigw_api_key = os.getenv("NCP_APIGW_API_KEY")
        
        if not self.api_key:
            raise ValueError("NCP_CLOVASTUDIO_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
        if not self.apigw_api_key:
            raise ValueError("NCP_APIGW_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
        
        # 여러 가능한 API 경로 시도
        self.possible_base_urls = [
            "https://clovastudio.apigw.ntruss.com",
            "https://clovastudio.stream.ntruss.com",
            "https://clovastudio.ntruss.com"
        ]
        
        # 여러 가능한 API 경로 패턴
        self.possible_api_paths = [
            "/tuning/v1/files",
            "/tuning/v3/files", 
            "/tuning/files",
            "/files",
            "/api/tuning/v2/files",
            "/api/tuning/files",
            "/testapp/v1/tuning/files"
        ]
        
        self.logger = self._setup_logger()
        
        # 튜닝 설정
        self.tuning_config = {
            "model_engine": "HCX-003",
            "hyperparameters": {
                "learning_rate": 0.0001,
                "batch_size": 8,
                "epochs": 3,
                "warmup_steps": 100
            }
        }
        
        # 작동하는 API 경로 찾기
        self.working_base_url = None
        self.working_upload_path = None
        self.working_tuning_path = None
        
        self.logger.info("LangChain HyperClova X 튜닝 클라이언트 초기화 완료")
    
    def _setup_logger(self) -> logging.Logger:
        """로거 설정"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _get_headers(self, request_id: str = None) -> Dict[str, str]:
        """API 요청 헤더 생성"""
        if request_id is None:
            request_id = f"langchain-tuning-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return {
            "Content-Type": "application/json; charset=utf-8",
            "X-NCP-CLOVASTUDIO-API-KEY": self.api_key,
            "X-NCP-APIGW-API-KEY": self.apigw_api_key,
            "X-NCP-CLOVASTUDIO-REQUEST-ID": request_id
        }
    
    def _find_working_api_endpoints(self) -> bool:
        """작동하는 API 엔드포인트 찾기"""
        self.logger.info("작동하는 API 엔드포인트 탐색 중...")
        
        # 간단한 헬스 체크나 GET 요청으로 엔드포인트 확인
        for base_url in self.possible_base_urls:
            for api_path in self.possible_api_paths:
                test_url = f"{base_url}{api_path}"
                
                try:
                    # HEAD 요청으로 엔드포인트 존재 여부 확인
                    response = requests.head(
                        test_url, 
                        headers=self._get_headers(),
                        timeout=5
                    )
                    
                    # 404가 아닌 경우 (401, 403, 405 등도 엔드포인트가 존재함을 의미)
                    if response.status_code != 404:
                        self.working_base_url = base_url
                        self.working_upload_path = api_path
                        # 튜닝 작업 경로는 files를 tasks로 변경
                        self.working_tuning_path = api_path.replace('/files', '/tasks')
                        
                        self.logger.info(f"작동하는 엔드포인트 발견: {test_url}")
                        return True
                        
                except requests.exceptions.RequestException:
                    continue
        
        return False
    
    def upload_training_file(self, jsonl_file_path: str) -> str:
        """튜닝 데이터 파일 업로드 (다중 경로 시도)"""
        try:
            self.logger.info(f"튜닝 파일 업로드 시작: {jsonl_file_path}")
            
            # 작동하는 엔드포인트가 없으면 찾기
            if not self.working_base_url:
                if not self._find_working_api_endpoints():
                    # 모든 자동 탐색이 실패하면 수동으로 시도
                    return self._manual_upload_attempts(jsonl_file_path)
            
            # 작동하는 엔드포인트로 업로드 시도
            upload_url = f"{self.working_base_url}{self.working_upload_path}"
            return self._attempt_file_upload(upload_url, jsonl_file_path)
            
        except Exception as e:
            self.logger.error(f"파일 업로드 오류: {str(e)}")
            # 백업 방법으로 수동 시도
            return self._manual_upload_attempts(jsonl_file_path)
    
    def _manual_upload_attempts(self, jsonl_file_path: str) -> str:
        """수동으로 모든 가능한 URL 시도"""
        self.logger.info("수동으로 모든 가능한 API 경로 시도 중...")
        
        # 가능한 모든 조합 시도
        for base_url in self.possible_base_urls:
            for api_path in self.possible_api_paths:
                upload_url = f"{base_url}{api_path}"
                
                try:
                    self.logger.info(f"시도 중: {upload_url}")
                    result = self._attempt_file_upload(upload_url, jsonl_file_path)
                    
                    # 성공하면 작동하는 경로 저장
                    self.working_base_url = base_url
                    self.working_upload_path = api_path
                    self.working_tuning_path = api_path.replace('/files', '/tasks')
                    
                    self.logger.info(f"성공한 업로드 URL: {upload_url}")
                    return result
                    
                except Exception as e:
                    self.logger.debug(f"실패: {upload_url} - {str(e)}")
                    continue
        
        # 모든 시도가 실패한 경우
        raise Exception("모든 가능한 API 경로에서 파일 업로드 실패. API 문서를 확인하거나 네이버 클라우드 지원팀에 문의하세요.")
    
    def _attempt_file_upload(self, upload_url: str, jsonl_file_path: str) -> str:
        """실제 파일 업로드 시도"""
        with open(jsonl_file_path, 'rb') as f:
            files = {
                'file': (os.path.basename(jsonl_file_path), f, 'application/json')
            }
            headers = {
                "X-NCP-CLOVASTUDIO-API-KEY": self.api_key,
                "X-NCP-APIGW-API-KEY": self.apigw_api_key,
                "X-NCP-CLOVASTUDIO-REQUEST-ID": f"upload-{int(time.time())}"
            }
            
            response = requests.post(upload_url, files=files, headers=headers, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                file_id = result["result"]["fileId"]
                self.logger.info(f"파일 업로드 성공: {file_id}")
                return file_id
            else:
                raise Exception(f"파일 업로드 실패: {response.status_code} - {response.text}")
    
    def create_tuning_job(self, file_id: str, task_name: str = None) -> str:
        """튜닝 작업 생성"""
        try:
            if task_name is None:
                task_name = f"langchain-time-llm-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.logger.info(f"튜닝 작업 생성 중: {task_name}")
            
            # 작동하는 튜닝 경로 사용
            if self.working_base_url and self.working_tuning_path:
                tuning_url = f"{self.working_base_url}{self.working_tuning_path}"
            else:
                # 기본 경로 사용
                tuning_url = f"{self.possible_base_urls[0]}/tuning/v2/tasks"
            
            request_data = {
                "taskName": task_name,
                "modelEngine": self.tuning_config["model_engine"],
                "trainingFileId": file_id,
                "hyperParameters": self.tuning_config["hyperparameters"],
                "description": "LangChain-based Time-LLM for Korean ETF price prediction"
            }
            
            response = requests.post(
                tuning_url,
                headers=self._get_headers(),
                json=request_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                task_id = result["result"]["taskId"]
                self.logger.info(f"튜닝 작업 생성 성공: {task_id}")
                return task_id
            else:
                raise Exception(f"튜닝 작업 생성 실패: {response.status_code} - {response.text}")
                
        except Exception as e:
            self.logger.error(f"튜닝 작업 생성 오류: {str(e)}")
            raise
    
    def check_tuning_status(self, task_id: str) -> Dict:
        """튜닝 작업 상태 확인"""
        try:
            # 작동하는 기본 URL 사용
            if self.working_base_url:
                status_url = f"{self.working_base_url}/tuning/v2/tasks/{task_id}"
            else:
                status_url = f"{self.possible_base_urls[0]}/tuning/v2/tasks/{task_id}"
            
            response = requests.get(status_url, headers=self._get_headers(), timeout=30)
            
            if response.status_code == 200:
                result = response.json()["result"]
                status = result["status"]
                
                self.logger.info(f"튜닝 상태: {status}")
                
                if status == "COMPLETED":
                    model_id = result.get("tunedModelId")
                    self.logger.info(f"튜닝 완료! 모델 ID: {model_id}")
                elif status == "FAILED":
                    error_message = result.get("errorMessage", "Unknown error")
                    self.logger.error(f"튜닝 실패: {error_message}")
                
                return result
            else:
                raise Exception(f"상태 확인 실패: {response.status_code} - {response.text}")
                
        except Exception as e:
            self.logger.error(f"상태 확인 오류: {str(e)}")
            raise
    
    def wait_for_completion(self, task_id: str, check_interval: int = 60) -> str:
        """튜닝 완료까지 대기"""
        self.logger.info(f"튜닝 완료 대기 중... (체크 간격: {check_interval}초)")
        
        while True:
            try:
                status_result = self.check_tuning_status(task_id)
                status = status_result["status"]
                
                if status == "COMPLETED":
                    model_id = status_result["tunedModelId"]
                    self.logger.info(f"튜닝 완료! 최종 모델 ID: {model_id}")
                    return model_id
                elif status == "FAILED":
                    error_message = status_result.get("errorMessage", "Unknown error")
                    raise Exception(f"튜닝 실패: {error_message}")
                elif status in ["RUNNING", "PENDING", "QUEUED"]:
                    self.logger.info(f"튜닝 진행 중... 상태: {status}")
                    time.sleep(check_interval)
                else:
                    self.logger.warning(f"알 수 없는 상태: {status}")
                    time.sleep(check_interval)
                    
            except KeyboardInterrupt:
                self.logger.info("사용자에 의해 대기 중단됨")
                break
            except Exception as e:
                self.logger.error(f"상태 확인 중 오류: {str(e)}")
                time.sleep(check_interval)
