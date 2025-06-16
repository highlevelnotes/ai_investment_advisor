# hyperclova_fine_tuner.py
import requests
import json
import time
from datetime import datetime
import pandas as pd
from config import Config
from data_preprocessor import DataPreprocessor

class HyperClovaFineTuner:
    def __init__(self):
        self.api_key = Config.HYPERCLOVA_X_API_KEY
        self.base_url = "https://clovastudio.stream.ntruss.com/testapp/v1"
        self.headers = {
            "X-NCP-CLOVASTUDIO-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        
    def upload_dataset(self, csv_file_path: str, dataset_name: str):
        """데이터셋 업로드"""
        upload_url = f"{self.base_url}/tune/upload"
        
        with open(csv_file_path, 'rb') as f:
            files = {
                'file': (csv_file_path, f, 'text/csv'),
                'dataset_name': (None, dataset_name),
                'dataset_type': (None, 'instruction')
            }
            
            response = requests.post(
                upload_url, 
                headers={"X-NCP-CLOVASTUDIO-API-KEY": self.api_key},
                files=files
            )
        
        if response.status_code == 200:
            result = response.json()
            dataset_id = result.get('dataset_id')
            print(f"Dataset uploaded successfully. ID: {dataset_id}")
            return dataset_id
        else:
            print(f"Upload failed: {response.status_code}, {response.text}")
            return None
    
    def start_fine_tuning(self, train_dataset_id: str, val_dataset_id: str = None, 
                         model_name: str = "hcx-005", job_name: str = None):
        """파인튜닝 작업 시작"""
        if not job_name:
            job_name = f"pension_portfolio_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        tune_url = f"{self.base_url}/tune/start"
        
        payload = {
            "model_name": model_name,
            "job_name": job_name,
            "train_dataset_id": train_dataset_id,
            "hyperparameters": {
                "learning_rate": 2e-5,
                "num_train_epochs": 3,
                "per_device_train_batch_size": 1,
                "warmup_steps": 100
            }
        }
        
        if val_dataset_id:
            payload["validation_dataset_id"] = val_dataset_id
        
        response = requests.post(tune_url, headers=self.headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            job_id = result.get('job_id')
            print(f"Fine-tuning started successfully. Job ID: {job_id}")
            return job_id
        else:
            print(f"Fine-tuning start failed: {response.status_code}, {response.text}")
            return None
    
    def check_job_status(self, job_id: str):
        """작업 상태 확인"""
        status_url = f"{self.base_url}/tune/status/{job_id}"
        
        response = requests.get(status_url, headers=self.headers)
        
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            print(f"Status check failed: {response.status_code}, {response.text}")
            return None
    
    def monitor_training(self, job_id: str, check_interval=300):
        """훈련 과정 모니터링"""
        print(f"Monitoring training job: {job_id}")
        
        while True:
            status = self.check_job_status(job_id)
            
            if status:
                job_status = status.get('status')
                progress = status.get('progress', {})
                
                print(f"Status: {job_status}")
                if progress:
                    print(f"Progress: {progress}")
                
                if job_status in ['completed', 'failed', 'cancelled']:
                    break
                    
            time.sleep(check_interval)
        
        return status
    
    def get_model_info(self, job_id: str):
        """완료된 모델 정보 조회"""
        model_url = f"{self.base_url}/tune/model/{job_id}"
        
        response = requests.get(model_url, headers=self.headers)
        
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            print(f"Model info retrieval failed: {response.status_code}, {response.text}")
            return None

# 파인튜닝 실행 메인 함수
def run_fine_tuning_pipeline():
    """파인튜닝 파이프라인 실행"""
    # 1. 데이터 전처리
    print("Step 1: Data preprocessing...")
    preprocessor = DataPreprocessor()
    
    # 생성된 데이터 로드
    df = pd.read_csv("pension_portfolio_training_data_20250616_110000.csv")
    
    # 데이터 검증
    validation_report = preprocessor.validate_data(df)
    print(f"Validation report: {validation_report}")
    
    # 데이터 정제
    cleaned_df = preprocessor.clean_data(df)
    print(f"Cleaned data size: {len(cleaned_df)}")
    
    # 훈련/검증 분할
    train_df, val_df = preprocessor.create_train_validation_split(cleaned_df)
    
    # 정제된 데이터 저장
    train_file = "train_data_cleaned.csv"
    val_file = "val_data_cleaned.csv"
    train_df.to_csv(train_file, index=False, encoding='utf-8')
    val_df.to_csv(val_file, index=False, encoding='utf-8')
    
    # 2. 파인튜닝 실행
    print("Step 2: Starting fine-tuning...")
    fine_tuner = HyperClovaFineTuner()
    
    # 데이터셋 업로드
    train_dataset_id = fine_tuner.upload_dataset(train_file, "pension_portfolio_train")
    val_dataset_id = fine_tuner.upload_dataset(val_file, "pension_portfolio_val")
    
    if not train_dataset_id:
        print("Failed to upload training dataset")
        return
    
    # 파인튜닝 시작
    job_id = fine_tuner.start_fine_tuning(train_dataset_id, val_dataset_id)
    
    if not job_id:
        print("Failed to start fine-tuning")
        return
    
    # 3. 훈련 모니터링
    print("Step 3: Monitoring training...")
    final_status = fine_tuner.monitor_training(job_id)
    
    # 4. 결과 확인
    print("Step 4: Checking results...")
    if final_status.get('status') == 'completed':
        model_info = fine_tuner.get_model_info(job_id)
        print(f"Fine-tuning completed successfully!")
        print(f"Model info: {model_info}")
        
        # 모델 설정 업데이트 코드 생성
        generate_model_config_update(model_info)
    else:
        print(f"Fine-tuning failed: {final_status}")

def generate_model_config_update(model_info):
    """파인튜닝된 모델 사용을 위한 설정 업데이트 코드 생성"""
    model_id = model_info.get('model_id')
    
    config_update = f"""
# config.py에 추가할 설정
class Config:
    # 기존 설정...
    
    # 파인튜닝된 모델 설정
    FINETUNED_MODEL_ID = "{model_id}"
    USE_FINETUNED_MODEL = True

# ai_analyzer.py 업데이트 코드
class AIAnalyzer:
    def __init__(self):
        model_name = Config.FINETUNED_MODEL_ID if Config.USE_FINETUNED_MODEL else Config.HYPERCLOVA_MODEL
        
        self.client = ChatClovaX(
            api_key=Config.HYPERCLOVA_X_API_KEY,
            model=model_name,
            max_tokens=Config.HYPERCLOVA_MAX_TOKENS,
            temperature=0.3
        )
"""
    
    # 파일로 저장
    with open("finetuned_model_config.py", "w", encoding="utf-8") as f:
        f.write(config_update)
    
    print("Model configuration update code saved to 'finetuned_model_config.py'")

if __name__ == "__main__":
    run_fine_tuning_pipeline()
