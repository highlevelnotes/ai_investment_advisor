import os
import torch
import warnings
warnings.filterwarnings('ignore')

# 로컬 모듈 import
from data_collection import BitcoinDataCollector
from data_preprocessing import BitcoinDataPreprocessor
from patchtst_model import SimplePatchTST
from trainer import PatchTSTTrainer
from evaluator import ModelEvaluator

def main():
    print("=== 비트코인 PatchTST 예측 프로젝트 시작 ===")
    
    # 1. 데이터 수집
    print("\n1. 비트코인 데이터 수집 중...")
    collector = BitcoinDataCollector()
    if not collector.collect_and_save():
        print("데이터 수집 실패!")
        return
    
    # 2. 데이터 전처리
    print("\n2. 데이터 전처리 중...")
    preprocessor = BitcoinDataPreprocessor()
    
    # 하이퍼파라미터 설정
    sequence_length = 48  # 48시간 (2일)
    prediction_length = 1  # 1시간 예측
    batch_size = 32
    
    # 데이터 로드
    data = preprocessor.load_data('data/raw/bitcoin_data.csv')
    if data is None:
        return
    
    # 정규화
    normalized_data = preprocessor.normalize_data(data)
    
    # 데이터 분할
    train_data, val_data, test_data = preprocessor.split_data(normalized_data)
    
    # 데이터로더 생성
    try:
        train_loader, val_loader, test_loader = preprocessor.create_dataloaders(
            train_data, val_data, test_data, 
            sequence_length, prediction_length, batch_size
        )
    except ValueError as e:
        print(f"데이터로더 생성 실패: {e}")
        print("시퀀스 길이를 줄여서 다시 시도합니다...")
        sequence_length = 24
        train_loader, val_loader, test_loader = preprocessor.create_dataloaders(
            train_data, val_data, test_data, 
            sequence_length, prediction_length, batch_size
        )
    
    # 3. 모델 설정
    print("\n3. 모델 초기화 중...")
    
    model_config = {
        'sequence_length': sequence_length,
        'prediction_length': prediction_length,
        'num_features': normalized_data.shape[1],
        'patch_length': 8,
        'patch_stride': 4,
        'd_model': 128,
        'num_heads': 8,
        'num_layers': 4,
        'ffn_dim': 256,
        'dropout': 0.1,
        'attention_dropout': 0.1
    }
    
    model = SimplePatchTST(model_config)
    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. 모델 학습
    print("\n4. 모델 학습 시작...")
    trainer = PatchTSTTrainer(model)
    trainer.train(train_loader, val_loader, epochs=50)
    
    # 5. 모델 평가
    print("\n5. 모델 평가 중...")
    trainer.load_model('outputs/models/best_model.pth')
    evaluator = ModelEvaluator(trainer.model)
    metrics, predictions, actuals = evaluator.evaluate_model(test_loader)
    
    print("\n=== 프로젝트 완료 ===")
    print("결과 파일들:")
    print("- outputs/models/best_model.pth (학습된 모델)")
    print("- outputs/predictions/test_results.csv (예측 결과)")
    print("- outputs/predictions/metrics.csv (평가 지표)")
    print("- outputs/visualizations/ (시각화 결과)")

if __name__ == "__main__":
    main()
