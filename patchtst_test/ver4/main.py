import os
import torch
import warnings
warnings.filterwarnings('ignore')

from bitcoin_data_collector import BitcoinDataCollector
from data_preprocessor import DataPreprocessor
from patchtst_model import BitcoinPatchTST, FixedPatchTSTClassifier
from trainer import BitcoinTrainer
from evaluator import ModelEvaluator

def main():
    print("🚀 32 시퀀스 PatchTST 분류 시스템")
    print("Shape 문제 완전 해결 버전")
    print("="*50)
    
    # 1. 32 시퀀스용 데이터 준비
    print("\n1️⃣ 32 시퀀스용 데이터 수집")
    collector = BitcoinDataCollector()
    train_data, val_data, test_data = collector.prepare_data()
    
    if train_data is None:
        print("❌ 데이터 준비 실패!")
        return
    
    # 2. 32 시퀀스 전처리
    print("\n2️⃣ 32 시퀀스 전처리")
    preprocessor = DataPreprocessor()
    
    # 고정 설정
    sequence_length = 32
    batch_size = 16  # 작은 배치로 안정성 확보
    
    try:
        (train_loader, val_loader, test_loader, 
         num_features, class_weights) = preprocessor.create_dataloaders(
            train_data, val_data, test_data, 
            sequence_length=sequence_length, 
            batch_size=batch_size
        )
        
        print(f"✅ 32 시퀀스 전처리 완료")
        print(f"   특성 수: {num_features}")
        
    except Exception as e:
        print(f"❌ 전처리 실패: {e}")
        return
    
    # 3. PatchTST 모델 초기화
    print("\n3️⃣ PatchTST 모델 초기화")
    
    model_config = {
        'sequence_length': 32,
        'num_features': num_features,
        'num_classes': 3
    }
    
    # PatchTST 시도
    try:
        model = BitcoinPatchTST(model_config)
        model_name = "BitcoinPatchTST"
        print("✅ 진짜 PatchTST 사용!")
        
    except Exception as e:
        print(f"⚠️ PatchTST 실패: {e}")
        print("FixedPatchTST 사용")
        model = FixedPatchTSTClassifier(model_config)
        model_name = "FixedPatchTST"
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   파라미터 수: {total_params:,}")
    
    # 4. 학습
    print("\n4️⃣ 32 시퀀스 PatchTST 학습")
    trainer = BitcoinTrainer(model, class_weights)
    
    save_path = f'models/{model_name.lower()}_seq32.pth'
    trainer.train(train_loader, val_loader, epochs=30, save_path=save_path)
    
    # 5. 평가
    print("\n5️⃣ 32 시퀀스 PatchTST 평가")
    if os.path.exists(save_path):
        trainer.load_model(save_path)
    
    evaluator = ModelEvaluator(trainer.model)
    metrics, results_df = evaluator.evaluate(test_loader, save_dir='results')
    
    # 6. 결과
    print("\n" + "="*50)
    print("🎉 32 시퀀스 PatchTST 결과")
    print("="*50)
    
    accuracy = metrics.get('accuracy', 0)
    
    print(f"📊 성능:")
    print(f"   모델: {model_name}")
    print(f"   정확도: {accuracy:.1%}")
    print(f"   F1-Score: {metrics.get('f1', 0):.4f}")
    print(f"   시퀀스: 32 (고정)")
    print(f"   특성: {num_features}개")
    
    if accuracy > 0.5:
        print("🟢 32 시퀀스 PatchTST 성공!")
    else:
        print("🔴 추가 튜닝 필요")
    
    print("\n✅ 완료!")

if __name__ == "__main__":
    main()
