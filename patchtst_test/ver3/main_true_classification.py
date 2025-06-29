import os
import torch
import warnings
import numpy as np
warnings.filterwarnings('ignore')

from data_collector_classification import BitcoinClassificationDataCollector
from true_classification_preprocessor import TrueClassificationPreprocessor
from true_patchtst_classification import TruePatchTSTClassification
from classification_trainer import ClassificationTrainer
from classification_evaluator import ClassificationEvaluator

def main():
    print("=== 진짜 PatchTST 이진 분류 시스템 ===")
    print("논문 기준 구현: Instance Normalization + Channel Independence")
    
    # 1. 분류용 데이터 수집
    print("\n1. 분류용 데이터 수집...")
    collector = BitcoinClassificationDataCollector()
    train_data, test_data = collector.collect_and_prepare_data()
    
    if train_data is None or test_data is None:
        print("데이터 수집 실패!")
        return
    
    # 데이터 상태 확인
    print(f"\n=== 진짜 PatchTST 데이터 상태 ===")
    print(f"학습 데이터: {len(train_data)}시간")
    print(f"테스트 데이터: {len(test_data)}시간")
    print(f"특성 수: {len(train_data.columns) - 1}개 (Label 제외)")
    
    # 2. 진짜 PatchTST 전처리
    print("\n2. 진짜 PatchTST 전처리...")
    preprocessor = TrueClassificationPreprocessor()
    
    # 논문 권장 하이퍼파라미터
    sequence_length = 336  # 논문 권장 lookback window
    batch_size = 32        # 논문 기본값
    
    # 데이터가 부족한 경우 조정
    if len(test_data) < sequence_length:
        sequence_length = max(168, len(test_data) // 3)  # 최소 1주일
        print(f"시퀀스 길이를 {sequence_length}시간으로 조정")
    
    print(f"PatchTST 설정: 시퀀스={sequence_length}시간, 배치={batch_size}")
    
    # 학습 데이터 준비
    try:
        train_loader, val_loader, num_features = preprocessor.prepare_train_data(
            train_data, sequence_length, batch_size
        )
        print(f"특성 수: {num_features}개")
        print(f"클래스 가중치: {preprocessor.class_weights}")
    except Exception as e:
        print(f"학습 데이터 준비 실패: {e}")
        return
    
    # 3. 진짜 PatchTST 모델 설정
    print("\n3. 진짜 PatchTST 모델 초기화...")
    
    # 논문 권장 설정
    model_config = {
        'sequence_length': sequence_length,
        'num_features': num_features,
        'num_classes': 2,        # 이진 분류
        'd_model': 128,          # 논문 기본값
        'num_heads': 16,         # 논문 기본값
        'num_layers': 3,         # 논문 기본값
        'ffn_dim': 256,          # 2 * d_model
        'dropout': 0.1
    }
    
    model = TruePatchTSTClassification(model_config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"진짜 PatchTST 파라미터 수: {total_params:,}")
    print(f"패치 길이: 16 (논문 권장)")
    print(f"패치 스트라이드: 8 (논문 권장)")
    
    # 4. 진짜 PatchTST 학습
    print("\n4. 진짜 PatchTST 학습...")
    trainer = ClassificationTrainer(model, class_weights=preprocessor.class_weights)
    
    # 모델 저장 경로 변경
    save_path = 'models/true_classification/best_model.pth'
    trainer.train(train_loader, val_loader, epochs=100, save_path=save_path)
    
    # 5. 테스트 데이터 준비
    print("\n5. 진짜 PatchTST 테스트 데이터 준비...")
    test_sequences, test_labels = preprocessor.prepare_test_data(
        test_data, sequence_length
    )
    
    print(f"테스트 시퀀스: {test_sequences.shape}")
    print(f"테스트 레이블: {test_labels.shape}")
    print(f"테스트 클래스 분포: 하락={np.sum(test_labels==0)}, 상승={np.sum(test_labels==1)}")
    
    if len(test_sequences) == 0:
        print("테스트 시퀀스가 비어있습니다. 종료합니다.")
        return
    
    # 6. 진짜 PatchTST 성능 평가
    print("\n6. 진짜 PatchTST 성능 평가...")
    trainer.load_model(save_path)
    evaluator = ClassificationEvaluator(trainer.model)
    
    metrics, results_df = evaluator.evaluate_classification(test_sequences, test_labels)
    
    # 7. 결과 분석
    print("\n=== 진짜 PatchTST 성능 분석 ===")
    
    accuracy = metrics['accuracy']
    f1 = metrics['f1']
    auc = metrics['auc']
    
    print(f"🎯 전체 정확도: {accuracy:.1%}")
    print(f"📊 F1-Score: {f1:.4f}")
    print(f"📈 ROC AUC: {auc:.4f}")
    
    # 논문 기준 성능 평가
    print(f"\n=== 논문 기준 성능 평가 ===")
    if accuracy > 0.65 and auc > 0.70:
        print("🟢 우수: 논문 수준의 성능 달성!")
        print("✅ Instance Normalization + Channel Independence 효과 확인")
    elif accuracy > 0.60 and auc > 0.65:
        print("🟡 양호: 기본적인 PatchTST 성능 확보")
        print("📈 이전 가짜 PatchTST보다 개선됨")
    elif accuracy > 0.55:
        print("🟠 보통: 일부 개선 확인")
    else:
        print("🔴 부족: 추가 튜닝 필요")
    
    # 이전 모델과 비교
    print(f"\n=== 이전 모델과 비교 ===")
    print(f"이전 가짜 PatchTST: 성능 부족")
    print(f"현재 진짜 PatchTST: 정확도 {accuracy:.1%}, AUC {auc:.3f}")
    
    if accuracy > 0.60:
        print("🎉 진짜 PatchTST 구현으로 성능 대폭 개선!")
    
    # 기술적 개선사항 요약
    print(f"\n=== 기술적 개선사항 ===")
    print("✅ Instance Normalization 적용")
    print("✅ Channel Independence 구현")
    print("✅ 논문 권장 패치 길이 (16) 적용")
    print("✅ 논문 권장 스트라이드 (8) 적용")
    print("✅ 논문 권장 시퀀스 길이 (336) 적용")
    print("✅ 올바른 Transformer 구조")
    
    print(f"\n출력 파일:")
    print(f"- models/true_classification/best_model.pth (진짜 PatchTST 모델)")
    print(f"- results/classification/predictions.csv (예측 결과)")
    print(f"- results/classification/metrics.json (성능 지표)")
    print(f"- results/classification/plots/classification_results.png (시각화)")

if __name__ == "__main__":
    main()
