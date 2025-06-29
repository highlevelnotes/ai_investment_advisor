import os
import torch
import warnings
import numpy as np
warnings.filterwarnings('ignore')

from data_collector_classification import BitcoinClassificationDataCollector
from classification_preprocessor import BitcoinClassificationPreprocessor
from patchtst_classification import PatchTSTClassification
from classification_trainer import ClassificationTrainer
from classification_evaluator import ClassificationEvaluator

def main():
    print("=== 비트코인 PatchTST 이진 분류 시스템 ===")
    print("목표: 1시간 후 가격 상승/하락 예측")
    
    # 1. 분류용 데이터 수집
    print("\n1. 분류용 데이터 수집...")
    collector = BitcoinClassificationDataCollector()
    train_data, test_data = collector.collect_and_prepare_data()
    
    if train_data is None or test_data is None:
        print("데이터 수집 실패!")
        return
    
    # 데이터 상태 확인
    print(f"\n=== 분류 데이터 상태 ===")
    print(f"학습 데이터: {len(train_data)}시간")
    print(f"테스트 데이터: {len(test_data)}시간")
    print(f"특성 수: {len(train_data.columns) - 1}개 (Label 제외)")
    
    # 2. 분류용 데이터 전처리
    print("\n2. 분류용 데이터 전처리...")
    preprocessor = BitcoinClassificationPreprocessor()
    
    # 분류에 최적화된 하이퍼파라미터
    sequence_length = 168  # 1주일 (168시간)
    batch_size = 64
    
    # 데이터가 부족한 경우 조정
    if len(test_data) < sequence_length:
        sequence_length = max(72, len(test_data) // 3)
        print(f"시퀀스 길이를 {sequence_length}시간으로 조정")
    
    print(f"설정: 시퀀스={sequence_length}시간, 배치={batch_size}")
    
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
    
    # 3. 분류용 모델 설정
    print("\n3. 분류용 PatchTST 모델 초기화...")
    
    model_config = {
        'sequence_length': sequence_length,
        'num_features': num_features,
        'num_classes': 2,        # 이진 분류 (상승/하락)
        'patch_length': 24,      # 24시간 패치
        'patch_stride': 12,      # 12시간 스트라이드
        'd_model': 256,
        'num_heads': 8,
        'num_layers': 4,
        'ffn_dim': 512,
        'dropout': 0.1
    }
    
    model = PatchTSTClassification(model_config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"분류 모델 파라미터 수: {total_params:,}")
    
    # 4. 분류 모델 학습
    print("\n4. 분류 모델 학습...")
    trainer = ClassificationTrainer(model, class_weights=preprocessor.class_weights)
    trainer.train(train_loader, val_loader, epochs=100)
    
    # 5. 테스트 데이터 준비
    print("\n5. 분류용 테스트 데이터 준비...")
    test_sequences, test_labels = preprocessor.prepare_test_data(
        test_data, sequence_length
    )
    
    print(f"테스트 시퀀스: {test_sequences.shape}")
    print(f"테스트 레이블: {test_labels.shape}")
    print(f"테스트 클래스 분포: 하락={np.sum(test_labels==0)}, 상승={np.sum(test_labels==1)}")
    
    if len(test_sequences) == 0:
        print("테스트 시퀀스가 비어있습니다. 종료합니다.")
        return
    
    # 6. 분류 성능 평가
    print("\n6. 분류 성능 평가...")
    trainer.load_model('models/classification/best_model.pth')
    evaluator = ClassificationEvaluator(trainer.model)
    
    metrics, results_df = evaluator.evaluate_classification(test_sequences, test_labels)
    
    # 7. 결과 분석
    print("\n=== 최종 분류 성능 분석 ===")
    
    accuracy = metrics['accuracy']
    f1 = metrics['f1']
    auc = metrics['auc']
    
    print(f"🎯 전체 정확도: {accuracy:.1%}")
    print(f"📊 F1-Score: {f1:.4f}")
    print(f"📈 ROC AUC: {auc:.4f}")
    
    # 성능 등급 평가
    if accuracy > 0.60 and auc > 0.65:
        print("🟢 우수: 실용적인 수준의 예측 성능!")
    elif accuracy > 0.55 and auc > 0.60:
        print("🟡 양호: 기본적인 예측 성능 확보")
    elif accuracy > 0.50:
        print("🟠 보통: 랜덤보다는 나은 성능")
    else:
        print("🔴 부족: 성능 개선 필요")
    
    # 실용성 분석
    print(f"\n=== 실용성 분석 ===")
    correct_predictions = np.sum(results_df['Correct'])
    total_predictions = len(results_df)
    
    print(f"총 예측 횟수: {total_predictions:,}회")
    print(f"정확한 예측: {correct_predictions:,}회")
    print(f"잘못된 예측: {total_predictions - correct_predictions:,}회")
    
    # 수익성 시뮬레이션 (간단한 예시)
    up_predictions = results_df['Predicted'] == 1
    actual_up = results_df['Actual'] == 1
    
    true_positives = np.sum(up_predictions & actual_up)
    false_positives = np.sum(up_predictions & ~actual_up)
    
    if np.sum(up_predictions) > 0:
        precision_up = true_positives / np.sum(up_predictions)
        print(f"\n상승 예측 정밀도: {precision_up:.1%}")
        print(f"상승 예측 시 실제 상승 확률: {precision_up:.1%}")
    
    print(f"\n출력 파일:")
    print(f"- models/classification/best_model.pth (학습된 분류 모델)")
    print(f"- results/classification/predictions.csv (예측 결과)")
    print(f"- results/classification/metrics.json (성능 지표)")
    print(f"- results/classification/plots/classification_results.png (시각화)")

if __name__ == "__main__":
    main()
