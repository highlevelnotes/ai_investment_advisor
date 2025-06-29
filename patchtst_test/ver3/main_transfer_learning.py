import os
import torch
import warnings
import numpy as np
warnings.filterwarnings('ignore')

from data_collector_classification import BitcoinClassificationDataCollector
from true_classification_preprocessor import TrueClassificationPreprocessor
from transfer_learning_patchtst import TransferLearningPatchTST, FineTuningTrainer
from classification_evaluator import ClassificationEvaluator

def main():
    print("=== 🚀 PatchTST 전이학습 비트코인 분류 시스템 ===")
    print("사전훈련된 PatchTST 모델을 비트코인 데이터로 파인튜닝")
    
    # 1. 데이터 수집
    print("\n1. 비트코인 분류 데이터 수집...")
    collector = BitcoinClassificationDataCollector()
    train_data, test_data = collector.collect_and_prepare_data()
    
    if train_data is None or test_data is None:
        print("❌ 데이터 수집 실패!")
        return
    
    print(f"✅ 데이터 수집 완료")
    print(f"   학습 데이터: {len(train_data):,}시간")
    print(f"   테스트 데이터: {len(test_data):,}시간")
    
    # 2. 전이학습용 데이터 전처리
    print("\n2. 전이학습용 데이터 전처리...")
    preprocessor = TrueClassificationPreprocessor()
    
    # 전이학습에 적합한 설정
    sequence_length = 336  # 사전훈련된 모델과 호환
    batch_size = 16        # 전이학습은 작은 배치 사용
    
    if len(test_data) < sequence_length:
        sequence_length = max(168, len(test_data) // 3)
        print(f"   시퀀스 길이 조정: {sequence_length}시간")
    
    # 학습 데이터 준비
    try:
        train_loader, val_loader, num_features = preprocessor.prepare_train_data(
            train_data, sequence_length, batch_size
        )
        print(f"✅ 전처리 완료")
        print(f"   특성 수: {num_features}개")
        print(f"   클래스 가중치: {preprocessor.class_weights}")
    except Exception as e:
        print(f"❌ 전처리 실패: {e}")
        return
    
    # 3. 전이학습 모델 초기화
    print("\n3. 전이학습 PatchTST 모델 초기화...")
    
    model_config = {
        'sequence_length': sequence_length,
        'num_features': num_features,
        'num_classes': 2,
    }
    
    # 사용 가능한 사전훈련된 모델들 시도
    pretrained_models = [
        "microsoft/patchtst-etth1-forecast",
        "microsoft/patchtst-etth2-forecast", 
        "microsoft/patchtst-ettm1-forecast",
        "microsoft/patchtst-base"
    ]
    
    model = None
    for pretrained_name in pretrained_models:
        try:
            print(f"   시도 중: {pretrained_name}")
            model = TransferLearningPatchTST(model_config, pretrained_name)
            print(f"✅ 사전훈련된 모델 로드 성공: {pretrained_name}")
            break
        except Exception as e:
            print(f"   실패: {e}")
            continue
    
    if model is None:
        print("⚠️ 사전훈련된 모델 로드 실패, 기본 모델로 초기화")
        model = TransferLearningPatchTST(model_config, None)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   전체 파라미터: {total_params:,}")
    print(f"   훈련 가능 파라미터: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    # 4. 전이학습 실행
    print("\n4. 🎯 전이학습 시작...")
    trainer = FineTuningTrainer(model, class_weights=preprocessor.class_weights)
    
    save_path = 'models/transfer_learning/best_model.pth'
    trainer.fine_tune(train_loader, val_loader, epochs=30, save_path=save_path)
    
    # 5. 테스트 데이터 준비
    print("\n5. 테스트 데이터 준비...")
    test_sequences, test_labels = preprocessor.prepare_test_data(
        test_data, sequence_length
    )
    
    if len(test_sequences) == 0:
        print("❌ 테스트 시퀀스가 비어있습니다.")
        return
    
    print(f"✅ 테스트 데이터 준비 완료")
    print(f"   테스트 시퀀스: {test_sequences.shape}")
    print(f"   클래스 분포: 하락={np.sum(test_labels==0)}, 상승={np.sum(test_labels==1)}")
    
    # 6. 전이학습 모델 평가
    print("\n6. 🔍 전이학습 모델 평가...")
    trainer.load_model(save_path)
    evaluator = ClassificationEvaluator(trainer.model)
    
    metrics, results_df = evaluator.evaluate_classification(test_sequences, test_labels)
    
    # 7. 전이학습 결과 분석
    print("\n" + "="*50)
    print("🎉 전이학습 PatchTST 최종 결과")
    print("="*50)
    
    accuracy = metrics['accuracy']
    f1 = metrics['f1']
    auc = metrics['auc']
    precision = metrics['precision']
    recall = metrics['recall']
    
    print(f"📊 성능 지표:")
    print(f"   정확도 (Accuracy): {accuracy:.1%}")
    print(f"   정밀도 (Precision): {precision:.4f}")
    print(f"   재현율 (Recall): {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    print(f"   ROC AUC: {auc:.4f}")
    
    # 전이학습 효과 분석
    print(f"\n🚀 전이학습 효과 분석:")
    if accuracy > 0.65 and auc > 0.70:
        print("🟢 우수: 전이학습이 매우 효과적!")
        print("   사전훈련된 지식이 비트코인 예측에 성공적으로 적용됨")
    elif accuracy > 0.60 and auc > 0.65:
        print("🟡 양호: 전이학습 효과 확인")
        print("   기본 모델보다 개선된 성능")
    elif accuracy > 0.55:
        print("🟠 보통: 일부 전이학습 효과")
    else:
        print("🔴 부족: 전이학습 효과 미미")
        print("   도메인 차이가 클 수 있음")
    
    # 실용성 평가
    print(f"\n💰 투자 관점 분석:")
    if accuracy > 0.60:
        expected_return = (accuracy - 0.5) * 2 * 100  # 간단한 기대수익률 계산
        print(f"   기대 수익률: +{expected_return:.1f}% (이론적)")
        print(f"   실용적 활용 가능성: 높음")
    else:
        print(f"   랜덤 예측과 유사한 수준")
        print(f"   추가 개선 필요")
    
    print(f"\n📁 출력 파일:")
    print(f"   - {save_path}")
    print(f"   - results/classification/predictions.csv")
    print(f"   - results/classification/metrics.json")
    print(f"   - results/classification/plots/classification_results.png")
    
    print(f"\n✨ 전이학습 완료!")

if __name__ == "__main__":
    main()
