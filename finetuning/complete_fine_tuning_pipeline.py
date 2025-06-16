# complete_fine_tuning_pipeline.py
"""
완전한 파인튜닝 파이프라인 실행 스크립트
이 스크립트는 데이터 생성부터 모델 평가까지 전체 과정을 실행합니다.
"""

import os
import sys
from datetime import datetime

def main():
    """메인 실행 함수"""
    print("=== HyperClova X 파인튜닝 파이프라인 시작 ===")
    print(f"시작 시간: {datetime.now()}")
    
    try:
        # 1단계: 파인튜닝 데이터 생성
        print("\n1단계: 파인튜닝 데이터 생성 중...")
        from fine_tuning_data_generator import generate_fine_tuning_dataset
        training_data, csv_file, jsonl_file = generate_fine_tuning_dataset()
        print(f"✅ 데이터 생성 완료: {len(training_data)}개 샘플")
        
        # 2단계: 파인튜닝 실행
        print("\n2단계: 파인튜닝 실행 중...")
        from hyperclova_fine_tuner import run_fine_tuning_pipeline
        run_fine_tuning_pipeline()
        print("✅ 파인튜닝 완료")
        
        # 3단계: 모델 평가
        print("\n3단계: 모델 성능 평가 중...")
        from model_evaluator import run_model_evaluation
        base_report, finetuned_report = run_model_evaluation()
        print("✅ 모델 평가 완료")
        
        # 4단계: 결과 요약
        print("\n=== 파인튜닝 결과 요약 ===")
        if finetuned_report:
            print(f"베이스 모델 성능: {base_report['overall_score']:.3f}")
            print(f"파인튜닝 모델 성능: {finetuned_report['overall_score']:.3f}")
            improvement = finetuned_report['overall_score'] - base_report['overall_score']
            print(f"성능 향상: {improvement:.3f} ({improvement/base_report['overall_score']*100:.1f}%)")
        else:
            print("파인튜닝 모델 평가를 완료하지 못했습니다.")
        
        print(f"\n완료 시간: {datetime.now()}")
        print("=== 파인튜닝 파이프라인 완료 ===")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
