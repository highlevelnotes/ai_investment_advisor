import json
import os
from datetime import datetime

def fix_tuning_data_format(input_file, output_file):
    """
    JSONL 파일의 데이터 형식을 HyperCLOVA X 튜닝에 맞게 수정
    """
    fixed_data = []
    error_count = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        data = json.loads(line)
                        
                        # 올바른 형식으로 변환
                        if 'messages' in data:
                            # 이미 올바른 형식
                            fixed_item = {
                                "messages": data['messages']
                            }
                        else:
                            # 다른 형식에서 변환 시도
                            print(f"라인 {line_num}: 알 수 없는 형식, 건너뛰기")
                            error_count += 1
                            continue
                        
                        # messages 배열 검증 및 수정
                        messages = fixed_item['messages']
                        if isinstance(messages, list) and len(messages) >= 2:
                            # role과 content 검증
                            valid_messages = []
                            for msg in messages:
                                if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                                    # role 정규화
                                    role = msg['role'].lower()
                                    if role in ['user', 'assistant', 'system']:
                                        valid_messages.append({
                                            "role": role,
                                            "content": str(msg['content'])
                                        })
                            
                            if len(valid_messages) >= 2:
                                fixed_item['messages'] = valid_messages
                                fixed_data.append(fixed_item)
                            else:
                                print(f"라인 {line_num}: 유효한 메시지가 부족함")
                                error_count += 1
                        else:
                            print(f"라인 {line_num}: messages 형식 오류")
                            error_count += 1
                            
                    except json.JSONDecodeError as e:
                        print(f"라인 {line_num} JSON 파싱 오류: {e}")
                        error_count += 1
                        continue
        
        # 수정된 데이터 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in fixed_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"✓ 데이터 형식 수정 완료:")
        print(f"  - 입력: {input_file}")
        print(f"  - 출력: {output_file}")
        print(f"  - 성공: {len(fixed_data)}개")
        print(f"  - 오류: {error_count}개")
        
        return output_file
        
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {input_file}")
        return None

def create_sample_tuning_data(output_file, num_samples=50):
    """
    샘플 튜닝 데이터 생성 (테스트용)
    """
    sample_data = []
    
    # ETF 시계열 예측 샘플 데이터 생성
    for i in range(num_samples):
        # 랜덤 정규화 데이터 생성 (실제로는 실제 데이터 사용)
        import random
        historical_data = [round(random.random(), 6) for _ in range(30)]
        predicted_data = [round(random.random(), 6) for _ in range(10)]
        
        historical_str = ' '.join(map(str, historical_data))
        predicted_str = ' '.join(map(str, predicted_data))
        
        sample = {
            "messages": [
                {
                    "role": "user",
                    "content": f"다음은 310970 ETF의 정규화된 가격 시계열 데이터입니다.\n과거 30일간의 데이터를 바탕으로 향후 10일의 가격을 예측하세요.\n\n입력 데이터: {historical_str}\n\n예상 출력:"
                },
                {
                    "role": "assistant", 
                    "content": predicted_str
                }
            ]
        }
        sample_data.append(sample)
    
    # 파일 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in sample_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✓ 샘플 튜닝 데이터 생성 완료: {output_file} ({num_samples}개)")
    return output_file

def main_data_fix():
    """데이터 형식 수정 메인 함수"""
    print("=== 튜닝 데이터 형식 수정 도구 ===\n")
    
    input_file = 'data/hyperclova_tuning/time_llm_tuning_20250626_155419.jsonl'
    output_file = f'data/hyperclova_tuning/fixed_tuning_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl'
    
    # 디렉토리 생성
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 원본 파일이 있으면 수정, 없으면 샘플 생성
    if os.path.exists(input_file):
        print("1. 기존 데이터 형식 수정 중...")
        fixed_file = fix_tuning_data_format(input_file, output_file)
        
        if fixed_file:
            print(f"수정된 파일: {fixed_file}")
        else:
            print("데이터 수정 실패")
    else:
        print("1. 원본 파일이 없어 샘플 데이터 생성 중...")
        sample_file = create_sample_tuning_data(output_file, 100)
        print(f"샘플 파일: {sample_file}")

if __name__ == '__main__':
    main_data_fix()
