import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class BitcoinDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 
                               'SMA_20', 'EMA_12', 'RSI', 'MACD', 'Volatility']
    
    def load_data(self, file_path):
        """CSV 파일에서 데이터 로드"""
        try:
            data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            print(f"데이터 로드 완료: {len(data)}개 레코드")
            print(f"컬럼: {list(data.columns)}")
            print(f"데이터 타입:\n{data.dtypes}")
            return data
        except Exception as e:
            print(f"데이터 로드 실패: {e}")
            return None
    
    def clean_data(self, data):
        """데이터 정리"""
        df = data.copy()
        
        # MultiIndex 컬럼 처리
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if len(col) == 1 or col[1] == '' else f"{col[0]}_{col[1]}" for col in df.columns]
        
        # 문자열 컬럼 제거
        string_columns = df.select_dtypes(include=['object']).columns
        if len(string_columns) > 0:
            print(f"문자열 컬럼 제거: {list(string_columns)}")
            df = df.drop(columns=string_columns)
        
        # 숫자형 컬럼만 선택
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        df = df[numeric_columns]
        
        # 무한대 값 처리
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # NaN 값 처리
        df = df.dropna()
        
        print(f"데이터 정리 완료: {len(df)}개 레코드, {len(df.columns)}개 컬럼")
        return df
    
    def normalize_data(self, data):
        """데이터 정규화"""
        # 데이터 정리
        cleaned_data = self.clean_data(data)
        
        # 사용할 특성 컬럼만 선택
        available_columns = [col for col in self.feature_columns if col in cleaned_data.columns]
        if not available_columns:
            # 기본 OHLCV 컬럼만 사용
            basic_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            available_columns = [col for col in basic_columns if col in cleaned_data.columns]
        
        if not available_columns:
            raise ValueError("사용 가능한 컬럼이 없습니다.")
        
        print(f"정규화할 컬럼: {available_columns}")
        
        # 정규화 수행
        data_to_normalize = cleaned_data[available_columns]
        normalized_data = self.scaler.fit_transform(data_to_normalize)
        
        # DataFrame으로 변환
        normalized_df = pd.DataFrame(
            normalized_data, 
            columns=available_columns, 
            index=data_to_normalize.index
        )
        
        print(f"정규화 완료: {len(normalized_df)}개 레코드")
        return normalized_df
    
    def create_sequences(self, data, sequence_length, prediction_length):
        """시계열 시퀀스 생성"""
        sequences = []
        targets = []
        
        print(f"시퀀스 생성 - 데이터 크기: {len(data)}, 시퀀스 길이: {sequence_length}, 예측 길이: {prediction_length}")
        
        # 최소 필요한 데이터 크기 확인
        min_required_length = sequence_length + prediction_length
        if len(data) < min_required_length:
            print(f"경고: 데이터 크기({len(data)})가 최소 필요 크기({min_required_length})보다 작습니다.")
            return np.array([]), np.array([])
        
        # 시퀀스 생성
        data_values = data.values
        for i in range(len(data) - sequence_length - prediction_length + 1):
            # 입력 시퀀스 (sequence_length 길이)
            seq = data_values[i:i + sequence_length]
            # 타겟 시퀀스 (prediction_length 길이, Close 가격 인덱스 사용)
            close_idx = data.columns.get_loc('Close') if 'Close' in data.columns else 3
            target = data_values[i + sequence_length:i + sequence_length + prediction_length, close_idx]
            
            sequences.append(seq)
            targets.append(target)
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        print(f"생성된 시퀀스 개수: {len(sequences)}")
        print(f"시퀀스 shape: {sequences.shape}")
        print(f"타겟 shape: {targets.shape}")
        
        return sequences, targets
    
    def split_data(self, data, train_ratio=0.7, val_ratio=0.15):
        """데이터를 학습/검증/테스트로 분할"""
        n = len(data)
        print(f"전체 데이터 크기: {n}")
        
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)
        
        train_data = data.iloc[:train_size]
        val_data = data.iloc[train_size:train_size + val_size]
        test_data = data.iloc[train_size + val_size:]
        
        print(f"분할 후 크기 - 학습: {len(train_data)}, 검증: {len(val_data)}, 테스트: {len(test_data)}")
        
        return train_data, val_data, test_data
    
    def create_dataloaders(self, train_data, val_data, test_data, sequence_length, 
                          prediction_length, batch_size=32):
        """데이터로더 생성"""
        print("=== 데이터로더 생성 시작 ===")
        
        # 시퀀스 생성
        train_sequences, train_targets = self.create_sequences(
            train_data, sequence_length, prediction_length)
        val_sequences, val_targets = self.create_sequences(
            val_data, sequence_length, prediction_length)
        test_sequences, test_targets = self.create_sequences(
            test_data, sequence_length, prediction_length)
        
        # 빈 데이터셋 확인
        if len(train_sequences) == 0:
            raise ValueError(f"학습 데이터셋이 비어있습니다. sequence_length({sequence_length}) 또는 prediction_length({prediction_length})를 줄여보세요.")
        if len(val_sequences) == 0:
            raise ValueError("검증 데이터셋이 비어있습니다.")
        if len(test_sequences) == 0:
            raise ValueError("테스트 데이터셋이 비어있습니다.")
        
        # 데이터셋 생성
        train_dataset = TimeSeriesDataset(train_sequences, train_targets)
        val_dataset = TimeSeriesDataset(val_sequences, val_targets)
        test_dataset = TimeSeriesDataset(test_sequences, test_targets)
        
        # 데이터로더 생성
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print("데이터로더 생성 완료!")
        return train_loader, val_loader, test_loader
