import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pickle
import os

class ClassificationDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

class BitcoinClassificationPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_12', 'SMA_24', 'SMA_72', 'EMA_12', 'EMA_24',
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
            'BB_Middle', 'BB_Upper', 'BB_Lower', 'BB_Position',
            'Volatility_12', 'Volatility_24',
            'Return_1h', 'Return_4h', 'Return_24h',
            'Volume_SMA', 'Volume_Ratio',
            'Hour_sin', 'Hour_cos', 'DayOfWeek_sin', 'DayOfWeek_cos',
            'Stoch_K', 'Stoch_D', 'Williams_R'
        ]
        self.is_fitted = False
        self.class_weights = None
    
    def prepare_features(self, data):
        """특성 준비"""
        available_columns = [col for col in self.feature_columns if col in data.columns]
        if not available_columns:
            available_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        print(f"사용할 특성: {len(available_columns)}개")
        return data[available_columns].copy()
    
    def fit_scaler(self, train_data):
        """스케일러 학습"""
        features = self.prepare_features(train_data)
        self.scaler.fit(features)
        self.is_fitted = True
        
        # 클래스 가중치 계산
        labels = train_data['Label'].values
        class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
        self.class_weights = torch.FloatTensor(class_weights)
        
        print(f"클래스 가중치: {self.class_weights}")
        
        # 스케일러 저장
        os.makedirs('models/classification', exist_ok=True)
        with open('models/classification/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        return features.columns.tolist()
    
    def transform_data(self, data):
        """데이터 변환"""
        if not self.is_fitted:
            raise ValueError("스케일러가 학습되지 않았습니다.")
        
        features = self.prepare_features(data)
        scaled_data = self.scaler.transform(features)
        
        return pd.DataFrame(scaled_data, columns=features.columns, index=data.index)
    
    def create_sequences(self, data, labels, sequence_length=168):
        """시계열 시퀀스 생성 (분류용)"""
        sequences = []
        sequence_labels = []
        
        print(f"분류용 시퀀스 생성: 길이={sequence_length}시간")
        
        data_values = data.values
        
        for i in range(len(data) - sequence_length):
            # 입력 시퀀스
            seq = data_values[i:i + sequence_length]
            # 레이블 (시퀀스 마지막 시점의 레이블)
            label = labels[i + sequence_length - 1]
            
            sequences.append(seq)
            sequence_labels.append(label)
        
        sequences = np.array(sequences)
        sequence_labels = np.array(sequence_labels)
        
        print(f"생성된 시퀀스: {sequences.shape}, 레이블: {sequence_labels.shape}")
        print(f"레이블 분포: 하락={np.sum(sequence_labels==0)}, 상승={np.sum(sequence_labels==1)}")
        
        return sequences, sequence_labels
    
    def create_balanced_sampler(self, labels):
        """불균형 데이터를 위한 가중 샘플러 생성"""
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels]
        
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
    
    def prepare_train_data(self, train_data, sequence_length=168, batch_size=64):
        """학습 데이터 준비"""
        print(f"분류용 학습 데이터 준비: {len(train_data)}개")
        
        # 스케일러 학습
        feature_columns = self.fit_scaler(train_data)
        
        # 데이터 변환
        scaled_data = self.transform_data(train_data)
        labels = train_data['Label'].values
        
        # 학습/검증 분할 (90:10)
        split_idx = int(len(scaled_data) * 0.9)
        train_split = scaled_data.iloc[:split_idx]
        val_split = scaled_data.iloc[split_idx:]
        train_labels = labels[:split_idx]
        val_labels = labels[split_idx:]
        
        print(f"학습 분할: {len(train_split)}개, 검증 분할: {len(val_split)}개")
        
        # 시퀀스 생성
        train_sequences, train_seq_labels = self.create_sequences(train_split, train_labels, sequence_length)
        val_sequences, val_seq_labels = self.create_sequences(val_split, val_labels, sequence_length)
        
        if len(train_sequences) == 0:
            raise ValueError("학습 시퀀스가 생성되지 않았습니다.")
        
        # 데이터셋 생성
        train_dataset = ClassificationDataset(train_sequences, train_seq_labels)
        val_dataset = ClassificationDataset(val_sequences, val_seq_labels)
        
        # 균형 잡힌 샘플러 생성
        train_sampler = self.create_balanced_sampler(train_seq_labels)
        
        # 데이터로더 생성
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            sampler=train_sampler,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0
        )
        
        print(f"분류용 데이터로더 생성 완료")
        return train_loader, val_loader, len(feature_columns)
    
    def prepare_test_data(self, test_data, sequence_length=168):
        """테스트 데이터 준비"""
        print(f"\n=== 분류용 테스트 데이터 준비 ===")
        print(f"원본 테스트 데이터 크기: {len(test_data)}시간")
        
        # 시퀀스 길이 조정
        if len(test_data) < sequence_length:
            new_sequence_length = max(24, len(test_data) - 1)
            print(f"시퀀스 길이를 {sequence_length}에서 {new_sequence_length}로 조정")
            sequence_length = new_sequence_length
        
        # 스케일러 로드
        try:
            with open('models/classification/scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            self.is_fitted = True
            print("분류용 스케일러 로드 완료")
        except Exception as e:
            print(f"스케일러 로드 실패: {e}")
            return np.array([]), np.array([])
        
        # 데이터 변환
        scaled_data = self.transform_data(test_data)
        labels = test_data['Label'].values
        
        # 시퀀스 생성
        test_sequences, test_labels = self.create_sequences(scaled_data, labels, sequence_length)
        
        if len(test_sequences) == 0:
            print("ERROR: 테스트 시퀀스가 생성되지 않았습니다!")
            return np.array([]), np.array([])
        
        print(f"테스트 시퀀스 생성 완료: {test_sequences.shape}")
        return test_sequences, test_labels
