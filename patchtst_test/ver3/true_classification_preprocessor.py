import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.utils.class_weight import compute_class_weight
import pickle
import os

class TrueClassificationDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

class TrueClassificationPreprocessor:
    """PatchTST 논문 기준 전처리기"""
    def __init__(self):
        # 논문에서는 Instance Normalization을 사용하므로 StandardScaler 사용 안함
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
        self.class_weights = None
    
    def prepare_features(self, data):
        """특성 준비 - 정규화 없이 원본 데이터 사용"""
        available_columns = [col for col in self.feature_columns if col in data.columns]
        if not available_columns:
            available_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        print(f"사용할 특성: {len(available_columns)}개")
        
        # NaN과 무한대 값 처리
        features_data = data[available_columns].copy()
        features_data = features_data.replace([np.inf, -np.inf], np.nan)
        features_data = features_data.fillna(method='ffill').fillna(method='bfill')
        
        return features_data
    
    def calculate_class_weights(self, labels):
        """클래스 가중치 계산"""
        class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
        self.class_weights = torch.FloatTensor(class_weights)
        print(f"클래스 가중치: {self.class_weights}")
        return self.class_weights
    
    def create_sequences(self, data, labels, sequence_length=336):  # 논문 권장 336
        """시계열 시퀀스 생성 - 논문 기준"""
        sequences = []
        sequence_labels = []
        
        print(f"PatchTST 시퀀스 생성: 길이={sequence_length}시간 (논문 권장)")
        
        data_values = data.values
        
        for i in range(len(data) - sequence_length):
            # 입력 시퀀스 (논문 권장 336 길이)
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
        """불균형 데이터를 위한 가중 샘플러"""
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels]
        
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
    
    def prepare_train_data(self, train_data, sequence_length=336, batch_size=32):  # 논문 권장 설정
        """학습 데이터 준비 - PatchTST 논문 기준"""
        print(f"PatchTST 학습 데이터 준비: {len(train_data)}개")
        
        # 특성 준비 (정규화 없이)
        features = self.prepare_features(train_data)
        labels = train_data['Label'].values
        
        # 클래스 가중치 계산
        self.calculate_class_weights(labels)
        
        # 학습/검증 분할 (90:10)
        split_idx = int(len(features) * 0.9)
        train_split = features.iloc[:split_idx]
        val_split = features.iloc[split_idx:]
        train_labels = labels[:split_idx]
        val_labels = labels[split_idx:]
        
        print(f"학습 분할: {len(train_split)}개, 검증 분할: {len(val_split)}개")
        
        # 시퀀스 생성
        train_sequences, train_seq_labels = self.create_sequences(train_split, train_labels, sequence_length)
        val_sequences, val_seq_labels = self.create_sequences(val_split, val_labels, sequence_length)
        
        if len(train_sequences) == 0:
            raise ValueError("학습 시퀀스가 생성되지 않았습니다. sequence_length를 줄여보세요.")
        
        # 데이터셋 생성
        train_dataset = TrueClassificationDataset(train_sequences, train_seq_labels)
        val_dataset = TrueClassificationDataset(val_sequences, val_seq_labels)
        
        # 균형 잡힌 샘플러
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
        
        print(f"PatchTST 데이터로더 생성 완료")
        
        # 특성 정보 저장
        os.makedirs('models/true_classification', exist_ok=True)
        with open('models/true_classification/feature_info.pkl', 'wb') as f:
            pickle.dump({
                'feature_columns': features.columns.tolist(),
                'class_weights': self.class_weights
            }, f)
        
        return train_loader, val_loader, len(features.columns)
    
    def prepare_test_data(self, test_data, sequence_length=336):
        """테스트 데이터 준비 - PatchTST 논문 기준"""
        print(f"\n=== PatchTST 테스트 데이터 준비 ===")
        print(f"원본 테스트 데이터 크기: {len(test_data)}시간")
        
        # 시퀀스 길이 조정
        if len(test_data) < sequence_length:
            new_sequence_length = max(100, len(test_data) - 1)  # 최소 100시간
            print(f"시퀀스 길이를 {sequence_length}에서 {new_sequence_length}로 조정")
            sequence_length = new_sequence_length
        
        # 특성 정보 로드
        try:
            with open('models/true_classification/feature_info.pkl', 'rb') as f:
                feature_info = pickle.load(f)
            self.class_weights = feature_info['class_weights']
            print("PatchTST 특성 정보 로드 완료")
        except Exception as e:
            print(f"특성 정보 로드 실패: {e}")
            return np.array([]), np.array([])
        
        # 특성 준비 (정규화 없이)
        features = self.prepare_features(test_data)
        labels = test_data['Label'].values
        
        # 시퀀스 생성
        test_sequences, test_labels = self.create_sequences(features, labels, sequence_length)
        
        if len(test_sequences) == 0:
            print("ERROR: 테스트 시퀀스가 생성되지 않았습니다!")
            return np.array([]), np.array([])
        
        print(f"PatchTST 테스트 시퀀스 생성 완료: {test_sequences.shape}")
        return test_sequences, test_labels
