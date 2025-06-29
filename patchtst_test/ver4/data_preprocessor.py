import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

class BitcoinDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_fitted = False
        
    def select_features(self, data):
        """특성 선택"""
        feature_cols = [col for col in data.columns if col != 'Label']
        numeric_cols = data[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        valid_cols = []
        for col in numeric_cols:
            nan_ratio = data[col].isnull().sum() / len(data)
            if nan_ratio < 0.1:
                valid_cols.append(col)
        
        self.feature_columns = valid_cols
        print(f"📋 선택된 특성: {len(self.feature_columns)}개")
        
        return data[self.feature_columns]
    
    def fit_scaler(self, train_data):
        """스케일러 학습"""
        features = self.select_features(train_data)
        features = features.fillna(features.mean())
        
        self.scaler.fit(features)
        self.is_fitted = True
        
        print("✅ 스케일러 학습 완료")
        return features
    
    def transform_data(self, data):
        """데이터 변환"""
        if not self.is_fitted:
            raise ValueError("스케일러가 학습되지 않았습니다.")
        
        features = data[self.feature_columns]
        features = features.fillna(features.mean())
        
        scaled_features = self.scaler.transform(features)
        
        return pd.DataFrame(
            scaled_features, 
            columns=self.feature_columns, 
            index=data.index
        )
    
    def create_sequences(self, data, labels, sequence_length=32):  # PatchTST 기본값 32
        """시계열 시퀀스 생성 - PatchTST 기본 설정"""
        sequences = []
        sequence_labels = []
        
        print(f"🔄 PatchTST용 시퀀스 생성 (길이: {sequence_length})")
        print(f"   입력 데이터 크기: {len(data)}")
        
        if len(data) < sequence_length:
            print(f"❌ 데이터 크기({len(data)})가 시퀀스 길이({sequence_length})보다 작습니다!")
            return np.array([]), np.array([])
        
        data_values = data.values
        
        for i in range(len(data) - sequence_length + 1):
            seq = data_values[i:i + sequence_length]
            label = labels[i + sequence_length - 1]
            
            if seq.shape[0] == sequence_length:
                sequences.append(seq)
                sequence_labels.append(label)
        
        sequences = np.array(sequences)
        sequence_labels = np.array(sequence_labels)
        
        print(f"✅ PatchTST 시퀀스 생성 완료: {sequences.shape}")
        
        # 클래스 분포 확인
        unique, counts = np.unique(sequence_labels, return_counts=True)
        for cls, count in zip(unique, counts):
            print(f"   클래스 {cls}: {count:,}개 ({count/len(sequence_labels)*100:.1f}%)")
        
        return sequences, sequence_labels
    
    def calculate_class_weights(self, labels):
        """클래스 가중치 계산"""
        classes = np.unique(labels)
        weights = compute_class_weight('balanced', classes=classes, y=labels)
        class_weights = torch.FloatTensor(weights)
        
        print(f"⚖️ 클래스 가중치: {dict(zip(classes, weights))}")
        return class_weights
    
    def create_dataloaders(self, train_data, val_data, test_data, 
                          sequence_length=32, batch_size=32):  # PatchTST 기본값
        """PatchTST용 데이터로더 생성"""
        print("🔧 PatchTST용 데이터로더 생성 중...")
        
        # 1. 스케일러 학습
        self.fit_scaler(train_data)
        
        # 2. 데이터 변환
        train_features = self.transform_data(train_data)
        val_features = self.transform_data(val_data)
        test_features = self.transform_data(test_data)
        
        # 3. PatchTST 기본 설정으로 시퀀스 생성
        train_sequences, train_labels = self.create_sequences(
            train_features, train_data['Label'].values, sequence_length
        )
        val_sequences, val_labels = self.create_sequences(
            val_features, val_data['Label'].values, sequence_length
        )
        test_sequences, test_labels = self.create_sequences(
            test_features, test_data['Label'].values, sequence_length
        )
        
        if len(train_sequences) == 0:
            raise ValueError("PatchTST용 시퀀스 생성 실패!")
        
        # 4. 클래스 가중치 계산
        class_weights = self.calculate_class_weights(train_labels)
        
        # 5. 데이터셋 생성
        train_dataset = BitcoinDataset(train_sequences, train_labels)
        val_dataset = BitcoinDataset(val_sequences, val_labels)
        test_dataset = BitcoinDataset(test_sequences, test_labels)
        
        # 6. 데이터로더 생성
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, 
            num_workers=0, pin_memory=True, drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=True, drop_last=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=True, drop_last=True
        )
        
        print("✅ PatchTST용 데이터로더 생성 완료!")
        
        return (train_loader, val_loader, test_loader, 
                len(self.feature_columns), class_weights)
