# data_preprocessor.py
import pandas as pd
import json
import re
from typing import List, Dict

class DataPreprocessor:
    def __init__(self):
        self.max_length = 8000  # HyperClova X 제한사항
        
    def validate_data(self, df: pd.DataFrame) -> Dict:
        """데이터 검증"""
        validation_report = {
            'total_samples': len(df),
            'valid_samples': 0,
            'invalid_samples': 0,
            'issues': []
        }
        
        # 필수 컬럼 체크
        required_columns = ['C_ID', 'T_ID', 'Text', 'Completion']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            validation_report['issues'].append(f"Missing columns: {missing_columns}")
            return validation_report
        
        # 각 행 검증
        for idx, row in df.iterrows():
            issues = []
            
            # 길이 체크
            total_length = len(str(row['Text'])) + len(str(row['Completion']))
            if total_length > self.max_length:
                issues.append(f"Row {idx}: Total length ({total_length}) exceeds limit")
            
            # 빈 값 체크
            if pd.isna(row['Text']) or pd.isna(row['Completion']):
                issues.append(f"Row {idx}: Empty Text or Completion")
            
            # 최소 길이 체크
            if len(str(row['Text'])) < 10 or len(str(row['Completion'])) < 10:
                issues.append(f"Row {idx}: Text or Completion too short")
            
            if issues:
                validation_report['invalid_samples'] += 1
                validation_report['issues'].extend(issues)
            else:
                validation_report['valid_samples'] += 1
        
        return validation_report
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 정제"""
        cleaned_df = df.copy()
        
        # 특수문자 정제
        for col in ['Text', 'Completion']:
            cleaned_df[col] = cleaned_df[col].astype(str)
            # 불필요한 공백 제거
            cleaned_df[col] = cleaned_df[col].str.strip()
            # 연속된 공백 정리
            cleaned_df[col] = cleaned_df[col].str.replace(r'\s+', ' ', regex=True)
        
        # 길이 제한 적용
        mask = (cleaned_df['Text'].str.len() + cleaned_df['Completion'].str.len()) <= self.max_length
        cleaned_df = cleaned_df[mask]
        
        # 빈 값 제거
        cleaned_df = cleaned_df.dropna(subset=['Text', 'Completion'])
        
        # ID 재정렬
        cleaned_df['C_ID'] = range(len(cleaned_df))
        cleaned_df['T_ID'] = 0
        
        return cleaned_df
    
    def create_train_validation_split(self, df: pd.DataFrame, split_ratio=0.8):
        """훈련/검증 데이터 분할"""
        train_size = int(len(df) * split_ratio)
        
        train_df = df.iloc[:train_size].copy()
        val_df = df.iloc[train_size:].copy()
        
        # ID 재정렬
        train_df['C_ID'] = range(len(train_df))
        val_df['C_ID'] = range(len(val_df))
        
        return train_df, val_df
