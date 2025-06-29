import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class BitcoinClassificationDataCollector:
    def __init__(self, symbol="BTC-USD"):
        self.symbol = symbol
    
    def collect_historical_data(self, start_date="2023-01-01", end_date="2025-06-30"):
        """1시간 단위 히스토리컬 데이터 수집"""
        try:
            print(f"{self.symbol} 1시간 데이터 수집 중: {start_date} ~ {end_date}")
            
            all_data = []
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            
            current_start = start
            while current_start < end:
                current_end = min(current_start + pd.DateOffset(days=700), end)
                
                print(f"수집 중: {current_start.date()} ~ {current_end.date()}")
                
                try:
                    data_chunk = yf.download(
                        self.symbol, 
                        start=current_start.strftime('%Y-%m-%d'),
                        end=current_end.strftime('%Y-%m-%d'),
                        interval="1h"
                    )
                    
                    if not data_chunk.empty:
                        all_data.append(data_chunk)
                        print(f"  수집된 데이터: {len(data_chunk)}개")
                    
                except Exception as e:
                    print(f"  청크 수집 실패: {e}")
                
                current_start = current_end
            
            if not all_data:
                # 백업: 최근 2년 데이터
                print("백업 방법: 최근 2년 데이터 수집")
                data = yf.download(self.symbol, period="2y", interval="1h")
                return data
            
            # 모든 데이터 결합
            data = pd.concat(all_data, axis=0)
            data = data.sort_index()
            data = data[~data.index.duplicated(keep='first')]
            
            print(f"전체 데이터 수집 완료: {len(data)}개 레코드")
            return data
        
        except Exception as e:
            print(f"데이터 수집 실패: {e}")
            return None
    
    def add_technical_indicators(self, data):
        """분류용 기술적 지표 추가"""
        df = data.copy()
        
        # 컬럼명 정리
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        
        # 가격 기반 지표
        df['SMA_12'] = df['Close'].rolling(window=12).mean()
        df['SMA_24'] = df['Close'].rolling(window=24).mean()
        df['SMA_72'] = df['Close'].rolling(window=72).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_24'] = df['Close'].ewm(span=24).mean()
        
        # 모멘텀 지표
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_24']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # 볼린저 밴드
        df['BB_Middle'] = df['Close'].rolling(window=24).mean()
        bb_std = df['Close'].rolling(window=24).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # 변동성 지표
        df['Volatility_12'] = df['Close'].rolling(window=12).std()
        df['Volatility_24'] = df['Close'].rolling(window=24).std()
        
        # 수익률 지표
        df['Return_1h'] = df['Close'].pct_change()
        df['Return_4h'] = df['Close'].pct_change(periods=4)
        df['Return_24h'] = df['Close'].pct_change(periods=24)
        
        # 거래량 지표
        df['Volume_SMA'] = df['Volume'].rolling(window=24).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # 시간 특성 (순환 인코딩)
        df['Hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['Hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        df['DayOfWeek_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['DayOfWeek_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        
        # 고급 기술적 지표
        # Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        # Williams %R
        df['Williams_R'] = -100 * ((high_14 - df['Close']) / (high_14 - low_14))
        
        # 무한대 값과 NaN 처리
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        print(f"기술적 지표 추가 완료. 최종 데이터: {len(df)}개 레코드")
        return df
    
    def create_classification_labels(self, data, prediction_hours=1):
        """분류 레이블 생성 (상승=1, 하락=0)"""
        df = data.copy()
        
        # 미래 가격 변화율 계산
        df['Future_Return'] = df['Close'].shift(-prediction_hours) / df['Close'] - 1
        
        # 이진 분류 레이블 생성
        df['Label'] = (df['Future_Return'] > 0).astype(int)
        
        # 미래 데이터가 없는 마지막 행들 제거
        df = df.dropna()
        
        # 클래스 분포 확인
        class_counts = df['Label'].value_counts()
        print(f"클래스 분포:")
        print(f"  하락(0): {class_counts.get(0, 0)}개 ({class_counts.get(0, 0)/len(df)*100:.1f}%)")
        print(f"  상승(1): {class_counts.get(1, 0)}개 ({class_counts.get(1, 0)/len(df)*100:.1f}%)")
        
        return df
    
    def split_train_test_data(self, data, test_hours=720):
        """학습용과 테스트용 데이터 분할"""
        train_data = data.iloc[:-test_hours]
        test_data = data.iloc[-test_hours:]
        
        print(f"학습 데이터: {len(train_data)}개 ({train_data.index[0]} ~ {train_data.index[-1]})")
        print(f"테스트 데이터: {len(test_data)}개 ({test_data.index[0]} ~ {test_data.index[-1]})")
        
        # 각 세트의 클래스 분포 확인
        train_dist = train_data['Label'].value_counts()
        test_dist = test_data['Label'].value_counts()
        
        print(f"학습 데이터 클래스 분포: 하락 {train_dist.get(0, 0)}, 상승 {train_dist.get(1, 0)}")
        print(f"테스트 데이터 클래스 분포: 하락 {test_dist.get(0, 0)}, 상승 {test_dist.get(1, 0)}")
        
        return train_data, test_data
    
    def collect_and_prepare_data(self):
        """전체 데이터 수집 및 준비 파이프라인"""
        # 1. 데이터 수집
        raw_data = self.collect_historical_data()
        if raw_data is None:
            return None, None
        
        # 2. 기술적 지표 추가
        processed_data = self.add_technical_indicators(raw_data)
        
        # 3. 분류 레이블 생성
        labeled_data = self.create_classification_labels(processed_data)
        
        # 4. 학습/테스트 분할
        train_data, test_data = self.split_train_test_data(labeled_data)
        
        # 5. 데이터 저장
        os.makedirs('data/classification', exist_ok=True)
        train_data.to_csv('data/classification/train_data.csv')
        test_data.to_csv('data/classification/test_data.csv')
        
        print("분류용 데이터 저장 완료!")
        return train_data, test_data
