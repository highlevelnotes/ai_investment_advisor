import yfinance as yf
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class BitcoinDataCollector:
    def __init__(self, symbol="BTC-USD"):
        self.symbol = symbol
    
    def collect_data(self, period="6mo", interval="1h"):  # 6개월로 줄임
        """32 시퀀스용 비트코인 데이터 수집"""
        try:
            print(f"📊 {self.symbol} 데이터 수집 중...")
            print(f"   기간: {period}, 간격: {interval}")
            
            data = yf.download(self.symbol, period=period, interval=interval, progress=False)
            
            if data.empty:
                raise ValueError("데이터 수집 실패")
            
            # 컬럼명 정리
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] for col in data.columns]
            
            print(f"✅ 데이터 수집 완료: {len(data):,}개 레코드")
            print(f"   기간: {data.index[0]} ~ {data.index[-1]}")
            
            return data
            
        except Exception as e:
            print(f"❌ 데이터 수집 실패: {e}")
            return None
    
    def add_technical_indicators(self, data):
        """32 시퀀스에 적합한 기술적 지표"""
        df = data.copy()
        
        print("📈 32 시퀀스용 기술적 지표 계산 중...")
        
        try:
            # 단기 지표 (32 시퀀스에 적합)
            df['SMA_5'] = ta.trend.sma_indicator(df['Close'], window=5)
            df['SMA_10'] = ta.trend.sma_indicator(df['Close'], window=10)
            df['EMA_5'] = ta.trend.ema_indicator(df['Close'], window=5)
            df['EMA_10'] = ta.trend.ema_indicator(df['Close'], window=10)
            
            # 단기 모멘텀
            df['RSI_7'] = ta.momentum.rsi(df['Close'], window=7)
            df['RSI_14'] = ta.momentum.rsi(df['Close'], window=14)
            
            # 단기 MACD
            df['MACD'] = ta.trend.macd_diff(df['Close'], window_slow=12, window_fast=5)
            df['MACD_Signal'] = ta.trend.macd_signal(df['Close'], window_slow=12, window_fast=5)
            
            # 단기 볼린저 밴드
            bb = ta.volatility.BollingerBands(df['Close'], window=10)
            df['BB_Upper'] = bb.bollinger_hband()
            df['BB_Lower'] = bb.bollinger_lband()
            df['BB_Middle'] = bb.bollinger_mavg()
            
            # 가격 변화율
            df['Returns_1h'] = df['Close'].pct_change()
            df['Returns_4h'] = df['Close'].pct_change(periods=4)
            
            # 단기 변동성
            df['Volatility_5'] = df['Returns_1h'].rolling(window=5).std()
            df['Volatility_10'] = df['Returns_1h'].rolling(window=10).std()
            
            # 시간 특성
            df['Hour'] = df.index.hour
            df['DayOfWeek'] = df.index.dayofweek
            
            # 순환 인코딩
            df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
            df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
            
            # NaN 제거
            df = df.dropna()
            
            print(f"✅ 32 시퀀스용 기술적 지표 완료")
            print(f"   최종 데이터: {len(df):,}개 레코드")
            print(f"   특성 수: {len(df.columns)}개")
            
            return df
            
        except Exception as e:
            print(f"❌ 기술적 지표 계산 실패: {e}")
            return data.dropna()
    
    def create_classification_labels(self, data, prediction_hours=1, threshold=0.001):
        """32 시퀀스용 분류 레이블"""
        df = data.copy()
        
        print(f"🎯 32 시퀀스용 분류 레이블 생성...")
        
        # 미래 수익률
        future_returns = df['Close'].shift(-prediction_hours) / df['Close'] - 1
        
        # 3클래스 분류
        df['Label'] = 0  # 보합
        df.loc[future_returns > threshold, 'Label'] = 1    # 상승
        df.loc[future_returns < -threshold, 'Label'] = 2   # 하락
        
        # 미래 데이터 제거
        df = df[:-prediction_hours]
        
        # 클래스 분포
        label_counts = df['Label'].value_counts().sort_index()
        print(f"   클래스 분포:")
        for cls in [0, 1, 2]:
            count = label_counts.get(cls, 0)
            print(f"     클래스 {cls}: {count:,}개 ({count/len(df)*100:.1f}%)")
        
        return df
    
    def split_data(self, data, train_ratio=0.7, val_ratio=0.15):
        """32 시퀀스용 데이터 분할"""
        n = len(data)
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)
        
        train_data = data.iloc[:train_size]
        val_data = data.iloc[train_size:train_size + val_size]
        test_data = data.iloc[train_size + val_size:]
        
        print(f"📊 32 시퀀스용 데이터 분할:")
        print(f"   학습: {len(train_data):,}개")
        print(f"   검증: {len(val_data):,}개") 
        print(f"   테스트: {len(test_data):,}개")
        
        return train_data, val_data, test_data
    
    def prepare_data(self):
        """32 시퀀스용 전체 파이프라인"""
        print("🚀 32 시퀀스용 비트코인 데이터 준비")
        
        # 1. 데이터 수집
        raw_data = self.collect_data()
        if raw_data is None:
            return None, None, None
        
        # 2. 기술적 지표
        processed_data = self.add_technical_indicators(raw_data)
        
        # 3. 분류 레이블
        labeled_data = self.create_classification_labels(processed_data)
        
        # 4. 데이터 분할
        train_data, val_data, test_data = self.split_data(labeled_data)
        
        print("✅ 32 시퀀스용 데이터 준비 완료!")
        return train_data, val_data, test_data
