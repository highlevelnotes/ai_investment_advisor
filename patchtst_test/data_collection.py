import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

class BitcoinDataCollector:
    def __init__(self, symbol="BTC-USD"):
        self.symbol = symbol
    
    def fetch_data(self, period="2y", interval="1h"):
        """비트코인 데이터 수집"""
        try:
            print(f"{self.symbol} 데이터 수집 중...")
            data = yf.download(self.symbol, period=period, interval=interval)
            
            if data.empty:
                raise ValueError("데이터를 가져올 수 없습니다.")
            
            print(f"데이터 수집 완료: {len(data)}개 레코드")
            print(f"기간: {data.index[0]} ~ {data.index[-1]}")
            
            return data
        
        except Exception as e:
            print(f"데이터 수집 실패: {e}")
            return None
    
    def add_technical_indicators(self, data):
        """기술적 지표 추가"""
        df = data.copy()
        
        # 이동평균
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        
        # 변동성
        df['Volatility'] = df['Close'].rolling(window=20).std()
        
        # 가격 변화율
        df['Price_Change'] = df['Close'].pct_change()
        
        # NaN 값 제거
        df = df.dropna()
        
        print(f"기술적 지표 추가 완료. 최종 데이터: {len(df)}개 레코드")
        return df
    
    def save_data(self, data, filepath):
        """데이터 저장"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            data.to_csv(filepath)
            print(f"데이터 저장 완료: {filepath}")
            return True
        except Exception as e:
            print(f"데이터 저장 실패: {e}")
            return False
    
    def collect_and_save(self, filepath="data/raw/bitcoin_data.csv"):
        """데이터 수집 및 저장 파이프라인"""
        # 데이터 수집
        raw_data = self.fetch_data()
        if raw_data is None:
            return False
        
        # 기술적 지표 추가
        processed_data = self.add_technical_indicators(raw_data)
        
        # 데이터 저장
        return self.save_data(processed_data, filepath)
