# src/collector/data_processor.py
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
from tqdm import tqdm
import logging

class ETFPriceDataCollector:
    """ETF 가격 데이터 수집기"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def get_etf_price_data(self, etf_code: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """개별 ETF 가격 데이터 수집"""
        try:
            if start_date is None:
                start_date = self.config.start_date
            if end_date is None:
                end_date = self.config.end_date
                
            self.logger.info(f"{etf_code} 가격 데이터 수집 중... ({start_date} ~ {end_date})")
            
            # FinanceDataReader 먼저 시도
            try:
                df = fdr.DataReader(etf_code, start_date, end_date)
                if not df.empty:
                    df.reset_index(inplace=True)
                    df['code'] = etf_code
                    return self._standardize_price_data(df)
            except Exception as e:
                self.logger.warning(f"FinanceDataReader 실패 ({etf_code}): {str(e)}")
            
            # yfinance 시도 (한국 ETF는 .KS 접미사)
            try:
                ticker = f"{etf_code}.KS"
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not df.empty:
                    df.reset_index(inplace=True)
                    df['code'] = etf_code
                    return self._standardize_price_data(df)
            except Exception as e:
                self.logger.warning(f"yfinance 실패 ({etf_code}): {str(e)}")
            
            self.logger.error(f"{etf_code} 데이터 수집 완전 실패")
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"{etf_code} 가격 데이터 수집 오류: {str(e)}")
            return pd.DataFrame()
    
    def _standardize_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """가격 데이터 표준화"""
        # 컬럼명 통일
        column_mapping = {
            'Date': 'date',
            'Open': 'open_price',
            'High': 'high_price',
            'Low': 'low_price',
            'Close': 'close_price',
            'Adj Close': 'adj_close_price',
            'Volume': 'volume'
        }
        
        df = df.rename(columns=column_mapping)
        
        # 필수 컬럼 확인
        required_columns = ['date', 'close_price']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"필수 컬럼 누락: {col}")
        
        # 날짜 형식 통일
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # 결측값 처리
        df = df.dropna(subset=['close_price'])
        
        # 정렬
        df = df.sort_values('date').reset_index(drop=True)
        
        return df
    
    def collect_all_etf_prices(self, etf_list: List[str]) -> Dict[str, pd.DataFrame]:
        """모든 ETF 가격 데이터 수집"""
        self.logger.info(f"{len(etf_list)}개 ETF 가격 데이터 수집 시작...")
        
        price_data = {}
        failed_etfs = []
        
        for etf_code in tqdm(etf_list, desc="ETF 가격 데이터 수집"):
            try:
                df = self.get_etf_price_data(etf_code)
                
                if not df.empty and len(df) >= self.config.min_data_points:
                    price_data[etf_code] = df
                    self.logger.info(f"{etf_code}: {len(df)}개 데이터 포인트 수집 성공")
                else:
                    failed_etfs.append(etf_code)
                    self.logger.warning(f"{etf_code}: 데이터 부족 (최소 {self.config.min_data_points}개 필요)")
                
                # API 제한을 위한 대기
                time.sleep(0.1)
                
            except Exception as e:
                failed_etfs.append(etf_code)
                self.logger.error(f"{etf_code} 수집 실패: {str(e)}")
        
        self.logger.info(f"가격 데이터 수집 완료: 성공 {len(price_data)}개, 실패 {len(failed_etfs)}개")
        
        if failed_etfs:
            self.logger.warning(f"실패한 ETF: {failed_etfs}")
        
        return price_data
