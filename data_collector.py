# data_collector.py
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from pykrx import stock
from pykrx import bond
from config import Config, ETF_CODES, ECONOMIC_INDICATORS
import time
import warnings
warnings.filterwarnings('ignore')

class DataCollector:
    def __init__(self):
        self.ecos_api_key = Config.ECOS_API_KEY
        self.ecos_url = Config.ECOS_API_URL
        
    def get_etf_data(self, period='1y'):
        """PyKRX를 사용한 ETF 데이터 수집"""
        etf_data = {}
        
        # 기간 설정
        end_date = datetime.now().strftime('%Y%m%d')
        if period == '1y':
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        elif period == '6m':
            start_date = (datetime.now() - timedelta(days=180)).strftime('%Y%m%d')
        elif period == '3m':
            start_date = (datetime.now() - timedelta(days=90)).strftime('%Y%m%d')
        else:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        
        try:
            for category, etfs in ETF_CODES.items():
                category_data = {}
                
                for code, name in etfs.items():
                    try:
                        # PyKRX로 ETF 가격 데이터 수집
                        print(f"데이터 수집 중: {name} ({code})")
                        
                        # 일별 가격 데이터
                        df = stock.get_etf_ohlcv_by_date(start_date, end_date, code)
                        
                        if df is not None and not df.empty:
                            # 수익률 계산
                            returns = df['종가'].pct_change().dropna()
                            
                            # 최신 정보
                            latest_price = df['종가'].iloc[-1]
                            latest_volume = df['거래량'].iloc[-1]
                            
                            # ETF 기본 정보 (포트폴리오 구성 등)
                            try:
                                etf_portfolio = stock.get_etf_portfolio_deposit_file(code)
                                portfolio_info = etf_portfolio if etf_portfolio is not None else pd.DataFrame()
                            except:
                                portfolio_info = pd.DataFrame()
                            
                            category_data[name] = {
                                'code': code,
                                'price': latest_price,
                                'returns': returns,
                                'volume': latest_volume,
                                'history': df.rename(columns={
                                    '시가': 'Open',
                                    '고가': 'High', 
                                    '저가': 'Low',
                                    '종가': 'Close',
                                    '거래량': 'Volume'
                                }),
                                'portfolio_composition': portfolio_info,
                                'category': category
                            }
                            
                            print(f"✅ {name} 데이터 수집 완료")
                            
                        else:
                            print(f"❌ {name} 데이터 없음 - 샘플 데이터 생성")
                            category_data[name] = self._generate_sample_etf_data(code, name, category)
                        
                        # API 호출 제한 방지
                        time.sleep(0.1)
                        
                    except Exception as e:
                        print(f"❌ {name} 데이터 수집 실패: {e}")
                        category_data[name] = self._generate_sample_etf_data(code, name, category)
                
                etf_data[category] = category_data
                
            return etf_data
            
        except Exception as e:
            print(f"ETF 데이터 수집 중 전체 오류: {e}")
            return self._get_sample_etf_data()
    
    def get_market_data(self):
        """시장 전체 데이터 수집"""
        try:
            today = datetime.now().strftime('%Y%m%d')
            
            # KOSPI 지수
            kospi = stock.get_index_ohlcv_by_date(
                (datetime.now() - timedelta(days=30)).strftime('%Y%m%d'),
                today,
                "1001"  # KOSPI
            )
            
            # KOSDAQ 지수
            kosdaq = stock.get_index_ohlcv_by_date(
                (datetime.now() - timedelta(days=30)).strftime('%Y%m%d'),
                today,
                "2001"  # KOSDAQ
            )
            
            # ETF 시장 현황
            etf_list = stock.get_etf_ticker_list(today)
            
            market_data = {
                'kospi': {
                    'current': kospi['종가'].iloc[-1] if not kospi.empty else 2500,
                    'change': kospi['종가'].iloc[-1] - kospi['종가'].iloc[-2] if len(kospi) > 1 else 0,
                    'change_pct': ((kospi['종가'].iloc[-1] / kospi['종가'].iloc[-2]) - 1) * 100 if len(kospi) > 1 else 0
                },
                'kosdaq': {
                    'current': kosdaq['종가'].iloc[-1] if not kosdaq.empty else 800,
                    'change': kosdaq['종가'].iloc[-1] - kosdaq['종가'].iloc[-2] if len(kosdaq) > 1 else 0,
                    'change_pct': ((kosdaq['종가'].iloc[-1] / kosdaq['종가'].iloc[-2]) - 1) * 100 if len(kosdaq) > 1 else 0
                },
                'etf_count': len(etf_list) if etf_list is not None else 500
            }
            
            return market_data
            
        except Exception as e:
            print(f"시장 데이터 수집 실패: {e}")
            return {
                'kospi': {'current': 2500, 'change': 10, 'change_pct': 0.4},
                'kosdaq': {'current': 800, 'change': 5, 'change_pct': 0.6},
                'etf_count': 500
            }
    
    def get_economic_indicators(self):
        """경제지표 데이터 수집 (ECOS API)"""
        if not self.ecos_api_key:
            return self._get_sample_economic_data()
        
        indicators = {}
        
        try:
            for name, code in ECONOMIC_INDICATORS.items():
                try:
                    # ECOS API 호출
                    url = f"{self.ecos_url}/StatisticSearch/{self.ecos_api_key}/json/kr/1/100/{code}/M/202301/202412/"
                    response = requests.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if 'StatisticSearch' in data and 'row' in data['StatisticSearch']:
                            rows = data['StatisticSearch']['row']
                            values = [float(row['DATA_VALUE']) for row in rows if row['DATA_VALUE'] != '-']
                            
                            if values:
                                indicators[name] = {
                                    'current': values[-1],
                                    'previous': values[-2] if len(values) > 1 else values[-1],
                                    'trend': 'up' if len(values) > 1 and values[-1] > values[-2] else 'down',
                                    'history': values[-12:],  # 최근 12개월
                                    'unit': rows[0].get('UNIT_NAME', '') if rows else ''
                                }
                            else:
                                indicators[name] = self._get_sample_indicator(name)
                        else:
                            indicators[name] = self._get_sample_indicator(name)
                    else:
                        indicators[name] = self._get_sample_indicator(name)
                    
                    time.sleep(0.1)  # API 호출 제한
                    
                except Exception as e:
                    print(f"경제지표 {name} 수집 실패: {e}")
                    indicators[name] = self._get_sample_indicator(name)
            
            return indicators
            
        except Exception as e:
            print(f"경제지표 수집 중 오류: {e}")
            return self._get_sample_economic_data()
    
    def get_etf_detailed_info(self, etf_code):
        """특정 ETF 상세 정보"""
        try:
            # ETF 기본 정보
            today = datetime.now().strftime('%Y%m%d')
            
            # 포트폴리오 구성
            portfolio = stock.get_etf_portfolio_deposit_file(etf_code)
            
            # 최근 가격 정보
            ohlcv = stock.get_etf_ohlcv_by_date(
                (datetime.now() - timedelta(days=5)).strftime('%Y%m%d'),
                today,
                etf_code
            )
            
            # 수수료 정보 (PyKRX에서 직접 제공하지 않으므로 기본값 사용)
            fee_info = {
                'management_fee': 0.005,  # 0.5% (기본값)
                'total_expense_ratio': 0.006  # 0.6% (기본값)
            }
            
            return {
                'portfolio': portfolio,
                'recent_prices': ohlcv,
                'fees': fee_info
            }
            
        except Exception as e:
            print(f"ETF {etf_code} 상세 정보 수집 실패: {e}")
            return None
    
    def _generate_sample_etf_data(self, code, name, category):
        """개별 ETF 샘플 데이터 생성"""
        dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
        
        # 카테고리별 다른 특성 부여
        if '주식' in category:
            base_return = 0.0008
            volatility = 0.02
            base_price = 10000
        elif '채권' in category:
            base_return = 0.0003
            volatility = 0.008
            base_price = 100000
        else:
            base_return = 0.0005
            volatility = 0.015
            base_price = 50000
        
        # 가격 시뮬레이션
        returns = np.random.normal(base_return, volatility, 252)
        prices = base_price * np.exp(np.cumsum(returns))
        volumes = np.random.randint(10000, 100000, 252)
        
        history_df = pd.DataFrame({
            'Open': prices * 0.999,
            'High': prices * 1.002,
            'Low': prices * 0.998,
            'Close': prices,
            'Volume': volumes
        }, index=dates)
        
        return {
            'code': code,
            'price': prices[-1],
            'returns': pd.Series(returns[1:], index=dates[1:]),
            'volume': volumes[-1],
            'history': history_df,
            'portfolio_composition': pd.DataFrame(),
            'category': category
        }
    
    def _get_sample_etf_data(self):
        """전체 샘플 ETF 데이터 생성"""
        sample_data = {}
        
        for category, etfs in ETF_CODES.items():
            category_data = {}
            for code, name in etfs.items():
                category_data[name] = self._generate_sample_etf_data(code, name, category)
            sample_data[category] = category_data
        
        return sample_data
    
    def _get_sample_economic_data(self):
        """샘플 경제지표 데이터 생성"""
        return {
            'GDP': {
                'current': 3.2, 'previous': 3.0, 'trend': 'up', 
                'history': [2.8, 2.9, 3.0, 3.1, 3.2], 'unit': '%'
            },
            'CPI': {
                'current': 2.1, 'previous': 2.3, 'trend': 'down', 
                'history': [2.5, 2.4, 2.3, 2.2, 2.1], 'unit': '%'
            },
            'PPI': {
                'current': 1.8, 'previous': 1.9, 'trend': 'down', 
                'history': [2.0, 1.9, 1.9, 1.8, 1.8], 'unit': '%'
            },
            'INTEREST_RATE': {
                'current': 3.5, 'previous': 3.25, 'trend': 'up', 
                'history': [3.0, 3.25, 3.25, 3.5, 3.5], 'unit': '%'
            },
            'EXCHANGE_RATE': {
                'current': 1320, 'previous': 1310, 'trend': 'up', 
                'history': [1300, 1305, 1310, 1315, 1320], 'unit': '원/달러'
            },
            'MONEY_SUPPLY': {
                'current': 2850, 'previous': 2830, 'trend': 'up', 
                'history': [2800, 2820, 2830, 2840, 2850], 'unit': '조원'
            },
            'EMPLOYMENT': {
                'current': 67.2, 'previous': 67.0, 'trend': 'up', 
                'history': [66.8, 66.9, 67.0, 67.1, 67.2], 'unit': '%'
            },
            'INDUSTRIAL_PRODUCTION': {
                'current': 105.2, 'previous': 104.8, 'trend': 'up', 
                'history': [104.0, 104.5, 104.8, 105.0, 105.2], 'unit': '지수'
            }
        }
    
    def _get_sample_indicator(self, name):
        """개별 지표 샘플 데이터"""
        base_values = {
            'GDP': 3.0, 'CPI': 2.0, 'PPI': 1.5, 'INTEREST_RATE': 3.0,
            'EXCHANGE_RATE': 1300, 'MONEY_SUPPLY': 2800, 'EMPLOYMENT': 67.0,
            'INDUSTRIAL_PRODUCTION': 105.0
        }
        
        base_value = base_values.get(name, 100)
        current = base_value + np.random.normal(0, base_value * 0.02)
        previous = base_value + np.random.normal(0, base_value * 0.02)
        
        return {
            'current': current,
            'previous': previous,
            'trend': 'up' if current > previous else 'down',
            'history': [base_value + np.random.normal(0, base_value * 0.02) for _ in range(12)],
            'unit': '원/달러' if name == 'EXCHANGE_RATE' else ('조원' if name == 'MONEY_SUPPLY' else '%')
        }
