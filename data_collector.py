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
        
    def get_etf_data(self, period='1y', progress_callback=None):
        """PyKRX를 사용한 ETF 데이터 수집 - 2025년 6월까지 반영"""
        etf_data = {}
        
        # 현재 날짜 기준으로 기간 설정
        end_date = datetime.now().strftime('%Y%m%d')  # 2025년 6월 17일
        
        if period == '1y':
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')  # 2024년 6월
        elif period == '6m':
            start_date = (datetime.now() - timedelta(days=180)).strftime('%Y%m%d')  # 2024년 12월
        elif period == '3m':
            start_date = (datetime.now() - timedelta(days=90)).strftime('%Y%m%d')   # 2025년 3월
        elif period == '2y':
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y%m%d')  # 2023년 6월
        else:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        
        print(f"📅 데이터 수집 기간: {start_date} ~ {end_date}")
        
        # 전체 ETF 수 계산
        total_etfs = sum(len(etfs) for etfs in ETF_CODES.values())
        current_etf = 0
        
        try:
            for category, etfs in ETF_CODES.items():
                category_data = {}
                
                for code, name in etfs.items():
                    current_etf += 1
                    
                    if progress_callback:
                        progress_callback(current_etf, total_etfs, f"수집 중: {name}")
                    
                    try:
                        print(f"📊 데이터 수집 중: {name} ({code}) - {start_date}~{end_date}")
                        
                        # PyKRX로 ETF 가격 데이터 수집
                        df = stock.get_etf_ohlcv_by_date(start_date, end_date, code)
                        
                        if df is not None and not df.empty:
                            # 수익률 계산
                            returns = df['종가'].pct_change().dropna()
                            
                            # 최신 정보
                            latest_price = df['종가'].iloc[-1]
                            latest_volume = df['거래량'].iloc[-1]
                            
                            # ETF 기본 정보 수집 시도
                            try:
                                etf_portfolio = stock.get_etf_portfolio_deposit_file(code)
                                portfolio_info = etf_portfolio if etf_portfolio is not None else pd.DataFrame()
                            except:
                                portfolio_info = pd.DataFrame()
                            
                            # 추가 정보 수집
                            try:
                                # 최근 1개월 성과
                                recent_start = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
                                recent_df = stock.get_etf_ohlcv_by_date(recent_start, end_date, code)
                                if recent_df is not None and len(recent_df) > 1:
                                    recent_return = (recent_df['종가'].iloc[-1] / recent_df['종가'].iloc[0] - 1) * 100
                                else:
                                    recent_return = 0
                            except:
                                recent_return = 0
                            
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
                                'category': category,
                                'recent_1m_return': recent_return,
                                'data_start_date': start_date,
                                'data_end_date': end_date,
                                'data_points': len(df)
                            }
                            
                            print(f"✅ {name} 데이터 수집 완료: {len(df)}일 데이터")
                            
                        else:
                            print(f"❌ {name} 데이터 없음 - 샘플 데이터 생성")
                            category_data[name] = self._generate_sample_etf_data(code, name, category, start_date, end_date)
                        
                        # API 호출 제한 방지
                        time.sleep(0.1)
                        
                    except Exception as e:
                        print(f"❌ {name} 데이터 수집 실패: {e}")
                        category_data[name] = self._generate_sample_etf_data(code, name, category, start_date, end_date)
                
                etf_data[category] = category_data
                print(f"📂 {category} 카테고리 완료: {len(category_data)}개 ETF")
                
            print(f"🎯 전체 ETF 데이터 수집 완료: {sum(len(cat) for cat in etf_data.values())}개")
            return etf_data
            
        except Exception as e:
            print(f"ETF 데이터 수집 중 전체 오류: {e}")
            return self._get_sample_etf_data()
    
    def get_market_data(self):
        """시장 전체 데이터 수집 - 2025년 6월까지"""
        try:
            today = datetime.now().strftime('%Y%m%d')  # 20250617
            month_ago = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
            
            print(f"📈 시장 데이터 수집: {month_ago} ~ {today}")
            
            # KOSPI 지수
            kospi = stock.get_index_ohlcv_by_date(month_ago, today, "1001")
            
            # KOSDAQ 지수
            kosdaq = stock.get_index_ohlcv_by_date(month_ago, today, "2001")
            
            # ETF 시장 현황
            etf_list = stock.get_etf_ticker_list(today)
            
            # KRX 300 지수 (추가)
            try:
                krx300 = stock.get_index_ohlcv_by_date(month_ago, today, "1028")
            except:
                krx300 = pd.DataFrame()
            
            market_data = {
                'kospi': {
                    'current': kospi['종가'].iloc[-1] if not kospi.empty else 2600,
                    'change': kospi['종가'].iloc[-1] - kospi['종가'].iloc[-2] if len(kospi) > 1 else 0,
                    'change_pct': ((kospi['종가'].iloc[-1] / kospi['종가'].iloc[-2]) - 1) * 100 if len(kospi) > 1 else 0,
                    'volume': kospi['거래량'].iloc[-1] if not kospi.empty else 0,
                    'data_date': today
                },
                'kosdaq': {
                    'current': kosdaq['종가'].iloc[-1] if not kosdaq.empty else 850,
                    'change': kosdaq['종가'].iloc[-1] - kosdaq['종가'].iloc[-2] if len(kosdaq) > 1 else 0,
                    'change_pct': ((kosdaq['종가'].iloc[-1] / kosdaq['종가'].iloc[-2]) - 1) * 100 if len(kosdaq) > 1 else 0,
                    'volume': kosdaq['거래량'].iloc[-1] if not kosdaq.empty else 0,
                    'data_date': today
                },
                'krx300': {
                    'current': krx300['종가'].iloc[-1] if not krx300.empty else 1800,
                    'change': krx300['종가'].iloc[-1] - krx300['종가'].iloc[-2] if len(krx300) > 1 else 0,
                    'change_pct': ((krx300['종가'].iloc[-1] / krx300['종가'].iloc[-2]) - 1) * 100 if len(krx300) > 1 else 0
                },
                'etf_count': len(etf_list) if etf_list is not None else 600,
                'collection_date': today,
                'market_status': 'active' if datetime.now().weekday() < 5 else 'closed'
            }
            
            print(f"✅ 시장 데이터 수집 완료")
            return market_data
            
        except Exception as e:
            print(f"시장 데이터 수집 실패: {e}")
            return self._get_sample_market_data()
    
    def get_economic_indicators(self):
        """경제지표 데이터 수집 - 2025년 6월까지"""
        if not self.ecos_api_key:
            return self._get_sample_economic_data()
        
        indicators = {}
        
        # 현재 날짜 기준으로 조회 기간 설정
        current_year = datetime.now().year  # 2025
        current_month = datetime.now().month  # 6
        
        # 2년간 데이터 조회 (2023년 6월 ~ 2025년 6월)
        start_period = f"{current_year-2:04d}{current_month:02d}"  # 202306
        end_period = f"{current_year:04d}{current_month:02d}"      # 202506
        
        print(f"📊 경제지표 수집 기간: {start_period} ~ {end_period}")
        
        try:
            for name, code in ECONOMIC_INDICATORS.items():
                try:
                    print(f"📈 경제지표 수집 중: {name} ({code})")
                    
                    # ECOS API 호출 - 최신 기간으로 수정
                    url = f"{self.ecos_url}/StatisticSearch/{self.ecos_api_key}/json/kr/1/100/{code}/M/{start_period}/{end_period}/"
                    response = requests.get(url, timeout=15)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if 'StatisticSearch' in data and 'row' in data['StatisticSearch']:
                            rows = data['StatisticSearch']['row']
                            values = []
                            dates = []
                            
                            for row in rows:
                                try:
                                    if row['DATA_VALUE'] != '-' and row['DATA_VALUE'] != '':
                                        values.append(float(row['DATA_VALUE']))
                                        dates.append(row['TIME'])
                                except:
                                    continue
                            
                            if values and len(values) >= 2:
                                indicators[name] = {
                                    'current': values[-1],
                                    'previous': values[-2],
                                    'trend': 'up' if values[-1] > values[-2] else 'down',
                                    'history': values[-24:],  # 최근 24개월
                                    'dates': dates[-24:],
                                    'unit': rows[0].get('UNIT_NAME', '') if rows else '',
                                    'last_update': dates[-1] if dates else end_period,
                                    'data_count': len(values)
                                }
                                print(f"✅ {name}: {values[-1]} (최신: {dates[-1] if dates else 'N/A'})")
                            else:
                                print(f"⚠️ {name}: 유효한 데이터 없음")
                                indicators[name] = self._get_sample_indicator(name)
                        else:
                            print(f"⚠️ {name}: API 응답 구조 오류")
                            indicators[name] = self._get_sample_indicator(name)
                    else:
                        print(f"⚠️ {name}: API 호출 실패 (상태코드: {response.status_code})")
                        indicators[name] = self._get_sample_indicator(name)
                    
                    time.sleep(0.2)  # API 호출 제한
                    
                except Exception as e:
                    print(f"경제지표 {name} 수집 실패: {e}")
                    indicators[name] = self._get_sample_indicator(name)
            
            print(f"🎯 경제지표 수집 완료: {len(indicators)}개")
            return indicators
            
        except Exception as e:
            print(f"경제지표 수집 중 오류: {e}")
            return self._get_sample_economic_data()
    
    def get_etf_detailed_info(self, etf_code):
        """특정 ETF 상세 정보 - 최신 데이터"""
        try:
            today = datetime.now().strftime('%Y%m%d')
            week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y%m%d')
            
            print(f"🔍 {etf_code} 상세 정보 수집: {week_ago} ~ {today}")
            
            # 포트폴리오 구성
            portfolio = stock.get_etf_portfolio_deposit_file(etf_code)
            
            # 최근 가격 정보
            ohlcv = stock.get_etf_ohlcv_by_date(week_ago, today, etf_code)
            
            # ETF 기본 정보
            try:
                etf_info = stock.get_etf_ticker_list(today)
                etf_name = None
                if etf_info is not None:
                    for ticker in etf_info:
                        if ticker == etf_code:
                            etf_name = stock.get_etf_ticker_name(etf_code)
                            break
            except:
                etf_name = None
            
            # 수수료 정보 (실제 데이터가 없으므로 추정값)
            fee_info = {
                'management_fee': 0.005,  # 0.5% (일반적인 값)
                'total_expense_ratio': 0.006,  # 0.6%
                'estimated': True
            }
            
            return {
                'portfolio': portfolio,
                'recent_prices': ohlcv,
                'fees': fee_info,
                'etf_name': etf_name,
                'collection_date': today,
                'data_period': f"{week_ago}~{today}"
            }
            
        except Exception as e:
            print(f"ETF {etf_code} 상세 정보 수집 실패: {e}")
            return None
    
    def _generate_sample_etf_data(self, code, name, category, start_date, end_date):
        """개별 ETF 샘플 데이터 생성 - 2025년 기준"""
        # 날짜 범위 계산
        start_dt = datetime.strptime(start_date, '%Y%m%d')
        end_dt = datetime.strptime(end_date, '%Y%m%d')
        date_range = pd.date_range(start=start_dt, end=end_dt, freq='D')
        business_days = pd.bdate_range(start=start_dt, end=end_dt)
        
        # 카테고리별 다른 특성 부여 (2025년 시장 환경 반영)
        if '주식' in category:
            base_return = 0.0008  # 연 20% 정도
            volatility = 0.025    # 변동성 증가
            base_price = 12000    # 2025년 기준 가격
        elif '채권' in category:
            base_return = 0.0004  # 연 10% 정도
            volatility = 0.012    # 낮은 변동성
            base_price = 105000   # 채권 ETF 가격
        elif '섹터' in category:
            base_return = 0.001   # 연 25% (성장 섹터)
            volatility = 0.03     # 높은 변동성
            base_price = 8000
        else:  # 대안투자
            base_return = 0.0006  # 연 15%
            volatility = 0.02
            base_price = 55000
        
        # 가격 시뮬레이션
        num_days = len(business_days)
        returns = np.random.normal(base_return, volatility, num_days)
        prices = base_price * np.exp(np.cumsum(returns))
        volumes = np.random.randint(50000, 200000, num_days)  # 2025년 거래량 증가
        
        history_df = pd.DataFrame({
            'Open': prices * 0.999,
            'High': prices * 1.003,
            'Low': prices * 0.997,
            'Close': prices,
            'Volume': volumes
        }, index=business_days)
        
        return {
            'code': code,
            'price': prices[-1],
            'returns': pd.Series(returns[1:], index=business_days[1:]),
            'volume': volumes[-1],
            'history': history_df,
            'portfolio_composition': pd.DataFrame(),
            'category': category,
            'recent_1m_return': np.random.uniform(-5, 8),  # 2025년 시장 환경
            'data_start_date': start_date,
            'data_end_date': end_date,
            'data_points': num_days,
            'is_sample': True
        }
    
    def _get_sample_etf_data(self):
        """전체 샘플 ETF 데이터 생성"""
        sample_data = {}
        
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        
        for category, etfs in ETF_CODES.items():
            category_data = {}
            for code, name in etfs.items():
                category_data[name] = self._generate_sample_etf_data(code, name, category, start_date, end_date)
            sample_data[category] = category_data
        
        return sample_data
    
    def _get_sample_market_data(self):
        """샘플 시장 데이터 - 2025년 6월 기준"""
        today = datetime.now().strftime('%Y%m%d')
        
        return {
            'kospi': {
                'current': 2650, 'change': 15, 'change_pct': 0.57,
                'volume': 450000000, 'data_date': today
            },
            'kosdaq': {
                'current': 880, 'change': 8, 'change_pct': 0.92,
                'volume': 280000000, 'data_date': today
            },
            'krx300': {
                'current': 1850, 'change': 12, 'change_pct': 0.65
            },
            'etf_count': 650,
            'collection_date': today,
            'market_status': 'active'
        }
    
    def _get_sample_economic_data(self):
        """샘플 경제지표 데이터 - 2025년 6월 기준"""
        current_period = datetime.now().strftime('%Y%m')
        
        return {
            'GDP': {
                'current': 3.4, 'previous': 3.2, 'trend': 'up', 
                'history': [2.8, 2.9, 3.0, 3.1, 3.2, 3.4], 'unit': '%',
                'last_update': current_period, 'data_count': 24
            },
            'CPI': {
                'current': 2.3, 'previous': 2.1, 'trend': 'up', 
                'history': [2.0, 2.1, 2.1, 2.2, 2.1, 2.3], 'unit': '%',
                'last_update': current_period, 'data_count': 24
            },
            'PPI': {
                'current': 1.9, 'previous': 1.8, 'trend': 'up', 
                'history': [1.5, 1.6, 1.7, 1.8, 1.8, 1.9], 'unit': '%',
                'last_update': current_period, 'data_count': 24
            },
            'INTEREST_RATE': {
                'current': 3.25, 'previous': 3.5, 'trend': 'down', 
                'history': [3.5, 3.5, 3.5, 3.5, 3.5, 3.25], 'unit': '%',
                'last_update': current_period, 'data_count': 24
            },
            'EXCHANGE_RATE': {
                'current': 1280, 'previous': 1320, 'trend': 'down', 
                'history': [1350, 1340, 1330, 1320, 1320, 1280], 'unit': '원/달러',
                'last_update': current_period, 'data_count': 24
            },
            'MONEY_SUPPLY': {
                'current': 2950, 'previous': 2850, 'trend': 'up', 
                'history': [2700, 2750, 2800, 2820, 2850, 2950], 'unit': '조원',
                'last_update': current_period, 'data_count': 24
            },
            'EMPLOYMENT': {
                'current': 67.8, 'previous': 67.2, 'trend': 'up', 
                'history': [66.5, 66.8, 67.0, 67.1, 67.2, 67.8], 'unit': '%',
                'last_update': current_period, 'data_count': 24
            },
            'INDUSTRIAL_PRODUCTION': {
                'current': 106.2, 'previous': 105.2, 'trend': 'up', 
                'history': [103.0, 104.0, 104.5, 105.0, 105.2, 106.2], 'unit': '지수',
                'last_update': current_period, 'data_count': 24
            }
        }
    
    def _get_sample_indicator(self, name):
        """개별 지표 샘플 데이터 - 2025년 기준"""
        current_period = datetime.now().strftime('%Y%m')
        
        # 2025년 경제 환경을 반영한 기본값
        base_values = {
            'GDP': 3.2, 'CPI': 2.2, 'PPI': 1.8, 'INTEREST_RATE': 3.25,
            'EXCHANGE_RATE': 1280, 'MONEY_SUPPLY': 2900, 'EMPLOYMENT': 67.5,
            'INDUSTRIAL_PRODUCTION': 106.0
        }
        
        base_value = base_values.get(name, 100)
        current = base_value + np.random.normal(0, base_value * 0.03)
        previous = base_value + np.random.normal(0, base_value * 0.03)
        
        return {
            'current': current,
            'previous': previous,
            'trend': 'up' if current > previous else 'down',
            'history': [base_value + np.random.normal(0, base_value * 0.03) for _ in range(24)],
            'unit': '원/달러' if name == 'EXCHANGE_RATE' else ('조원' if name == 'MONEY_SUPPLY' else '%'),
            'last_update': current_period,
            'data_count': 24,
            'is_sample': True
        }
