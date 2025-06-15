# portfolio_optimizer.py
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from config import LIFECYCLE_ALLOCATION, RISK_ALLOCATION, PORTFOLIO_SETTINGS

class PortfolioOptimizer:
    def __init__(self):
        self.risk_free_rate = PORTFOLIO_SETTINGS['RISK_FREE_RATE']
        self.transaction_cost = PORTFOLIO_SETTINGS['TRANSACTION_COST']
        
    def optimize_portfolio(self, returns_data, method='max_sharpe', user_profile=None):
        """포트폴리오 최적화"""
        returns_df = self._prepare_returns_data(returns_data)
        
        if returns_df.empty:
            return self._get_equal_weight_portfolio(returns_data)
        
        if method == 'min_variance':
            return self._minimize_variance(returns_df)
        elif method == 'max_sharpe':
            return self._maximize_sharpe_ratio(returns_df)
        elif method == 'lifecycle' and user_profile:
            return self._lifecycle_allocation(returns_df, user_profile)
        else:
            return self._get_equal_weight_portfolio(returns_data)
    
    def _prepare_returns_data(self, returns_data):
        """수익률 데이터 준비"""
        returns_dict = {}
        
        for category, etfs in returns_data.items():
            for name, data in etfs.items():
                if 'returns' in data and not data['returns'].empty:
                    returns_dict[name] = data['returns']
        
        if not returns_dict:
            return pd.DataFrame()
        
        returns_df = pd.DataFrame(returns_dict)
        returns_df = returns_df.dropna()
        
        return returns_df
    
    def _minimize_variance(self, returns_df):
        """최소분산 포트폴리오"""
        n_assets = len(returns_df.columns)
        
        # 공분산 행렬 추정 (Ledoit-Wolf 수축 추정량 사용)
        lw = LedoitWolf()
        cov_matrix = lw.fit(returns_df).covariance_
        
        # 목적함수: 포트폴리오 분산 최소화
        def objective(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        # 제약조건
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # 가중치 합 = 1
        ]
        
        # 경계조건
        bounds = [(PORTFOLIO_SETTINGS['MIN_WEIGHT'], PORTFOLIO_SETTINGS['MAX_WEIGHT']) 
                 for _ in range(n_assets)]
        
        # 초기값
        x0 = np.array([1/n_assets] * n_assets)
        
        # 최적화 실행
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            weights = dict(zip(returns_df.columns, result.x))
            portfolio_return = np.sum(returns_df.mean() * result.x) * 252
            portfolio_vol = np.sqrt(objective(result.x)) * np.sqrt(252)
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol
            
            return {
                'weights': weights,
                'expected_return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe_ratio': sharpe_ratio,
                'method': 'min_variance'
            }
        else:
            return self._get_equal_weight_portfolio(returns_df.columns)
    
    def _maximize_sharpe_ratio(self, returns_df):
        """최대 샤프비율 포트폴리오"""
        n_assets = len(returns_df.columns)
        
        # 공분산 행렬과 기대수익률
        lw = LedoitWolf()
        cov_matrix = lw.fit(returns_df).covariance_
        expected_returns = returns_df.mean() * 252
        
        # 목적함수: 음의 샤프비율 (최소화를 위해)
        def objective(weights):
            portfolio_return = np.sum(expected_returns * weights)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)) * 252)
            if portfolio_vol == 0:
                return -np.inf
            return -(portfolio_return - self.risk_free_rate) / portfolio_vol
        
        # 제약조건
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        ]
        
        # 경계조건
        bounds = [(PORTFOLIO_SETTINGS['MIN_WEIGHT'], PORTFOLIO_SETTINGS['MAX_WEIGHT']) 
                 for _ in range(n_assets)]
        
        # 초기값
        x0 = np.array([1/n_assets] * n_assets)
        
        # 최적화 실행
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            weights = dict(zip(returns_df.columns, result.x))
            portfolio_return = np.sum(expected_returns * result.x)
            portfolio_vol = np.sqrt(np.dot(result.x.T, np.dot(cov_matrix, result.x)) * 252)
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol
            
            return {
                'weights': weights,
                'expected_return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe_ratio': sharpe_ratio,
                'method': 'max_sharpe'
            }
        else:
            return self._get_equal_weight_portfolio(returns_df.columns)
    
    def _lifecycle_allocation(self, returns_df, user_profile):
        """생애주기별 자산배분"""
        age = user_profile.get('age', 30)
        risk_tolerance = user_profile.get('risk_tolerance', '위험중립형')
        
        # 생애주기 분류
        if age < 40:
            lifecycle_stage = '청년층'
        elif age < 55:
            lifecycle_stage = '중년층'
        else:
            lifecycle_stage = '장년층'
        
        # 기본 배분 비율
        base_allocation = LIFECYCLE_ALLOCATION[lifecycle_stage]
        risk_allocation = RISK_ALLOCATION[risk_tolerance]
        
        # 가중평균으로 최종 배분 결정
        final_allocation = {}
        for asset_class in base_allocation.keys():
            final_allocation[asset_class] = (
                base_allocation[asset_class] * 0.7 + 
                risk_allocation[asset_class] * 0.3
            )
        
        # ETF별 가중치 계산
        weights = {}
        total_weight = 0
        
        for etf_name in returns_df.columns:
            # ETF 카테고리 매핑 (간단한 규칙 기반)
            if any(keyword in etf_name for keyword in ['KODEX 200', 'TIGER', '반도체', '바이오']):
                category_weight = final_allocation['주식']
            elif any(keyword in etf_name for keyword in ['국고채', '회사채', '단기채권']):
                category_weight = final_allocation['채권']
            else:
                category_weight = final_allocation['대안투자']
            
            # 카테고리 내 균등분배 (실제로는 더 정교한 로직 필요)
            category_etfs = [name for name in returns_df.columns 
                           if self._get_etf_category(name) == self._get_etf_category(etf_name)]
            
            weights[etf_name] = category_weight / len(category_etfs)
            total_weight += weights[etf_name]
        
        # 가중치 정규화
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        # 성과 지표 계산
        weight_array = np.array([weights[col] for col in returns_df.columns])
        expected_returns = returns_df.mean() * 252
        cov_matrix = returns_df.cov() * 252
        
        portfolio_return = np.sum(expected_returns * weight_array)
        portfolio_vol = np.sqrt(np.dot(weight_array.T, np.dot(cov_matrix, weight_array)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
        
        return {
            'weights': weights,
            'expected_return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe_ratio,
            'method': 'lifecycle',
            'lifecycle_stage': lifecycle_stage,
            'allocation': final_allocation
        }
    
    def _get_etf_category(self, etf_name):
        """ETF 카테고리 분류"""
        if any(keyword in etf_name for keyword in ['KODEX 200', 'TIGER', '반도체', '바이오', '2차전지']):
            return '주식'
        elif any(keyword in etf_name for keyword in ['국고채', '회사채', '단기채권']):
            return '채권'
        else:
            return '대안투자'
    
    def _get_equal_weight_portfolio(self, assets):
        """균등가중 포트폴리오"""
        if isinstance(assets, pd.Index):
            asset_names = assets.tolist()
        elif isinstance(assets, dict):
            asset_names = []
            for category_etfs in assets.values():
                asset_names.extend(category_etfs.keys())
        else:
            asset_names = list(assets)
        
        n_assets = len(asset_names)
        equal_weight = 1.0 / n_assets
        
        weights = {name: equal_weight for name in asset_names}
        
        return {
            'weights': weights,
            'expected_return': 0.06,  # 기본값
            'volatility': 0.15,       # 기본값
            'sharpe_ratio': 0.4,      # 기본값
            'method': 'equal_weight'
        }
    
    def calculate_efficient_frontier(self, returns_df, n_portfolios=100):
        """효율적 프론티어 계산"""
        if returns_df.empty:
            return pd.DataFrame()
        
        n_assets = len(returns_df.columns)
        
        # 공분산 행렬과 기대수익률
        lw = LedoitWolf()
        cov_matrix = lw.fit(returns_df).covariance_ * 252
        expected_returns = returns_df.mean() * 252
        
        # 목표 수익률 범위 설정
        min_ret = expected_returns.min()
        max_ret = expected_returns.max()
        target_returns = np.linspace(min_ret, max_ret, n_portfolios)
        
        efficient_portfolios = []
        
        for target_return in target_returns:
            # 목적함수: 포트폴리오 분산 최소화
            def objective(weights):
                return np.dot(weights.T, np.dot(cov_matrix, weights))
            
            # 제약조건
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # 가중치 합 = 1
                {'type': 'eq', 'fun': lambda x: np.sum(expected_returns * x) - target_return}  # 목표 수익률
            ]
            
            # 경계조건
            bounds = [(0, 1) for _ in range(n_assets)]
            
            # 초기값
            x0 = np.array([1/n_assets] * n_assets)
            
            # 최적화 실행
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                portfolio_vol = np.sqrt(objective(result.x))
                efficient_portfolios.append({
                    'Return': target_return,
                    'Volatility': portfolio_vol,
                    'Sharpe_Ratio': (target_return - self.risk_free_rate) / portfolio_vol
                })
        
        return pd.DataFrame(efficient_portfolios)
    
    def monte_carlo_simulation(self, returns_df, n_simulations=10000):
        """몬테카를로 시뮬레이션"""
        if returns_df.empty:
            return pd.DataFrame()
        
        n_assets = len(returns_df.columns)
        expected_returns = returns_df.mean() * 252
        cov_matrix = returns_df.cov() * 252
        
        results = []
        
        for _ in range(n_simulations):
            # 랜덤 가중치 생성
            weights = np.random.random(n_assets)
            weights /= np.sum(weights)
            
            # 포트폴리오 성과 계산
            portfolio_return = np.sum(expected_returns * weights)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol
            
            results.append({
                'Return': portfolio_return,
                'Volatility': portfolio_vol,
                'Sharpe_Ratio': sharpe_ratio
            })
        
        return pd.DataFrame(results)
