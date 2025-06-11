# agents/risk_manager.py
import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class RiskManagerAgent:
    def __init__(self):
        self.risk_free_rate = 0.02  # 2% 무위험 수익률
    
    def analyze_portfolio_risk(self, historical_data: Dict[str, pd.DataFrame], 
                             user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """포트폴리오 리스크 분석"""
        try:
            # 가격 데이터 결합
            price_data = self._combine_price_data(historical_data)
            
            if price_data.empty:
                return {'error': '충분한 가격 데이터가 없습니다'}
            
            # 기대 수익률 및 공분산 계산
            mu = expected_returns.mean_historical_return(price_data)
            S = risk_models.sample_cov(price_data)
            
            # 포트폴리오 최적화
            optimization_results = self._optimize_portfolio(mu, S, user_profile)
            
            # 리스크 메트릭 계산
            risk_metrics = self._calculate_risk_metrics(price_data, optimization_results['weights'])
            
            # VaR 계산
            var_analysis = self._calculate_var(price_data, optimization_results['weights'])
            
            return {
                'optimization': optimization_results,
                'risk_metrics': risk_metrics,
                'var_analysis': var_analysis,
                'correlation_matrix': S.corr().to_dict()
            }
            
        except Exception as e:
            logger.error(f"포트폴리오 리스크 분석 오류: {e}")
            return {'error': str(e)}
    
    def _combine_price_data(self, historical_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """가격 데이터 결합"""
        price_data = pd.DataFrame()
        
        for ticker, data in historical_data.items():
            if not data.empty and 'Close' in data.columns:
                price_data[ticker] = data['Close']
        
        # 결측값 처리
        price_data = price_data.dropna()
        
        return price_data
    
    def _optimize_portfolio(self, mu: pd.Series, S: pd.DataFrame, 
                          user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """포트폴리오 최적화"""
        try:
            ef = EfficientFrontier(mu, S)
            
            # 사용자 위험 성향에 따른 최적화
            risk_tolerance = user_profile.get('risk_tolerance', 'moderate')
            
            if risk_tolerance == 'conservative':
                # 최소 변동성 포트폴리오
                ef.min_volatility()
            elif risk_tolerance == 'aggressive':
                # 최대 샤프 비율 포트폴리오
                ef.max_sharpe(risk_free_rate=self.risk_free_rate)
            else:  # moderate
                # 효율적 리스크 포트폴리오
                target_return = mu.mean()
                ef.efficient_return(target_return)
            
            weights = ef.clean_weights()
            performance = ef.portfolio_performance(
                risk_free_rate=self.risk_free_rate,
                verbose=False
            )
            
            return {
                'weights': weights,
                'expected_return': performance[0],
                'volatility': performance[1],
                'sharpe_ratio': performance[2]
            }
            
        except Exception as e:
            logger.error(f"포트폴리오 최적화 오류: {e}")
            # 균등 가중 포트폴리오로 대체
            tickers = list(mu.index)
            equal_weights = {ticker: 1/len(tickers) for ticker in tickers}
            return {
                'weights': equal_weights,
                'expected_return': mu.mean(),
                'volatility': 0.15,  # 추정값
                'sharpe_ratio': 0.5   # 추정값
            }
    
    def _calculate_risk_metrics(self, price_data: pd.DataFrame, 
                              weights: Dict[str, float]) -> Dict[str, Any]:
        """리스크 메트릭 계산"""
        returns = price_data.pct_change().dropna()
        
        # 포트폴리오 수익률 계산
        portfolio_returns = (returns * pd.Series(weights)).sum(axis=1)
        
        metrics = {
            'max_drawdown': self._calculate_max_drawdown(portfolio_returns),
            'beta': self._calculate_portfolio_beta(returns, weights),
            'tracking_error': portfolio_returns.std() * np.sqrt(252),
            'information_ratio': self._calculate_information_ratio(portfolio_returns),
            'sortino_ratio': self._calculate_sortino_ratio(portfolio_returns)
        }
        
        return metrics
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """최대 낙폭 계산"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _calculate_portfolio_beta(self, returns: pd.DataFrame, 
                                weights: Dict[str, float]) -> float:
        """포트폴리오 베타 계산 (SPY 대비)"""
        # 간단한 베타 계산 (실제로는 시장 지수 필요)
        portfolio_returns = (returns * pd.Series(weights)).sum(axis=1)
        market_returns = returns.mean(axis=1)  # 임시로 평균 사용
        
        covariance = np.cov(portfolio_returns, market_returns)[0][1]
        market_variance = np.var(market_returns)
        
        return covariance / market_variance if market_variance != 0 else 1.0
    
    def _calculate_information_ratio(self, portfolio_returns: pd.Series) -> float:
        """정보 비율 계산"""
        excess_returns = portfolio_returns - self.risk_free_rate/252
        return excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else 0
    
    def _calculate_sortino_ratio(self, portfolio_returns: pd.Series) -> float:
        """소르티노 비율 계산"""
        excess_returns = portfolio_returns - self.risk_free_rate/252
        downside_returns = excess_returns[excess_returns < 0]
        downside_deviation = downside_returns.std()
        
        return excess_returns.mean() / downside_deviation if downside_deviation != 0 else 0
    
    def _calculate_var(self, price_data: pd.DataFrame, 
                      weights: Dict[str, float], confidence: float = 0.05) -> Dict[str, float]:
        """VaR (Value at Risk) 계산"""
        returns = price_data.pct_change().dropna()
        portfolio_returns = (returns * pd.Series(weights)).sum(axis=1)
        
        # Historical VaR
        historical_var = np.percentile(portfolio_returns, confidence * 100)
        
        # Parametric VaR
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std()
        parametric_var = mean_return - (std_return * 1.645)  # 95% 신뢰구간
        
        return {
            'historical_var_5%': historical_var,
            'parametric_var_5%': parametric_var,
            'expected_shortfall': portfolio_returns[portfolio_returns <= historical_var].mean()
        }
