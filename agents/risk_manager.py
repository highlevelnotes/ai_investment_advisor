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
    
    # agents/risk_manager.py 수정
    def analyze_portfolio_risk(self, historical_data: Dict[str, Any], 
                            user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """포트폴리오 리스크 분석"""
        try:
            logger.info(f"리스크 분석 시작: {len(historical_data)}개 종목")
            
            # 가격 데이터 결합
            price_data = self._combine_price_data(historical_data)
            
            # 수정: DataFrame.empty 사용
            if price_data.empty:
                logger.warning("가격 데이터가 없어 기본 리스크 분석 반환")
                return self._get_default_risk_analysis()
            
            # 최소 데이터 요구사항 확인
            if len(price_data.columns) < 2:
                logger.warning("분석을 위한 최소 종목 수 부족")
                return self._get_default_risk_analysis()
            
            if len(price_data) < 30:
                logger.warning("분석을 위한 최소 데이터 포인트 부족")
                return self._get_default_risk_analysis()
            
            logger.info(f"가격 데이터 변환 완료: {price_data.shape}")
            
            # 기대 수익률 및 공분산 계산
            mu = expected_returns.mean_historical_return(price_data)
            S = risk_models.sample_cov(price_data)
            
            # 포트폴리오 최적화
            optimization_results = self._optimize_portfolio(mu, S, user_profile)
            
            # 리스크 메트릭 계산
            risk_metrics = self._calculate_risk_metrics(price_data, optimization_results['weights'])
            
            # VaR 계산
            var_analysis = self._calculate_var(price_data, optimization_results['weights'])
            
            result = {
                'optimization': optimization_results,
                'risk_metrics': risk_metrics,
                'var_analysis': var_analysis,
                'correlation_matrix': S.corr().to_dict(),
                'data_quality': {
                    'tickers_analyzed': len(price_data.columns),
                    'data_points': len(price_data),
                    'date_range': f"{price_data.index[0].strftime('%Y-%m-%d')} ~ {price_data.index[-1].strftime('%Y-%m-%d')}"
                }
            }
            
            logger.info("리스크 분석 완료")
            return result
            
        except Exception as e:
            logger.error(f"포트폴리오 리스크 분석 오류: {e}")
            return self._get_default_risk_analysis()

    def _combine_price_data(self, historical_data: Dict[str, Any]) -> pd.DataFrame:
        """가격 데이터 결합 (오류 수정)"""
        price_data = pd.DataFrame()
        
        for ticker, data in historical_data.items():
            # 수정: 올바른 조건 확인
            if not data or not isinstance(data, dict) or 'Close' not in data:
                logger.warning(f"{ticker}: 유효하지 않은 데이터 형식")
                continue
                
            try:
                # Dict 데이터를 Series로 변환
                dates = pd.to_datetime(data['dates'])
                close_prices = pd.Series(data['Close'], index=dates)
                
                # 수정: Series.empty 사용
                if close_prices.empty:
                    logger.warning(f"{ticker}: 빈 가격 데이터")
                    continue
                    
                price_data[ticker] = close_prices
                
            except Exception as e:
                logger.warning(f"{ticker} 가격 데이터 변환 실패: {e}")
                continue
        
        # 결측값 처리
        if not price_data.empty:
            price_data = price_data.dropna()
        
        return price_data

    def _optimize_portfolio(self, mu: pd.Series, S: pd.DataFrame, 
                          user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """포트폴리오 최적화 (오류 처리 개선)"""
        try:
            # 데이터 유효성 확인
            if mu.empty or S.empty:
                logger.warning("수익률 또는 공분산 데이터가 비어있음")
                return self._get_default_optimization()
            
            ef = EfficientFrontier(mu, S)
            
            # 사용자 위험 성향에 따른 최적화
            risk_tolerance = user_profile.get('risk_tolerance', 'moderate')
            
            if risk_tolerance == 'conservative':
                ef.min_volatility()
            elif risk_tolerance == 'aggressive':
                ef.max_sharpe(risk_free_rate=self.risk_free_rate)
            else:
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
            return self._get_default_optimization()

    def _get_default_optimization(self) -> Dict[str, Any]:
        """기본 최적화 결과"""
        return {
            'weights': {},
            'expected_return': 0.08,
            'volatility': 0.15,
            'sharpe_ratio': 0.5
        }

    def _calculate_risk_metrics(self, price_data: pd.DataFrame, 
                              weights: Dict[str, float]) -> Dict[str, Any]:
        """리스크 메트릭 계산 (오류 처리 개선)"""
        try:
            # 데이터 유효성 확인
            if price_data.empty or not weights:
                logger.warning("가격 데이터 또는 가중치가 비어있음")
                return self._get_default_risk_metrics()
            
            returns = price_data.pct_change().dropna()
            
            # 수정: returns.empty 확인
            if returns.empty:
                logger.warning("수익률 데이터가 비어있음")
                return self._get_default_risk_metrics()
            
            # 포트폴리오 수익률 계산
            portfolio_returns = (returns * pd.Series(weights)).sum(axis=1)
            
            metrics = {
                'max_drawdown': float(self._calculate_max_drawdown(portfolio_returns)),
                'beta': float(self._calculate_portfolio_beta(returns, weights)),
                'tracking_error': float(portfolio_returns.std() * np.sqrt(252)),
                'information_ratio': float(self._calculate_information_ratio(portfolio_returns)),
                'sortino_ratio': float(self._calculate_sortino_ratio(portfolio_returns))
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"리스크 메트릭 계산 오류: {e}")
            return self._get_default_risk_metrics()

    def _get_default_risk_metrics(self) -> Dict[str, Any]:
        """기본 리스크 메트릭"""
        return {
            'max_drawdown': -0.10,
            'beta': 1.0,
            'tracking_error': 0.05,
            'information_ratio': 0.3,
            'sortino_ratio': 0.6
        }
