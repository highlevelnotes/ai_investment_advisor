# portfolio_optimizer.py
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from config import PORTFOLIO_SETTINGS
from utils import calculate_portfolio_performance

class PortfolioOptimizer:
    def __init__(self):
        self.risk_free_rate = PORTFOLIO_SETTINGS['RISK_FREE_RATE']
        self.transaction_cost = PORTFOLIO_SETTINGS['TRANSACTION_COST']
        
    def optimize_portfolio(self, returns_data, method='ai_based', user_profile=None, ai_analyzer=None, macro_data=None):
        """포트폴리오 최적화 - AI 기반만 지원"""
        if method != 'ai_based' or not ai_analyzer:
            print("⚠️ AI 기반 최적화만 지원됩니다.")
            return self._get_equal_weight_portfolio(returns_data)
        
        try:
            print("🤖 AI 기반 포트폴리오 최적화 시작")
            
            # AI 종합 분석 실행
            comprehensive_result = ai_analyzer.comprehensive_market_analysis(
                macro_data, returns_data, user_profile
            )
            
            if not comprehensive_result or not comprehensive_result.get('portfolio', {}).get('weights'):
                print("❌ AI 분석 결과가 없습니다. 기본 포트폴리오를 사용합니다.")
                return self._get_equal_weight_portfolio(returns_data)
            
            weights = comprehensive_result['portfolio']['weights']
            
            # 실제 성과 계산
            performance = calculate_portfolio_performance(weights, returns_data, self.risk_free_rate)
            
            result = {
                'weights': weights,
                'expected_return': performance['expected_return'],
                'volatility': performance['volatility'],
                'sharpe_ratio': performance['sharpe_ratio'],
                'max_drawdown': performance['max_drawdown'],
                'method': 'ai_based',
                'ai_analysis': comprehensive_result['analysis'],
                'allocation_reasoning': comprehensive_result['portfolio']['allocation_reasoning'],
                'data_points': performance.get('data_points', 0),
                'is_sample': performance.get('is_sample', False)
            }
            
            print(f"✅ AI 최적화 완료: {len(weights)}개 ETF")
            return result
            
        except Exception as e:
            print(f"❌ AI 기반 최적화 실패: {e}")
            return self._get_equal_weight_portfolio(returns_data)
    
    def _get_equal_weight_portfolio(self, returns_data):
        """균등가중 포트폴리오 (폴백용)"""
        if isinstance(returns_data, dict):
            asset_names = []
            for category_etfs in returns_data.values():
                asset_names.extend(category_etfs.keys())
        else:
            asset_names = list(returns_data)
        
        if not asset_names:
            return {
                'weights': {},
                'expected_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'method': 'equal_weight'
            }
        
        n_assets = len(asset_names)
        equal_weight = 1.0 / n_assets
        weights = {name: equal_weight for name in asset_names}
        
        # 성과 계산
        performance = calculate_portfolio_performance(weights, returns_data, self.risk_free_rate)
        
        return {
            'weights': weights,
            'expected_return': performance['expected_return'],
            'volatility': performance['volatility'],
            'sharpe_ratio': performance['sharpe_ratio'],
            'max_drawdown': performance['max_drawdown'],
            'method': 'equal_weight'
        }
