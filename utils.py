# utils.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def format_currency(amount, currency='KRW'):
    """금액을 통화 단위에 맞게 포맷팅하는 함수"""
    if currency == 'KRW':
        return f"{amount:,.0f}원"
    else:
        return f"{amount:,.2f} {currency}"

def calculate_portfolio_metrics(returns, weights, risk_free_rate=0.025):
    """포트폴리오 성과 지표 계산"""
    if isinstance(returns, dict):
        # ETF 데이터 형태인 경우
        returns_series = []
        weight_list = []
        
        for category, etfs in returns.items():
            for name, data in etfs.items():
                if name in weights and 'returns' in data:
                    returns_series.append(data['returns'])
                    weight_list.append(weights[name])
        
        if not returns_series:
            return {}
        
        # 공통 날짜로 정렬
        returns_df = pd.concat(returns_series, axis=1, keys=range(len(returns_series)))
        returns_df = returns_df.dropna()
        
        if returns_df.empty:
            return {}
        
        weight_array = np.array(weight_list)
        weight_array = weight_array / weight_array.sum()  # 정규화
        
        portfolio_returns = returns_df.dot(weight_array)
        
    else:
        # 이미 포트폴리오 수익률인 경우
        portfolio_returns = returns
    
    if len(portfolio_returns) == 0:
        return {}
    
    # 연환산 수익률
    annual_return = portfolio_returns.mean() * 252
    
    # 연환산 변동성
    annual_volatility = portfolio_returns.std() * np.sqrt(252)
    
    # 샤프 비율
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
    
    # 최대 낙폭 (Maximum Drawdown)
    cumulative_returns = (1 + portfolio_returns).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # VaR (Value at Risk) - 95% 신뢰수준
    var_95 = np.percentile(portfolio_returns, 5)
    
    # CVaR (Conditional Value at Risk)
    cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
    
    return {
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'total_periods': len(portfolio_returns)
    }

def calculate_correlation_matrix(etf_data):
    """ETF 간 상관관계 매트릭스 계산"""
    returns_data = {}
    
    for category, etfs in etf_data.items():
        for name, data in etfs.items():
            if 'returns' in data and not data['returns'].empty:
                returns_data[name] = data['returns']
    
    if len(returns_data) < 2:
        return pd.DataFrame()
    
    returns_df = pd.DataFrame(returns_data)
    returns_df = returns_df.dropna()
    
    if returns_df.empty:
        return pd.DataFrame()
    
    return returns_df.corr()

def calculate_risk_metrics(returns_series, confidence_level=0.05):
    """리스크 지표 계산"""
    if returns_series.empty:
        return {}
    
    # VaR (Value at Risk)
    var = np.percentile(returns_series, confidence_level * 100)
    
    # CVaR (Conditional Value at Risk)
    cvar = returns_series[returns_series <= var].mean()
    
    # 최대 연속 손실일수
    negative_returns = returns_series < 0
    max_consecutive_losses = 0
    current_consecutive = 0
    
    for is_negative in negative_returns:
        if is_negative:
            current_consecutive += 1
            max_consecutive_losses = max(max_consecutive_losses, current_consecutive)
        else:
            current_consecutive = 0
    
    # 손실 확률
    loss_probability = (returns_series < 0).mean()
    
    # 평균 손실 크기
    losses = returns_series[returns_series < 0]
    avg_loss = losses.mean() if not losses.empty else 0
    
    return {
        'var_95': var,
        'cvar_95': cvar,
        'max_consecutive_losses': max_consecutive_losses,
        'loss_probability': loss_probability,
        'average_loss': avg_loss
    }

def rebalance_portfolio(current_weights, target_weights, threshold=0.05):
    """포트폴리오 리밸런싱 필요성 판단"""
    rebalancing_needed = False
    rebalancing_actions = []
    
    for asset in target_weights.keys():
        current_weight = current_weights.get(asset, 0)
        target_weight = target_weights[asset]
        
        weight_diff = abs(current_weight - target_weight)
        
        if weight_diff > threshold:
            rebalancing_needed = True
            action = "증가" if current_weight < target_weight else "감소"
            rebalancing_actions.append({
                'asset': asset,
                'current_weight': current_weight,
                'target_weight': target_weight,
                'difference': weight_diff,
                'action': action
            })
    
    return {
        'rebalancing_needed': rebalancing_needed,
        'actions': rebalancing_actions
    }

def calculate_compound_return(principal, monthly_contribution, annual_return, years):
    """복리 수익률 계산"""
    monthly_return = annual_return / 12
    months = years * 12
    
    # 원금의 복리 성장
    future_value_principal = principal * (1 + annual_return) ** years
    
    # 월 적립금의 복리 성장
    if monthly_contribution > 0 and monthly_return > 0:
        future_value_contributions = monthly_contribution * (
            ((1 + monthly_return) ** months - 1) / monthly_return
        )
    else:
        future_value_contributions = monthly_contribution * months
    
    total_future_value = future_value_principal + future_value_contributions
    total_contributions = principal + (monthly_contribution * months)
    total_return = total_future_value - total_contributions
    
    return {
        'future_value': total_future_value,
        'total_contributions': total_contributions,
        'total_return': total_return,
        'return_rate': (total_return / total_contributions) * 100 if total_contributions > 0 else 0
    }

def get_lifecycle_stage(age):
    """나이에 따른 생애주기 단계 분류"""
    if age < 35:
        return '청년층'
    elif age < 50:
        return '중년층'
    else:
        return '장년층'

def calculate_optimal_asset_allocation(age, risk_tolerance, investment_period):
    """최적 자산배분 계산"""
    # 기본 주식 비중 (100 - 나이 규칙의 변형)
    base_stock_ratio = max(0.3, min(0.8, (100 - age) / 100))
    
    # 위험성향에 따른 조정
    risk_adjustments = {
        '안정형': -0.2,
        '안정추구형': -0.1,
        '위험중립형': 0.0,
        '적극투자형': 0.1
    }
    
    stock_ratio = base_stock_ratio + risk_adjustments.get(risk_tolerance, 0)
    stock_ratio = max(0.2, min(0.8, stock_ratio))
    
    # 투자기간에 따른 조정
    if investment_period > 20:
        stock_ratio += 0.05
    elif investment_period < 10:
        stock_ratio -= 0.05
    
    stock_ratio = max(0.2, min(0.8, stock_ratio))
    bond_ratio = 0.8 - stock_ratio
    alternative_ratio = 0.2
    
    # 정규화
    total = stock_ratio + bond_ratio + alternative_ratio
    
    return {
        '주식': stock_ratio / total,
        '채권': bond_ratio / total,
        '대안투자': alternative_ratio / total
    }

def format_percentage(value, decimal_places=2):
    """백분율 포맷팅"""
    return f"{value * 100:.{decimal_places}f}%"

def format_number(value, decimal_places=2):
    """숫자 포맷팅"""
    return f"{value:,.{decimal_places}f}"

def calculate_tax_efficiency(returns, holding_period_years):
    """세금 효율성 계산 (간단한 모델)"""
    # 한국 세법 기준 간단 계산
    if holding_period_years >= 1:
        # 1년 이상 보유시 장기보유 특별공제 적용
        tax_rate = 0.154  # 15.4% (지방세 포함)
        if returns > 0:
            taxable_gain = max(0, returns - 2500000)  # 연간 250만원 비과세
            tax_amount = taxable_gain * tax_rate
        else:
            tax_amount = 0
    else:
        # 1년 미만 보유시
        tax_rate = 0.22  # 22% (지방세 포함)
        tax_amount = max(0, returns) * tax_rate
    
    after_tax_return = returns - tax_amount
    
    return {
        'before_tax_return': returns,
        'tax_amount': tax_amount,
        'after_tax_return': after_tax_return,
        'tax_rate': tax_rate,
        'tax_efficiency': (after_tax_return / returns) * 100 if returns > 0 else 100
    }

def validate_portfolio_weights(weights):
    """포트폴리오 가중치 유효성 검증"""
    if not weights:
        return False, "가중치가 비어있습니다."
    
    total_weight = sum(weights.values())
    
    if abs(total_weight - 1.0) > 0.01:
        return False, f"가중치 합계가 1이 아닙니다. (현재: {total_weight:.3f})"
    
    for asset, weight in weights.items():
        if weight < 0:
            return False, f"{asset}의 가중치가 음수입니다."
        if weight > 0.5:
            return False, f"{asset}의 가중치가 50%를 초과합니다."
    
    return True, "유효한 포트폴리오입니다."

def calculate_diversification_ratio(correlation_matrix, weights):
    """분산투자 비율 계산"""
    if correlation_matrix.empty or not weights:
        return 0
    
    try:
        # 가중평균 상관계수 계산
        weighted_correlations = []
        assets = list(weights.keys())
        
        for i, asset1 in enumerate(assets):
            for j, asset2 in enumerate(assets):
                if i != j and asset1 in correlation_matrix.index and asset2 in correlation_matrix.columns:
                    corr = correlation_matrix.loc[asset1, asset2]
                    weight_product = weights[asset1] * weights[asset2]
                    weighted_correlations.append(corr * weight_product)
        
        if weighted_correlations:
            avg_correlation = sum(weighted_correlations) / len(weighted_correlations)
            diversification_ratio = 1 - avg_correlation
            return max(0, min(1, diversification_ratio))
        else:
            return 0
    
    except Exception as e:
        print(f"분산투자 비율 계산 오류: {e}")
        return 0

def generate_portfolio_report(portfolio_data, user_profile):
    """포트폴리오 리포트 생성"""
    report = {
        'summary': {
            'total_assets': portfolio_data.get('total_value', 0),
            'expected_return': portfolio_data.get('expected_return', 0),
            'risk_level': portfolio_data.get('volatility', 0),
            'diversification_score': portfolio_data.get('diversification_ratio', 0)
        },
        'allocation': portfolio_data.get('weights', {}),
        'risk_metrics': portfolio_data.get('risk_metrics', {}),
        'recommendations': []
    }
    
    # 추천사항 생성
    if report['summary']['risk_level'] > 0.2:
        report['recommendations'].append("포트폴리오의 변동성이 높습니다. 채권 비중을 늘려보세요.")
    
    if report['summary']['diversification_score'] < 0.3:
        report['recommendations'].append("분산투자 효과가 부족합니다. 상관관계가 낮은 자산을 추가해보세요.")
    
    age = user_profile.get('age', 30)
    if age > 50 and any('주식' in str(k) for k in report['allocation'].keys()):
        stock_weight = sum(v for k, v in report['allocation'].items() if '주식' in str(k))
        if stock_weight > 0.6:
            report['recommendations'].append("나이를 고려하여 주식 비중을 줄이고 안정적인 자산 비중을 늘려보세요.")
    
    return report
