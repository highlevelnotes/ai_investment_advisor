# visualization.py
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_portfolio_pie_chart(weights):
    """포트폴리오 구성 파이차트"""
    if not weights:
        return go.Figure()
    
    labels = list(weights.keys())
    values = [w * 100 for w in weights.values()]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        textinfo='label+percent',
        textposition='outside',
        marker=dict(
            colors=px.colors.qualitative.Set3,
            line=dict(color='#FFFFFF', width=2)
        )
    )])
    
    fig.update_layout(
        title="포트폴리오 구성",
        title_x=0.5,
        font=dict(size=12),
        showlegend=True,
        height=500
    )
    
    return fig

def create_performance_chart(etf_data):
    """ETF 성과 비교 차트"""
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set1
    color_idx = 0
    
    for category, etfs in etf_data.items():
        for name, data in etfs.items():
            if 'history' in data and not data['history'].empty:
                # 정규화된 가격 (100 기준)
                normalized_prices = (data['history']['Close'] / data['history']['Close'].iloc[0]) * 100
                
                fig.add_trace(go.Scatter(
                    x=data['history'].index,
                    y=normalized_prices,
                    mode='lines',
                    name=f"{name} ({category})",
                    line=dict(color=colors[color_idx % len(colors)], width=2),
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                '날짜: %{x}<br>' +
                                '수익률: %{y:.1f}%<br>' +
                                '<extra></extra>'
                ))
                color_idx += 1
    
    fig.update_layout(
        title="ETF 성과 비교 (정규화 기준: 100)",
        title_x=0.5,
        xaxis_title="날짜",
        yaxis_title="정규화 가격",
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        ),
        height=600,
        margin=dict(r=200)
    )
    
    return fig

def create_risk_return_scatter(etf_data):
    """위험-수익률 산점도"""
    risk_return_data = []
    
    for category, etfs in etf_data.items():
        for name, data in etfs.items():
            if 'returns' in data and not data['returns'].empty:
                annual_return = data['returns'].mean() * 252 * 100
                annual_vol = data['returns'].std() * np.sqrt(252) * 100
                
                risk_return_data.append({
                    'ETF': name,
                    'Category': category,
                    'Return': annual_return,
                    'Risk': annual_vol
                })
    
    if not risk_return_data:
        return go.Figure()
    
    df = pd.DataFrame(risk_return_data)
    
    fig = px.scatter(
        df, 
        x='Risk', 
        y='Return',
        color='Category',
        size=[20] * len(df),  # 모든 점을 같은 크기로
        hover_name='ETF',
        hover_data={'Category': True, 'Return': ':.2f', 'Risk': ':.2f'},
        title="ETF 위험-수익률 분포"
    )
    
    fig.update_traces(
        marker=dict(
            line=dict(width=2, color='DarkSlateGrey'),
            sizemode='diameter'
        )
    )
    
    fig.update_layout(
        title_x=0.5,
        xaxis_title="위험 (연환산 변동성, %)",
        yaxis_title="수익률 (연환산, %)",
        height=500,
        hovermode='closest'
    )
    
    return fig

def create_correlation_heatmap(etf_data):
    """상관관계 히트맵"""
    returns_data = {}
    
    for category, etfs in etf_data.items():
        for name, data in etfs.items():
            if 'returns' in data and not data['returns'].empty:
                returns_data[name] = data['returns']
    
    if len(returns_data) < 2:
        return go.Figure()
    
    returns_df = pd.DataFrame(returns_data)
    returns_df = returns_df.dropna()
    
    if returns_df.empty:
        return go.Figure()
    
    corr_matrix = returns_df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False,
        hovertemplate='<b>%{y} vs %{x}</b><br>상관계수: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="ETF 간 상관관계",
        title_x=0.5,
        height=600,
        xaxis=dict(tickangle=45),
        yaxis=dict(tickangle=0)
    )
    
    return fig

def create_efficient_frontier_chart(efficient_frontier_df, optimal_portfolio=None):
    """효율적 프론티어 차트"""
    if efficient_frontier_df.empty:
        return go.Figure()
    
    fig = go.Figure()
    
    # 효율적 프론티어
    fig.add_trace(go.Scatter(
        x=efficient_frontier_df['Volatility'] * 100,
        y=efficient_frontier_df['Return'] * 100,
        mode='lines',
        name='효율적 프론티어',
        line=dict(color='blue', width=3),
        hovertemplate='위험: %{x:.2f}%<br>수익률: %{y:.2f}%<extra></extra>'
    ))
    
    # 최적 포트폴리오 표시
    if optimal_portfolio:
        fig.add_trace(go.Scatter(
            x=[optimal_portfolio['volatility'] * 100],
            y=[optimal_portfolio['expected_return'] * 100],
            mode='markers',
            name='최적 포트폴리오',
            marker=dict(
                color='red',
                size=15,
                symbol='star',
                line=dict(color='darkred', width=2)
            ),
            hovertemplate='<b>최적 포트폴리오</b><br>' +
                         '위험: %{x:.2f}%<br>' +
                         '수익률: %{y:.2f}%<br>' +
                         f'샤프비율: {optimal_portfolio["sharpe_ratio"]:.3f}<extra></extra>'
        ))
    
    fig.update_layout(
        title="효율적 프론티어",
        title_x=0.5,
        xaxis_title="위험 (변동성, %)",
        yaxis_title="기대수익률 (%)",
        height=500,
        hovermode='closest'
    )
    
    return fig

def create_monte_carlo_chart(monte_carlo_df, optimal_portfolio=None):
    """몬테카를로 시뮬레이션 차트"""
    if monte_carlo_df.empty:
        return go.Figure()
    
    fig = go.Figure()
    
    # 시뮬레이션 결과
    fig.add_trace(go.Scatter(
        x=monte_carlo_df['Volatility'] * 100,
        y=monte_carlo_df['Return'] * 100,
        mode='markers',
        name='시뮬레이션 포트폴리오',
        marker=dict(
            color=monte_carlo_df['Sharpe_Ratio'],
            colorscale='Viridis',
            size=3,
            opacity=0.6,
            colorbar=dict(title="샤프 비율"),
            line=dict(width=0)
        ),
        hovertemplate='위험: %{x:.2f}%<br>' +
                     '수익률: %{y:.2f}%<br>' +
                     '샤프비율: %{marker.color:.3f}<extra></extra>'
    ))
    
    # 최적 포트폴리오 표시
    if optimal_portfolio:
        fig.add_trace(go.Scatter(
            x=[optimal_portfolio['volatility'] * 100],
            y=[optimal_portfolio['expected_return'] * 100],
            mode='markers',
            name='최적 포트폴리오',
            marker=dict(
                color='red',
                size=15,
                symbol='star',
                line=dict(color='darkred', width=2)
            ),
            hovertemplate='<b>최적 포트폴리오</b><br>' +
                         '위험: %{x:.2f}%<br>' +
                         '수익률: %{y:.2f}%<br>' +
                         f'샤프비율: {optimal_portfolio["sharpe_ratio"]:.3f}<extra></extra>'
        ))
    
    fig.update_layout(
        title="몬테카를로 시뮬레이션 (10,000회)",
        title_x=0.5,
        xaxis_title="위험 (변동성, %)",
        yaxis_title="기대수익률 (%)",
        height=500,
        hovermode='closest'
    )
    
    return fig

def create_economic_indicators_chart(economic_data):
    """경제지표 차트"""
    if not economic_data:
        return go.Figure()
    
    indicators = list(economic_data.keys())
    current_values = [data['current'] for data in economic_data.values()]
    previous_values = [data['previous'] for data in economic_data.values()]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='현재',
        x=indicators,
        y=current_values,
        marker_color='lightblue',
        text=[f'{val:.2f}' for val in current_values],
        textposition='auto'
    ))
    
    fig.add_trace(go.Bar(
        name='이전',
        x=indicators,
        y=previous_values,
        marker_color='lightcoral',
        text=[f'{val:.2f}' for val in previous_values],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="주요 경제지표 현황",
        title_x=0.5,
        xaxis_title="경제지표",
        yaxis_title="값",
        barmode='group',
        height=400,
        xaxis=dict(tickangle=45)
    )
    
    return fig

def create_portfolio_performance_chart(portfolio_history):
    """포트폴리오 성과 추이"""
    if not portfolio_history:
        return go.Figure()
    
    dates = list(portfolio_history.keys())
    values = list(portfolio_history.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=values,
        mode='lines+markers',
        name='포트폴리오 가치',
        line=dict(color='green', width=3),
        marker=dict(size=6),
        hovertemplate='날짜: %{x}<br>가치: %{y:,.0f}원<extra></extra>'
    ))
    
    fig.update_layout(
        title="포트폴리오 성과 추이",
        title_x=0.5,
        xaxis_title="날짜",
        yaxis_title="포트폴리오 가치 (원)",
        height=400,
        hovermode='x'
    )
    
    return fig
