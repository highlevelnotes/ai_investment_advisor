# models.py
from typing import TypedDict, List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import pandas as pd

class InvestmentState(TypedDict):
    user_id: str
    tickers: List[str]
    user_preferences: Dict[str, Any]
    raw_data: Dict[str, Any]
    historical_data: Dict[str, pd.DataFrame]
    sentiment_data: Dict[str, Any]
    technical_analysis: Dict[str, Any]
    fundamental_analysis: Dict[str, Any]
    risk_analysis: Dict[str, Any]
    portfolio_optimization: Dict[str, Any]
    recommendations: Dict[str, Any]
    timestamp: str

@dataclass
class UserProfile:
    user_id: str
    age: int
    income: float
    net_worth: float
    risk_tolerance: str  # 'conservative', 'moderate', 'aggressive'
    investment_horizon: str  # '1y', '3y', '5y', '10y+'
    investment_goals: List[str]
    sector_preferences: List[str]
    current_portfolio: Dict[str, float]

@dataclass
class StockData:
    ticker: str
    current_price: float
    volume: int
    market_cap: float
    pe_ratio: Optional[float]
    dividend_yield: Optional[float]
    beta: Optional[float]
    timestamp: datetime
