# config.py
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class Config:
    """환경변수 기반 설정 클래스"""
    
    # API Keys
    ECOS_API_KEY = os.getenv('ECOS_API_KEY')
    HYPERCLOVA_X_API_KEY = os.getenv('HYPERCLOVA_X_API_KEY')
    
    # HyperClova X Settings
    HYPERCLOVA_MODEL = os.getenv('HYPERCLOVA_MODEL', 'hcx-005')
    HYPERCLOVA_MAX_TOKENS = int(os.getenv('HYPERCLOVA_MAX_TOKENS', '3000'))
    
    # API URLs
    ECOS_API_URL = "http://ecos.bok.or.kr/api"
    
    # Database
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///pension_portfolio.db')
    
    # App Settings
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# ETF 종목 코드 (PyKRX 호환)
ETF_CODES = {
    '국내주식형': {
        '069500': 'KODEX 200',
        '114800': 'KODEX 인버스',
        '122630': 'KODEX 레버리지',
        '091160': 'KODEX 반도체',
        '102110': 'TIGER 200',
        '251340': 'KODEX 코스닥150'
    },
    '해외주식형': {
        '138230': 'KODEX 미국S&P500선물(H)',
        '143850': 'TIGER 미국나스닥100',
        '195930': 'TIGER 차이나전기차SOLACTIVE',
        '261220': 'KODEX WTI원유선물(H)',
        '379800': 'KODEX 미국나스닥100TR'
    },
    '채권형': {
        '114260': 'KODEX 국고채3년',
        '148070': 'KODEX 국고채10년',
        '130730': 'KODEX 단기채권',
        '136340': 'TIGER 회사채AA-',
        '157450': 'TIGER 국고채10년'
    },
    '섹터/테마': {
        '117700': 'KODEX 2차전지산업',
        '139660': 'KODEX 바이오',
        '261240': 'KODEX K-신재생에너지',
        '278540': 'KODEX 2차전지소재',
        '364690': 'KODEX K-ESG'
    },
    '원자재/리츠': {
        '132030': 'KODEX 골드선물(H)',
        '130680': 'TIGER 원유선물Enhanced(H)',
        '365040': 'KODEX 미국리츠',
        '143460': 'TIGER 리츠',
        '411060': 'ACE KRX금현물'
    }
}

# 경제지표 코드
ECONOMIC_INDICATORS = {
    'GDP': '200Y001',
    'CPI': '901Y009',
    'PPI': '404Y014',
    'INTEREST_RATE': '722Y001',
    'EXCHANGE_RATE': '731Y001',
    'MONEY_SUPPLY': '101Y002',
    'EMPLOYMENT': '101Y003',
    'INDUSTRIAL_PRODUCTION': '403Y001'
}

# 포트폴리오 설정
PORTFOLIO_SETTINGS = {
    'REBALANCING_THRESHOLD': 0.05,
    'MIN_WEIGHT': 0.01,
    'MAX_WEIGHT': 0.4,
    'RISK_FREE_RATE': 0.025,
    'TRANSACTION_COST': 0.001
}

# 생애주기별 자산배분
LIFECYCLE_ALLOCATION = {
    '청년층': {'주식': 0.7, '채권': 0.25, '대안투자': 0.05},
    '중년층': {'주식': 0.5, '채권': 0.4, '대안투자': 0.1},
    '장년층': {'주식': 0.3, '채권': 0.6, '대안투자': 0.1}
}

# 위험성향별 자산배분
RISK_ALLOCATION = {
    '안정형': {'주식': 0.2, '채권': 0.7, '대안투자': 0.1},
    '안정추구형': {'주식': 0.4, '채권': 0.5, '대안투자': 0.1},
    '위험중립형': {'주식': 0.6, '채권': 0.3, '대안투자': 0.1},
    '적극투자형': {'주식': 0.8, '채권': 0.15, '대안투자': 0.05}
}

# 앱 설정
APP_CONFIG = {
    'TITLE': 'Smart Pension ETF Agent',
    'DESCRIPTION': 'PyKRX & LangChain HyperClova X 기반 ETF 퇴직연금 포트폴리오 관리 시스템',
    'VERSION': '1.0.0',
    'AUTHOR': 'KDT 해커톤 팀',
    'PAGE_ICON': '📊',
    'LAYOUT': 'wide'
}

def validate_config():
    """필수 환경변수 검증"""
    required_vars = ['HYPERCLOVA_X_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not getattr(Config, var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"⚠️  경고: 다음 환경변수가 설정되지 않았습니다: {', '.join(missing_vars)}")
        print("샘플 데이터로 시스템이 동작합니다.")
        return False
    
    return True

def get_api_status():
    """API 연결 상태 확인"""
    status = {
        'ecos': bool(Config.ECOS_API_KEY),
        'hyperclova_x': bool(Config.HYPERCLOVA_X_API_KEY),
        'pykrx': True  # PyKRX는 별도 API 키 불필요
    }
    return status
