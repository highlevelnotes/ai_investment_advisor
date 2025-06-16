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

# 순수 국내 ETF 종목 코드 (미국 리츠 제거)
ETF_CODES = {
    '국내주식형': {
        '069500': 'KODEX 200',
        '102110': 'TIGER 200',
        '114800': 'KODEX 인버스',
        '122630': 'KODEX 레버리지',
        '091160': 'KODEX 반도체',
        '251340': 'KODEX 코스닥150',
        '229200': 'KODEX 코스닥150선물',
        '233740': 'KODEX 코스닥150레버리지',
        '278240': 'KODEX 코스닥150선물레버리지',
        '117700': 'KODEX 2차전지산업'
    },
    '국내채권형': {
        '114260': 'KODEX 국고채3년',
        '148070': 'KODEX 국고채10년',
        '130730': 'KODEX 단기채권',
        '136340': 'TIGER 회사채AA-',
        '157450': 'TIGER 단기통안채',
        '214980': 'KODEX 단기채권PLUS',
        '182490': 'TIGER 국고채10년',
        '305080': 'TIGER 회사채BBB-',
        '273130': 'KODEX 종합채권(AA-이상)액티브'
    },
    '국내섹터/테마': {
        '139660': 'KODEX 바이오',
        '261240': 'KODEX K-신재생에너지',
        '278540': 'KODEX 2차전지소재',
        '228810': 'TIGER 미디어컨텐츠',
        '139220': 'TIGER 200건설',
        '102960': 'KODEX 기계장비',
        '140700': 'KODEX 보험',
        '228790': 'KODEX 게임',
        '261270': 'KODEX K-뉴딜',
        '364690': 'KODEX 혁신기술테마액티브'
    },
    '국내대안투자': {
        '132030': 'KODEX 골드선물(H)',
        '411060': 'ACE KRX금현물',
        '143460': 'TIGER 국내리츠',  # 국내 리츠만 유지
        '476800': 'KODEX 한국부동산리츠인프라',  # 국내 부동산 인프라
        '130680': 'TIGER 원유선물Enhanced(H)',
        '261220': 'KODEX WTI원유선물(H)'
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

# LLM 기반 동적 자산배분을 위한 기본 템플릿
LIFECYCLE_ALLOCATION = {
    '청년층': {'국내주식': 0.6, '국내채권': 0.25, '국내섹터': 0.1, '국내대안': 0.05},
    '중년층': {'국내주식': 0.4, '국내채권': 0.4, '국내섹터': 0.1, '국내대안': 0.1},
    '장년층': {'국내주식': 0.25, '국내채권': 0.6, '국내섹터': 0.05, '국내대안': 0.1}
}

RISK_ALLOCATION = {
    '안정형': {'국내주식': 0.15, '국내채권': 0.7, '국내섹터': 0.05, '국내대안': 0.1},
    '안정추구형': {'국내주식': 0.3, '국내채권': 0.5, '국내섹터': 0.1, '국내대안': 0.1},
    '위험중립형': {'국내주식': 0.5, '국내채권': 0.3, '국내섹터': 0.1, '국내대안': 0.1},
    '적극투자형': {'국내주식': 0.7, '국내채권': 0.15, '국내섹터': 0.1, '국내대안': 0.05}
}

# 앱 설정
APP_CONFIG = {
    'TITLE': 'Smart Pension ETF Agent',
    'DESCRIPTION': '순수 국내 ETF 중심 AI 기반 퇴직연금 포트폴리오 관리 시스템',
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
        'pykrx': True
    }
    return status
