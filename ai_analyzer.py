# ai_analyzer.py
import json
import re
from typing import Dict, Any, Optional, List
from langchain_naver import ChatClovaX
from config import Config

class AIAnalyzer:
    def __init__(self):
        """LangChain HyperClova X 초기화"""
        self.api_key = Config.HYPERCLOVA_X_API_KEY
        self.model_name = Config.HYPERCLOVA_MODEL
        self.max_tokens = Config.HYPERCLOVA_MAX_TOKENS
        
        if self.api_key:
            try:
                self.client = ChatClovaX(
                    api_key=self.api_key,
                    model=self.model_name,
                    max_tokens=self.max_tokens,
                    temperature=0.5,
                    top_p=0.8
                )
                self.available = True
                print("✅ HyperClova X 초기화 성공")
            except Exception as e:
                print(f"❌ HyperClova X 초기화 실패: {e}")
                self.available = False
        else:
            self.available = False
            print("❌ HyperClova X API 키가 없습니다")
    
    def comprehensive_market_analysis(self, macro_data: Dict, etf_data: Dict, user_profile: Dict) -> Dict:
        """AI 기반 종합 시장 분석 (기본값 없음)"""
        if not self.available:
            print("❌ AI 사용 불가능")
            return {}
        
        try:
            print("🤖 AI 종합 분석 시작")
            
            # 프롬프트 생성
            prompt = self._create_simple_prompt(macro_data, etf_data, user_profile)
            
            # AI 응답 생성
            response = self.client.invoke(prompt)
            print(f"📝 AI 응답 수신: {len(response.content)} 문자")
            
            # 줄바꿈 기반 파싱
            parsed_result = self._parse_by_linebreaks(response.content, etf_data, user_profile)
            
            if self._validate_result(parsed_result):
                print("✅ AI 분석 및 파싱 성공")
                return parsed_result
            else:
                print("❌ 파싱 결과 검증 실패")
                return {}
                
        except Exception as e:
            print(f"❌ AI 분석 실패: {e}")
            return {}
    
    def _create_simple_prompt(self, macro_data: Dict, etf_data: Dict, user_profile: Dict) -> str:
        """간단한 프롬프트 생성"""
        age = user_profile.get('age', 30)
        risk_tolerance = user_profile.get('risk_tolerance', '위험중립형')
        macro_summary = self._format_macro_data(macro_data)
        
        prompt = f"""
{age}세 {risk_tolerance} 투자자를 위한 퇴직연금 분석을 해주세요.
경제상황: {macro_summary}

다음 순서로 답변해주세요:

[매크로경제분석]
한국 경제 현황과 전망을 2-3문장으로 설명하세요.

[ETF시장동향]
국내 ETF 시장 트렌드를 2-3문장으로 설명하세요.

[투자전략]
맞춤 투자 전략을 2-3문장으로 설명하세요.

[리스크요인]
주요 위험요인을 2-3문장으로 설명하세요.

[포트폴리오]
8개 국내 ETF로 JSON 포트폴리오를 제시하세요.
{{"KODEX 200": 0.20, "TIGER 200": 0.15, "KODEX 국고채10년": 0.25, "KODEX 단기채권": 0.15, "KODEX 2차전지산업": 0.10, "KODEX 골드선물(H)": 0.08, "TIGER 국내리츠": 0.07}}

각 섹션 사이에 빈 줄을 넣어주세요.
"""
        return prompt
    
    def _parse_by_linebreaks(self, response_text: str, etf_data: Dict, user_profile: Dict) -> Dict:
        """줄바꿈 기반 파싱"""
        try:
            print("🔍 줄바꿈 기반 파싱 시작")
            
            # 두 번의 줄바꿈으로 블록 분할
            pattern = re.compile(r"\n\n+")
            blocks = pattern.split(response_text.strip())
            
            print(f"📦 분할된 블록 수: {len(blocks)}")
            
            # 각 블록을 섹션별로 분류
            sections = self._classify_blocks(blocks)
            
            # 포트폴리오 추출
            portfolio_weights = self._extract_portfolio_from_blocks(blocks, etf_data)
            
            # 빈 섹션이 있으면 실패로 처리
            if not self._all_sections_present(sections):
                print("❌ 일부 섹션이 누락됨")
                return {}
            
            # 포트폴리오가 없으면 실패로 처리
            if not portfolio_weights:
                print("❌ 포트폴리오 추출 실패")
                return {}
            
            # 결과 구성
            result = {
                'analysis': {
                    'macro_analysis': sections['macro'],
                    'market_trends': sections['market'],
                    'investment_strategy': sections['strategy'],
                    'risk_factors': sections['risk']
                },
                'portfolio': {
                    'weights': portfolio_weights,
                    'allocation_reasoning': 'AI 실시간 분석 기반 포트폴리오',
                    'diversification_strategy': 'AI 추천 다중 ETF 분산투자 전략',
                    'etf_count': len(portfolio_weights),
                    'category_distribution': self._analyze_category_distribution(portfolio_weights, etf_data)
                },
                'source': 'ai_real_analysis'
            }
            
            print("✅ AI 실제 분석 파싱 완료")
            return result
            
        except Exception as e:
            print(f"❌ 파싱 실패: {e}")
            return {}
    
    def _classify_blocks(self, blocks: List[str]) -> Dict[str, str]:
        """블록을 섹션별로 분류"""
        sections = {
            'macro': '',
            'market': '',
            'strategy': '',
            'risk': ''
        }
        
        # 섹션 헤더 패턴
        section_patterns = {
            'macro': [r'매크로경제분석', r'경제분석', r'경제상황', r'매크로'],
            'market': [r'ETF시장동향', r'시장동향', r'ETF동향', r'시장트렌드'],
            'strategy': [r'투자전략', r'투자방향', r'투자방법', r'전략'],
            'risk': [r'리스크요인', r'위험요인', r'리스크', r'위험']
        }
        
        for block in blocks:
            block = block.strip()
            if len(block) < 10:
                continue
            
            # 각 섹션 패턴과 매칭
            classified = False
            for section_key, patterns in section_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, block, re.IGNORECASE):
                        content = self._extract_content_from_block(block, pattern)
                        if content and len(content) > 20:  # 최소 길이 확인
                            sections[section_key] = content
                            print(f"✅ {section_key}: {len(content)} 문자 분류")
                            classified = True
                            break
                if classified:
                    break
        
        return sections
    
    def _extract_content_from_block(self, block: str, header_pattern: str) -> str:
        """블록에서 헤더를 제거하고 내용만 추출"""
        # 헤더 패턴 제거
        content = re.sub(rf'\[?{header_pattern}\]?', '', block, flags=re.IGNORECASE)
        
        # 앞뒤 공백, 줄바꿈, 특수문자 정리
        content = re.sub(r'^[\[\]\s\-\*\#]+', '', content)
        content = re.sub(r'[\[\]\s]+$', '', content)
        content = re.sub(r'\s+', ' ', content)
        
        return content.strip()
    
    def _extract_portfolio_from_blocks(self, blocks: List[str], etf_data: Dict) -> Dict[str, float]:
        """블록에서 포트폴리오 JSON 추출"""
        try:
            # 모든 ETF 이름 수집
            all_etf_names = []
            for category, etfs in etf_data.items():
                all_etf_names.extend(etfs.keys())
            
            # 포트폴리오 블록 찾기
            for block in blocks:
                if '포트폴리오' in block or '{' in block:
                    # JSON 패턴 찾기
                    json_patterns = [
                        r'\{[^{}]*\}',
                        r'\{.*?\}',
                    ]
                    
                    for pattern in json_patterns:
                        matches = re.findall(pattern, block, re.DOTALL)
                        for match in matches:
                            try:
                                # JSON 정제
                                cleaned_json = self._clean_json(match)
                                portfolio = json.loads(cleaned_json)
                                
                                if isinstance(portfolio, dict) and len(portfolio) >= 5:
                                    # ETF 이름 검증 및 정규화
                                    validated = self._validate_portfolio(portfolio, all_etf_names)
                                    if validated:
                                        print(f"📊 포트폴리오 추출 성공: {len(validated)}개 ETF")
                                        return validated
                            except Exception as e:
                                print(f"JSON 파싱 실패: {e}")
                                continue
            
            print("❌ 포트폴리오 JSON 추출 실패")
            return {}
            
        except Exception as e:
            print(f"❌ 포트폴리오 추출 실패: {e}")
            return {}
    
    def _clean_json(self, json_str: str) -> str:
        """JSON 문자열 정제"""
        # 작은따옴표를 큰따옴표로
        json_str = json_str.replace("'", '"')
        
        # 키에 따옴표 추가 (없는 경우)
        json_str = re.sub(r'([{,]\s*)([a-zA-Z가-힣][a-zA-Z가-힣0-9\s]*)\s*:', r'\1"\2":', json_str)
        
        # 불필요한 공백 정리
        json_str = re.sub(r'\s+', ' ', json_str)
        
        # 마지막 쉼표 제거
        json_str = re.sub(r',\s*}', '}', json_str)
        
        return json_str.strip()
    
    def _validate_portfolio(self, portfolio: Dict, etf_names: List[str]) -> Dict[str, float]:
        """포트폴리오 검증 및 정규화"""
        validated = {}
        
        for etf_name, weight in portfolio.items():
            # ETF 이름 매칭
            matched_name = self._find_etf_match(etf_name, etf_names)
            
            if matched_name:
                try:
                    weight = float(weight)
                    if weight > 1:  # 퍼센트 형태
                        weight = weight / 100
                    if 0.03 <= weight <= 0.5:  # 3%-50% 유효 범위
                        validated[matched_name] = weight
                except (ValueError, TypeError):
                    continue
        
        # 가중치 정규화
        if len(validated) >= 5:
            total_weight = sum(validated.values())
            if total_weight > 0:
                validated = {k: v/total_weight for k, v in validated.items()}
                return validated
        
        return {}
    
    def _find_etf_match(self, ai_name: str, etf_names: List[str]) -> Optional[str]:
        """ETF 이름 매칭"""
        # 1. 완전 일치
        if ai_name in etf_names:
            return ai_name
        
        # 2. 부분 문자열 매칭
        for etf_name in etf_names:
            if ai_name in etf_name or etf_name in ai_name:
                return etf_name
        
        # 3. 키워드 매칭
        ai_keywords = set(re.findall(r'[가-힣A-Za-z0-9]+', ai_name))
        for etf_name in etf_names:
            etf_keywords = set(re.findall(r'[가-힣A-Za-z0-9]+', etf_name))
            if len(ai_keywords.intersection(etf_keywords)) >= 2:
                return etf_name
        
        return None
    
    def _all_sections_present(self, sections: Dict[str, str]) -> bool:
        """모든 섹션이 존재하는지 확인"""
        required_sections = ['macro', 'market', 'strategy', 'risk']
        
        for section in required_sections:
            content = sections.get(section, '').strip()
            if len(content) < 20:  # 최소 길이 확인
                print(f"❌ {section} 섹션 부족: {len(content)} 문자")
                return False
        
        return True
    
    def _validate_result(self, result: Dict) -> bool:
        """결과 검증"""
        try:
            # 빈 결과 확인
            if not result:
                return False
            
            # 필수 구조 확인
            if not all(key in result for key in ['analysis', 'portfolio']):
                return False
            
            # 분석 섹션 확인
            analysis = result['analysis']
            required_sections = ['macro_analysis', 'market_trends', 'investment_strategy', 'risk_factors']
            
            for section in required_sections:
                if section not in analysis or len(analysis[section]) < 20:
                    print(f"❌ {section} 섹션 검증 실패")
                    return False
            
            # 포트폴리오 확인
            weights = result['portfolio'].get('weights', {})
            if len(weights) < 5:
                print(f"❌ ETF 수 부족: {len(weights)}개")
                return False
            
            total_weight = sum(weights.values())
            if abs(total_weight - 1.0) > 0.1:
                print(f"❌ 가중치 합계 오류: {total_weight}")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ 결과 검증 중 오류: {e}")
            return False
    
    def _analyze_category_distribution(self, weights: Dict, etf_data: Dict) -> Dict:
        """카테고리별 분포 분석"""
        category_weights = {}
        etf_by_category = {}
        
        for category, etfs in etf_data.items():
            category_weights[category] = 0
            etf_by_category[category] = []
            
            for etf_name in etfs.keys():
                if etf_name in weights:
                    category_weights[category] += weights[etf_name]
                    etf_by_category[category].append({
                        'name': etf_name,
                        'weight': weights[etf_name]
                    })
        
        return {
            'category_weights': category_weights,
            'etfs_by_category': etf_by_category
        }
    
    def _format_macro_data(self, macro_data: Dict) -> str:
        """매크로 데이터 포맷팅"""
        if not macro_data:
            return "GDP 3.2%, 인플레이션 2.1%, 기준금리 3.5%"
        
        formatted = []
        for indicator, data in macro_data.items():
            if isinstance(data, dict) and 'current' in data:
                formatted.append(f"{indicator} {data['current']:.1f}%")
        
        return ", ".join(formatted) if formatted else "경제지표 안정적"
