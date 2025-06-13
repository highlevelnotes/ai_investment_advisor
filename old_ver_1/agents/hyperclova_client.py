# agents/hyperclova_client.py
import os
from typing import List, Dict, Any
import logging
from langchain_naver import ChatClovaX
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

class HyperCLOVAXClient:
    def __init__(self):
        try:
            # LangChain을 통한 HyperCLOVA X 초기화
            self.chat = ChatClovaX(
                model="HCX-005",
                temperature=0.7,
                max_tokens=500,
                timeout=30,
                max_retries=2
            )
            self.available = True
            logger.info("HyperCLOVA X (HCX-005) 모델 초기화 완료")
            
        except Exception as e:
            logger.error(f"HyperCLOVA X 초기화 실패: {e}")
            self.available = False
    
    async def generate_response(self, messages: List[Dict[str, str]], 
                              max_tokens: int = 500,
                              temperature: float = 0.7) -> str:
        """LangChain을 통한 HyperCLOVA X 응답 생성"""
        
        if not self.available:
            return await self._fallback_analysis(messages)
        
        try:
            # LangChain 메시지 형식으로 변환
            langchain_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    langchain_messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
            
            # 모델 파라미터 동적 설정
            chat_with_params = ChatClovaX(
                model="HCX-005",
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=30
            )
            
            # 응답 생성
            response = await chat_with_params.ainvoke(langchain_messages)
            return response.content
            
        except Exception as e:
            logger.error(f"HyperCLOVA X 요청 오류: {e}")
            return await self._fallback_analysis(messages)
    
    async def _fallback_analysis(self, messages: List[Dict[str, str]]) -> str:
        """API 실패 시 대체 분석"""
        user_message = messages[-1]["content"] if messages else ""
        
        # 한국어 금융 키워드 기반 감정 분석
        positive_keywords = [
            '상승', '오름', '증가', '호재', '긍정', '성장', '개선', '상향', 
            '돌파', '급등', '강세', '회복', '반등', '실적', '수익', '이익'
        ]
        
        negative_keywords = [
            '하락', '내림', '감소', '악재', '부정', '하향', '급락', '약세',
            '손실', '적자', '우려', '리스크', '위험', '충격', '폭락', '침체'
        ]
        
        pos_count = sum(1 for keyword in positive_keywords if keyword in user_message)
        neg_count = sum(1 for keyword in negative_keywords if keyword in user_message)
        
        if pos_count > neg_count:
            sentiment_score = 0.3 + (pos_count - neg_count) * 0.1
            summary = "전반적으로 긍정적인 뉴스가 많습니다."
        elif neg_count > pos_count:
            sentiment_score = -0.3 - (neg_count - pos_count) * 0.1
            summary = "전반적으로 부정적인 뉴스가 많습니다."
        else:
            sentiment_score = 0.0
            summary = "중립적인 시장 분위기입니다."
        
        return f"감정 점수: {sentiment_score:.2f}\n신뢰도: 0.6\n분석 요약: {summary}"
