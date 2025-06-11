# agents/hyperclova_client.py 수정
import os
import json
import requests
import aiohttp
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class HyperCLOVAXClient:
    def __init__(self):
        self.api_key = os.getenv("NCP_CLOVASTUDIO_API_KEY")
        self.apigw_key = os.getenv("NCP_APIGW_API_KEY")
        
        # API 키 검증
        if not self.api_key or not self.apigw_key:
            logger.error("HyperCLOVA X API 키가 설정되지 않았습니다.")
            self.use_fallback = True
        else:
            self.use_fallback = False
            
        self.host = "https://clovastudio.stream.ntruss.com"
        self.request_id = "investment-advisor-001"
        
    def _get_headers(self):
        return {
            "Content-Type": "application/json; charset=utf-8",
            "X-NCP-CLOVASTUDIO-API-KEY": self.api_key,
            "X-NCP-APIGW-API-KEY": self.apigw_key,
            "X-NCP-CLOVASTUDIO-REQUEST-ID": self.request_id
        }
    
    async def generate_response(self, messages: List[Dict[str, str]], 
                              max_tokens: int = 500,
                              temperature: float = 0.7) -> str:
        """HyperCLOVA X를 사용하여 응답 생성 (폴백 포함)"""
        
        if self.use_fallback:
            return await self._fallback_analysis(messages)
        
        url = f"{self.host}/testapp/v1/chat-completions/HCX-003"
        
        data = {
            "messages": messages,
            "maxTokens": max_tokens,
            "temperature": temperature,
            "topP": 0.8,
            "repeatPenalty": 5.0,
            "includeAiFilters": True
        }
        
        try:
            timeout = aiohttp.ClientTimeout(total=30)  # 타임아웃 설정
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=data, headers=self._get_headers()) as response:
                    if response.status == 200:
                        response_data = await response.json()
                        return response_data["result"]["message"]["content"]
                    else:
                        error_text = await response.text()
                        logger.error(f"HyperCLOVA X API 오류: {response.status} - {error_text}")
                        return await self._fallback_analysis(messages)
                        
        except Exception as e:
            logger.error(f"HyperCLOVA X 요청 오류: {e}")
            return await self._fallback_analysis(messages)
    
    async def _fallback_analysis(self, messages: List[Dict[str, str]]) -> str:
        """API 실패 시 대체 감정 분석"""
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
