import pandas as pd
import numpy as np
from typing import Dict, List, Any
from datetime import datetime

class DataQualityValidator:
    def __init__(self):
        self.quality_thresholds = {
            'completeness': 0.95,
            'accuracy': 0.90,
            'consistency': 0.85,
            'timeliness': 0.80
        }
    
    def validate_comprehensive(self, hist_data: pd.DataFrame, stock_info: Dict = None) -> Dict:
        """종합 데이터 품질 검증"""
        
        quality_report = {}
        
        # 1. 완전성 검증
        completeness = self._check_completeness(hist_data)
        quality_report['completeness'] = completeness
        
        # 2. 정확성 검증
        accuracy = self._check_accuracy(hist_data)
        quality_report['accuracy'] = accuracy
        
        # 3. 일관성 검증
        consistency = self._check_consistency(hist_data)
        quality_report['consistency'] = consistency
        
        # 4. 적시성 검증
        timeliness = self._check_timeliness(hist_data)
        quality_report['timeliness'] = timeliness
        
        # 5. 종합 품질 점수 계산
        overall_score = self._calculate_overall_score(quality_report)
        quality_report['overall_score'] = overall_score
        
        # 6. 신뢰도 등급
        reliability = self._determine_reliability(overall_score)
        quality_report['reliability'] = reliability
        
        # 7. 품질 이슈 상세 분석
        issues = self._identify_quality_issues(hist_data, quality_report)
        quality_report['issues'] = issues
        
        # 8. 개선 권장사항
        recommendations = self._generate_recommendations(quality_report)
        quality_report['recommendations'] = recommendations
        
        return quality_report
    
    def _check_completeness(self, data: pd.DataFrame) -> Dict:
        """데이터 완전성 검증"""
        
        total_records = len(data)
        missing_data = data.isnull().sum()
        
        # 각 컬럼별 완전성 계산
        completeness_by_column = {}
        for column in data.columns:
            completeness_by_column[column] = 1 - (missing_data[column] / total_records)
        
        # 전체 완전성 점수
        overall_completeness = 1 - (missing_data.sum() / (total_records * len(data.columns)))
        
        # 중요 컬럼 (OHLCV) 완전성
        critical_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        critical_completeness = np.mean([
            completeness_by_column.get(col, 0) for col in critical_columns if col in data.columns
        ])
        
        return {
            'overall': overall_completeness,
            'critical_columns': critical_completeness,
            'by_column': completeness_by_column,
            'missing_records': missing_data.to_dict(),
            'score': critical_completeness * 100
        }
    
    def _check_accuracy(self, data: pd.DataFrame) -> Dict:
        """데이터 정확성 검증"""
        
        accuracy_issues = []
        accuracy_score = 100
        
        # 1. 가격 데이터 정확성 검증
        if 'Close' in data.columns:
            # 음수 가격 검증
            negative_prices = (data['Close'] < 0).sum()
            if negative_prices > 0:
                accuracy_issues.append(f"음수 가격: {negative_prices}개")
                accuracy_score -= 20
            
            # 이상치 검증 (3-sigma rule)
            price_mean = data['Close'].mean()
            price_std = data['Close'].std()
            outliers = ((data['Close'] - price_mean).abs() > 3 * price_std).sum()
            if outliers > len(data) * 0.05:  # 5% 이상이 이상치
                accuracy_issues.append(f"가격 이상치: {outliers}개")
                accuracy_score -= 15
        
        # 2. 거래량 정확성 검증
        if 'Volume' in data.columns:
            negative_volume = (data['Volume'] < 0).sum()
            if negative_volume > 0:
                accuracy_issues.append(f"음수 거래량: {negative_volume}개")
                accuracy_score -= 15
            
            # 거래량 0인 날 (주말/공휴일 제외하고 너무 많으면 문제)
            zero_volume = (data['Volume'] == 0).sum()
            if zero_volume > len(data) * 0.1:  # 10% 이상
                accuracy_issues.append(f"거래량 0인 날: {zero_volume}개")
                accuracy_score -= 10
        
        # 3. OHLC 관계 검증
        if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']):
            # High >= max(Open, Close) 검증
            high_errors = (data['High'] < np.maximum(data['Open'], data['Close'])).sum()
            # Low <= min(Open, Close) 검증
            low_errors = (data['Low'] > np.minimum(data['Open'], data['Close'])).sum()
            
            if high_errors > 0:
                accuracy_issues.append(f"High 가격 오류: {high_errors}개")
                accuracy_score -= 15
            
            if low_errors > 0:
                accuracy_issues.append(f"Low 가격 오류: {low_errors}개")
                accuracy_score -= 15
        
        return {
            'score': max(0, accuracy_score),
            'issues': accuracy_issues,
            'negative_prices': negative_prices if 'Close' in data.columns else 0,
            'price_outliers': outliers if 'Close' in data.columns else 0,
            'ohlc_errors': high_errors + low_errors if all(col in data.columns for col in ['Open', 'High', 'Low', 'Close']) else 0
        }
    
    def _check_consistency(self, data: pd.DataFrame) -> Dict:
        """데이터 일관성 검증"""
        
        consistency_issues = []
        consistency_score = 100
        
        # 1. 시계열 연속성 검증
        if len(data) > 1:
            date_gaps = pd.to_datetime(data.index).to_series().diff().dt.days
            large_gaps = (date_gaps > 7).sum()  # 7일 이상 갭
            if large_gaps > 0:
                consistency_issues.append(f"큰 날짜 갭: {large_gaps}개")
                consistency_score -= 10
        
        # 2. 가격 변동성 일관성
        if 'Close' in data.columns:
            daily_returns = data['Close'].pct_change().dropna()
            extreme_returns = (daily_returns.abs() > 0.2).sum()  # 20% 이상 변동
            if extreme_returns > len(daily_returns) * 0.02:  # 2% 이상
                consistency_issues.append(f"극단적 일일 수익률: {extreme_returns}개")
                consistency_score -= 15
        
        # 3. 거래량 패턴 일관성
        if 'Volume' in data.columns:
            volume_mean = data['Volume'].mean()
            volume_std = data['Volume'].std()
            extreme_volume = ((data['Volume'] - volume_mean).abs() > 5 * volume_std).sum()
            if extreme_volume > len(data) * 0.05:  # 5% 이상
                consistency_issues.append(f"극단적 거래량: {extreme_volume}개")
                consistency_score -= 10
        
        # 4. 데이터 타입 일관성
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        type_errors = 0
        for col in numeric_columns:
            if col in data.columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    type_errors += 1
                    consistency_issues.append(f"{col} 컬럼이 숫자형이 아님")
        
        if type_errors > 0:
            consistency_score -= type_errors * 20
        
        return {
            'score': max(0, consistency_score),
            'issues': consistency_issues,
            'date_gaps': large_gaps if len(data) > 1 else 0,
            'extreme_returns': extreme_returns if 'Close' in data.columns else 0,
            'extreme_volume': extreme_volume if 'Volume' in data.columns else 0,
            'type_errors': type_errors
        }
    
    def _check_timeliness(self, data: pd.DataFrame) -> Dict:
        """데이터 적시성 검증"""
        
        timeliness_score = 100
        timeliness_issues = []
        
        if len(data) > 0:
            # 최신 데이터 날짜
            latest_date = pd.to_datetime(data.index[-1])
            current_date = pd.Timestamp.now()
            
            # 데이터 지연 계산 (영업일 기준)
            business_days_delay = pd.bdate_range(latest_date, current_date).shape[0] - 1
            
            if business_days_delay > 5:  # 5영업일 이상 지연
                timeliness_issues.append(f"데이터 지연: {business_days_delay}영업일")
                timeliness_score -= min(50, business_days_delay * 5)
            elif business_days_delay > 2:  # 2영업일 이상 지연
                timeliness_issues.append(f"경미한 데이터 지연: {business_days_delay}영업일")
                timeliness_score -= business_days_delay * 2
            
            # 데이터 업데이트 빈도 확인
            data_frequency = self._analyze_data_frequency(data)
            if data_frequency['irregular']:
                timeliness_issues.append("불규칙한 데이터 업데이트 패턴")
                timeliness_score -= 15
        
        return {
            'score': max(0, timeliness_score),
            'issues': timeliness_issues,
            'latest_date': latest_date.strftime('%Y-%m-%d') if len(data) > 0 else 'N/A',
            'delay_days': business_days_delay if len(data) > 0 else 0,
            'frequency_analysis': data_frequency if len(data) > 0 else {}
        }
    
    def _analyze_data_frequency(self, data: pd.DataFrame) -> Dict:
        """데이터 업데이트 빈도 분석"""
        
        if len(data) < 10:
            return {'irregular': False, 'pattern': 'insufficient_data'}
        
        # 날짜 간격 분석
        date_diffs = pd.to_datetime(data.index).to_series().diff().dt.days.dropna()
        
        # 가장 일반적인 간격
        most_common_interval = date_diffs.mode().iloc[0] if len(date_diffs.mode()) > 0 else 1
        
        # 불규칙성 판단
        irregular_intervals = (date_diffs != most_common_interval).sum()
        irregularity_ratio = irregular_intervals / len(date_diffs)
        
        return {
            'irregular': irregularity_ratio > 0.3,  # 30% 이상 불규칙하면 문제
            'most_common_interval': int(most_common_interval),
            'irregularity_ratio': irregularity_ratio,
            'pattern': 'daily' if most_common_interval == 1 else 'weekly' if most_common_interval == 7 else 'irregular'
        }
    
    def _calculate_overall_score(self, quality_report: Dict) -> int:
        """종합 품질 점수 계산"""
        
        weights = {
            'completeness': 0.3,
            'accuracy': 0.4,
            'consistency': 0.2,
            'timeliness': 0.1
        }
        
        weighted_score = 0
        for dimension, weight in weights.items():
            if dimension in quality_report:
                score = quality_report[dimension].get('score', 0)
                weighted_score += score * weight
        
        return int(weighted_score)
    
    def _determine_reliability(self, overall_score: int) -> str:
        """신뢰도 등급 결정"""
        
        if overall_score >= 90:
            return '매우 높음'
        elif overall_score >= 80:
            return '높음'
        elif overall_score >= 70:
            return '보통'
        elif overall_score >= 60:
            return '낮음'
        else:
            return '매우 낮음'
    
    def _identify_quality_issues(self, data: pd.DataFrame, quality_report: Dict) -> List[str]:
        """품질 이슈 식별"""
        
        all_issues = []
        
        for dimension in ['completeness', 'accuracy', 'consistency', 'timeliness']:
            if dimension in quality_report:
                issues = quality_report[dimension].get('issues', [])
                all_issues.extend(issues)
        
        return all_issues
    
    def _generate_recommendations(self, quality_report: Dict) -> List[str]:
        """개선 권장사항 생성"""
        
        recommendations = []
        
        # 완전성 개선
        if quality_report.get('completeness', {}).get('score', 100) < 95:
            recommendations.append("누락된 데이터 보완 필요")
            recommendations.append("데이터 수집 프로세스 점검 권장")
        
        # 정확성 개선
        if quality_report.get('accuracy', {}).get('score', 100) < 90:
            recommendations.append("데이터 검증 로직 강화 필요")
            recommendations.append("이상치 탐지 및 처리 시스템 구축")
        
        # 일관성 개선
        if quality_report.get('consistency', {}).get('score', 100) < 85:
            recommendations.append("데이터 표준화 및 정규화 필요")
            recommendations.append("데이터 입력 규칙 정립")
        
        # 적시성 개선
        if quality_report.get('timeliness', {}).get('score', 100) < 80:
            recommendations.append("실시간 데이터 수집 시스템 구축")
            recommendations.append("데이터 업데이트 주기 단축")
        
        # 종합 점수가 낮은 경우
        if quality_report.get('overall_score', 100) < 70:
            recommendations.append("전체적인 데이터 관리 체계 재검토 필요")
            recommendations.append("데이터 품질 모니터링 시스템 도입")
        
        return recommendations
