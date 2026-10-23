# 💳 CARD-FDS-N : 이상거래 탐지 및 맞춤형 알림 서비스

> **SMOTENC와 클래스 가중치를 적용한 LightGBM 기반 카드 이상거래 탐지 및 디지털 취약계층 친화적 시·청각 알림 서비스**
> 
> **개발 기간**: 2026.10.19 ~ 2026.10.23  
> **팀원**: 오주원, 김지선, 신상아, 이인애  

---

## 📌 1. 프로젝트 개요 (Overview)

### 배경 및 필요성 (Why Now?)
- **기존 탐지 사각지대**: 개인 중심의 단일 패턴 위주 분석으로 인해 새로운 유형이나 상황별 사기 패턴을 놓칠 위험 존재.
- **디지털 역량 격차 심화**: 고령층 및 디지털 취약계층은 의심 거래가 발생했을 때 스스로 인지하고 빠르게 대응하기 어려움.
- **맞춤형 서비스 부재**: 이상거래 발생 시 누구나 이해하기 쉬운 직관적 경고 시스템 및 이탈 방지 솔루션이 절실함.

### 핵심 솔루션 (What is it?)
`CARD-FDS-N`은 **대규모 카드 거래 데이터를 바탕으로 실시간 이상거래 확률을 산출**하고, 탐지된 위험도에 따라 **디지털 취약계층(고령층, 언어/정보 소외자) 맞춤형 시각·청각(TTS) 알림**을 제공하는 End-to-End FDS 솔루션입니다.

---

## 🛠 2. Tech Stack & Architecture

### Tech Stack
| 구분 | 기술 스택 |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **ML Engine** | LightGBM 4.x, XGBoost 2.x, Scikit-learn |
| **Imbalanced Data handling** | imbalanced-learn (`SMOTENC`) |
| **XAI (설명 가능 AI)** | SHAP (`TreeExplainer`) |
| **Backend & Inference API**| FastAPI (High-performance Async API) |
| **Frontend & UI** | Streamlit, Web Speech API (TTS) |

### System Architecture
```
[AI Hub 카드거래 합성데이터]
        │
        ▼
[데이터 전처리 & 누수 차단] ── (수치형: RobustScaler / 범주형: Target Encoding)
        │
        ▼
[클래스 불균형 해소] ──────── (Train 전용 SMOTENC 0.3 적용)
        │
        ▼
[모델링 & 튜닝] ──────────── (LightGBM, XGBoost, Random Forest 비교 / Threshold Optimization)
        │
        ▼
[XAI & 초고속 API] ───────── (SHAP 변수 중요도 추출 + FastAPI 배치/단건 추론 < 100ms)
        │
        ▼
[시·청각 알림 UI] ────────── (Streamlit 대시보드 + Web Speech API TTS 경고)
```

---

## 📊 3. 데이터셋 및 전처리 (Data & Preprocessing)

### 데이터 구성 및 분할
- **데이터 출처**: AI Hub 카드거래 합성데이터셋
- **데이터 규모**: 학습/검증 데이터 약 195만 건 (학습 173.5만 건, 검증 21.6만 건 / 이상 비율 3.68%)
- **분할 비율**: Train (70%) : Validation (10%) : Test (20%) 층화 분할(Stratified Split) 적용

### 데이터 누수(Data Leakage) 방지 및 전처리 원칙
1. **정답 누수 차단**: '이상거래유형', '이상거래설명' 등 정답을 직접 암시하는 컬럼 사전 제거.
2. **Strict Train-only Standard**: 모든 스케일링(RobustScaler) 및 결측치 처리 기준은 **Train 데이터셋**에서만 학습 후 Valid/Test에 적용.
3. **특징 피처(Features)**: 승인금액, 카드한도, 거래시간, 연령, 성별, 해외/온라인 여부, 신규 가맹점 여부, 최근 이용 경과일 등.

---

## 📈 4. 모델링 및 성능 평가 (Experiments & Results)

### 핵심 실험 4단계
1. **Baseline**: 불균형 데이터를 그대로 학습한 기준선 확인
2. **SMOTENC 비율 비교**: 0.1, 0.2, 0.3, 0.5 중 **0.3 최적 비율** 선정
3. **모델 비교**: LightGBM, Random Forest, XGBoost performance benchmark
4. **고도화 LightGBM**: 연령, 성별 등 인구통계학적 피처 추가 및 임계값(Threshold $t^* = 0.55$) 최적화

### 최종 모델 비교 성능 (Test Set 기준)

| Model | Precision | Recall | F1-Score | PR-AUC |
| :--- | :---: | :---: | :---: | :---: |
| Base LightGBM | 0.8077 | 0.6511 | 0.7210 | 0.7650 |
| Random Forest | 0.7261 | **0.6945** | 0.7100 | 0.7548 |
| XGBoost | 0.8209 | 0.6488 | 0.7248 | 0.7644 |
| **고도화 LightGBM (Final)** | **0.8275** | 0.6633 | **0.7364** | **0.7776** |

> 📌 **최종 모델 선정 사유**: 극단적 불균형 데이터 특성상 Accuracy(0.9820)보다 **F1-Score 및 PR-AUC**를 핵심 지표로 설정했습니다. 고도화 LightGBM이 Validation PR-AUC 0.8018, Test PR-AUC 0.7776으로 가장 우수한 종합 성능을 기록하여 최종 모델로 채택되었습니다.

---

## 🔍 5. 모델 설명 가능성 (XAI - SHAP Analysis)

`SHAP (TreeExplainer)`를 활용하여 이상거래 판단에 미친 상위 5가지 주요 요인을 도출했습니다.

1. **`amt` (승인금액)**: 카드 차단 전 고액을 한 번에 결제하는 '초기 고액 결제 패턴'
2. **`hour` (거래시간)**: SMS/알림 확인이 어려운 '심야 취약 시간대(01~05시)' 집중 공략
3. **`age` (연령)**: 고령층의 정형화된 패턴 및 디지털 채널 이용 빈도 차이 반영
4. **`amt_to_limit_ratio` (한도 대비 승인금액)**: 잔여 한도를 모두 사용하는 '잔액 털기 시나리오'
5. **`days_since_last` (최근 거래 이후 경과일수)**: 잠자던 카드가 악용되는 '장기 휴면 카드 부활'

---

## 🚀 6. 서비스 기능 및 실시간 성능 검증

### 1) FastAPI 기반 초고속 추론 엔진
- **단건 추론 Latency**: **4.47ms** (목표 < 100ms 대비 대폭 단축)
- **대량 배치 검증**: 390,575건 대량 배치 추론 검증 완료 (배치 응답시간 < 200ms)

### 2) 3단계 위험등급 판정 및 사용자 알림 서비스 (Streamlit)
* **이상거래 확률 기준**:
  - 🟢 **정상 (< 30%)**: 정상 거래 안내
  - 🟡 **주의 (30% ~ 55%)**: 거래 재확인 유도
  - 🔴 **위험 (≥ 55%)**: 고위험 거래 즉시 차단 경고 + 보호자 알림 / 카드 일시정지 안내
* **맞춤형 UX/UI**: 고대비 색상, 대형 팝업 직관적 아이콘, **Web Speech API 기반 음성(TTS) 경고** 지원

---

## 💼 7. 비즈니스 성과 및 기대효과

- **정량적 효과**:
  - 이상거래 **금액 기준 탐지율 95.47%** 달성 (재현율 65.03%)
  - 연간 잠재 피해 노출 예방액 추정: **약 177.37억 원** (합성데이터 표본 역보정 기준)
- **정성적/사업적 효과**:
  - **B2B SaaS**: 금융/핀테크 기업 대상 FDS API 구독 수익 모델 창출
  - **B2G 공공 보안**: 디지털 약자 보호를 위한 안심 금융 서비스 선점

---

## 🗺 8. 향후 로드맵 (Roadmap)

- **Phase 1 (Short-term)**: 실사용자(고령층/외국인) 피드백 기반 UI/UX 개선 및 FDS 단일 API 표준화
- **Phase 2 (Mid-term)**: 다양한 금융 상품 라인업으로 FDS 탐지 커버리지 확대
- **Phase 3 (Long-term)**: 통신-금융 데이터 융합 실시간 파이프라인(Kafka/GNN 등) 구축 및 해외 시장 확장
