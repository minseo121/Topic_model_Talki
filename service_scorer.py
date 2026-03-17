# ============================================================
# service_scorer.py
# 역할: 실제 서비스에서 사용하는 채점 핵심 로직.
#
# 두 모델을 결합해서 최종 점수를 산출:
#   1. topic_model_mnr : 주제 적합성 평가 (KoSimCSE 파인튜닝)
#   2. label_model     : 문장 품질 평가 (klue/roberta-base 파인튜닝)
#
# 전체 채점 흐름:
#   anchor 생성 → 코사인 유사도 계산 → 0~3 캘리브레이션
#   → quality score 계산 → 가중평균(topic 35% : quality 65%)
#   → 0~100 변환 → 문장별 분석 → worst_sentence 추출
#
# main.py의 /score 엔드포인트에서 이 클래스를 사용.
# ============================================================

import re
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer


# ── 경로 설정 ─────────────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).resolve().parent
TOPIC_MODEL_PATH = BASE_DIR / "topic_model_mnr"  # KoSimCSE 파인튜닝 모델
LABEL_MODEL_PATH = BASE_DIR / "label_model"      # klue/roberta-base 파인튜닝 모델


# ── 캘리브레이션 파라미터 ─────────────────────────────────────────────────────
# calc_margin_stats.py 실행 결과(valid 데이터 분포)를 보고 설정한 값.
# 코사인 유사도가 이 범위 안에 있는 데이터가 대부분 → 이 범위를 0~3으로 선형 매핑.
SIM_MIN = 0.15   # 유사도 이 값 이하 → 0점에 가까운 수준 (p05 기준)
SIM_MAX = 0.90   # 유사도 이 값 이상 → 최고점 수준 (p95 기준)

# on/off topic 판정 threshold
# 코사인 유사도가 이 값 미만이면 주제 이탈로 판정 → final 점수 0점 처리
T_OFF_SIM = 0.30

# 최종 점수 가중평균 비율
# topic 점수보다 quality 점수 비중을 높게 설정한 이유:
# on/off 판정이 이미 topic 적합성을 1차로 걸러주기 때문에
# 점수 산출에서는 글 품질 비중을 더 크게 반영
ALPHA = 0.35  # final = 0.35 × topic_score + 0.65 × quality_score

# sentence_evidence에서 off-topic 문장을 표시할 조건
ABS_OFF_TH = 0.35   # 문장 유사도가 이 값보다 낮아야 off_topic 후보
GAP_TH     = 0.20   # best_sim - worst_sim이 이 값보다 커야 off_topic 후보
# ==================================================


def clip01(x: float) -> float:
    """값을 0~1 범위로 클리핑."""
    return max(0.0, min(1.0, x))


def score3_to_100(x: float) -> int:
    """0~3 점수를 0~100 정수로 변환."""
    return int(max(0, min(100, round((x / 3.0) * 100))))


def sim_to_topic_score(sim: float) -> float:
    """
    코사인 유사도(sim)를 0~3 연속 점수로 캘리브레이션.

    SIM_MIN(0.15) 이하 → 0점, SIM_MAX(0.90) 이상 → 3점, 그 사이는 선형 보간.
    범위 밖 값은 clip01()로 0~1로 제한.
    """
    norm = (sim - SIM_MIN) / (SIM_MAX - SIM_MIN + 1e-12)
    norm = clip01(float(norm))
    return 3.0 * norm


def build_anchor_text(summary: str, desc: str, tags: List[str]) -> str:
    """
    주제 정보(summary, desc, tags)를 anchor 텍스트로 변환.
    학습(make_anchor.py)과 동일한 포맷 사용 → 학습-서비스 일관성 보장.

    출력 예시:
      [요약] 직업 선택의 중요성
      [설명] 직업을 고를 때 고려해야 할 조건들
      [태그] #직업 #선택 #가치관
    """
    tags     = [t.lstrip("#").strip() for t in (tags or []) if t and t.strip()]
    tags     = [f"#{t}" for t in tags][:3]
    tags_str = " ".join(tags)
    return "\n".join([
        f"[요약] {summary.strip()}",
        f"[설명] {desc.strip()}",
        f"[태그] {tags_str}".rstrip(),
    ]).strip()


def split_sentences(text: str) -> List[str]:
    """
    텍스트를 문장 단위로 분리 (한국어/영어 혼합 대응).
    마침표/느낌표/물음표 뒤 공백 또는 '다' 뒤 공백 기준으로 분리.
    """
    text  = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+|(?<=다)\s+", text)
    return [p.strip() for p in parts if p and p.strip()]


def extract_keywords(text: str, top_n: int = 5) -> List[str]:
    """
    TF-IDF 기반 핵심 키워드 추출.

    단일 문서 TF-IDF는 사실상 "자주 등장하는 중요 단어" 추출.
    형태소 분석 없이도 동작하는 MVP 수준으로 구현.
    중복 방지: 이미 뽑힌 키워드의 부분 문자열이면 스킵.
    LLM 피드백 단계에서 주제 핵심어로 활용 예정.
    """
    if not text.strip():
        return []
    vec    = TfidfVectorizer(max_features=300, ngram_range=(1, 2))
    X      = vec.fit_transform([text])
    scores = X.toarray()[0]
    terms  = vec.get_feature_names_out()
    idx    = np.argsort(scores)[::-1]

    out = []
    for i in idx:
        if scores[i] <= 0:
            break
        t = terms[i]
        if any(t in existing or existing in t for existing in out):
            continue  # 중복 키워드 스킵
        out.append(t)
        if len(out) >= top_n:
            break
    return out


class ServiceScorer:
    """
    서비스용 채점기.
    두 파인튜닝 모델을 로드하고 predict()로 최종 결과를 반환.
    main.py에서 앱 시작 시 한 번만 인스턴스 생성 후 재사용.
    """

    def __init__(self):
        # 주제 적합성 임베딩 모델 (KoSimCSE 파인튜닝)
        self.topic_model = SentenceTransformer(str(TOPIC_MODEL_PATH))

        # 문장 품질 분류 모델 (klue/roberta-base 파인튜닝)
        self.tokenizer   = AutoTokenizer.from_pretrained(str(LABEL_MODEL_PATH))
        self.label_model = AutoModelForSequenceClassification.from_pretrained(str(LABEL_MODEL_PATH))
        self.label_model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_model.to(self.device)

    def topic_similarity(self, anchor: str, doc: str) -> float:
        """
        anchor와 doc 텍스트의 코사인 유사도 계산.
        두 텍스트를 임베딩 후 내적(normalize된 벡터이므로 내적 = 코사인 유사도).
        반환값: -1~1 사이 (실제로는 대부분 0~1)
        """
        a = self.topic_model.encode([anchor], normalize_embeddings=True, convert_to_numpy=True)[0]
        d = self.topic_model.encode([doc],    normalize_embeddings=True, convert_to_numpy=True)[0]
        return float(a @ d)

    @torch.inference_mode()
    def quality_score(self, doc: str) -> float:
        """
        문장 품질을 낮음/보통/높음 3단계로 분류 후 확률 가중합으로 0~2 연속값 반환.

        단순 argmax(가장 높은 확률 클래스)가 아닌 확률 가중합을 쓰는 이유:
          "거의 높음"인 경우 1.8, "낮음/보통 반반"인 경우 0.5 같은
          부드러운 연속 값을 얻어 점수 변화가 더 자연스러움.

        반환값: 0~2 연속값 (0=낮음, 1=보통, 2=높음)
        """
        enc    = self.tokenizer(doc, return_tensors="pt", truncation=True, max_length=256)
        enc    = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.label_model(**enc).logits
        probs  = torch.softmax(logits, dim=1)[0]   # [p_낮음, p_보통, p_높음]
        score  = probs[0] * 0.0 + probs[1] * 1.0 + probs[2] * 2.0
        return float(score)

    def sentence_evidence(
        self,
        anchor: str,
        doc: str,
        abs_off_th: float = ABS_OFF_TH,
        gap_th: float = GAP_TH,
        top_k_on: int = 1,
        top_k_off: int = 1,
    ) -> Dict[str, List[str]]:
        """
        주제와 가장 가까운 문장(on_topic)과 가장 먼 문장(off_topic)을 추출.

        off_topic 판정 조건 (둘 다 만족해야 표시):
          1. 절대 유사도가 ABS_OFF_TH(0.35) 미만
          2. best_sim - worst_sim이 GAP_TH(0.20) 이상
             → 전체적으로 주제에 맞는데 한 문장만 튀는 경우만 표시
             → 전체가 다 낮으면 off_topic 문장이 따로 없는 것

        반환:
          on_topic_sentences : 유사도 상위 top_k_on개 문장
          off_topic_sentences: 조건 만족 시 유사도 하위 top_k_off개 문장
        """
        sents = split_sentences(doc)
        if len(sents) == 0:
            return {"on_topic_sentences": [], "off_topic_sentences": []}
        if len(sents) == 1:
            return {"on_topic_sentences": sents, "off_topic_sentences": []}

        anchor_vec = self.topic_model.encode([anchor], normalize_embeddings=True, convert_to_numpy=True)[0]
        sent_vecs  = self.topic_model.encode(sents,   normalize_embeddings=True, convert_to_numpy=True)
        sims       = sent_vecs @ anchor_vec  # 각 문장의 anchor와의 유사도

        order_desc = np.argsort(-sims)  # 유사도 내림차순
        order_asc  = np.argsort(sims)   # 유사도 오름차순

        best_sim  = float(sims[int(order_desc[0])])
        worst_sim = float(sims[int(order_asc[0])])

        on_sents = [sents[int(i)] for i in order_desc[:top_k_on]]

        # off_topic 조건: 절대값이 낮고 다른 문장과 격차가 클 때만 표시
        off_sents: List[str] = []
        if worst_sim < abs_off_th and (best_sim - worst_sim) > gap_th:
            off_sents = [sents[int(i)] for i in order_asc[:top_k_off]]

        return {"on_topic_sentences": on_sents, "off_topic_sentences": off_sents}

    def sentence_analysis(
        self,
        anchor: str,
        doc: str,
        off_topic_th: float = ABS_OFF_TH,
        coherence_th: float = 0.40,
        quality_th: int = 35,
    ) -> List[Dict[str, Any]]:
        """
        문장별 상세 분석. LLM 피드백 생성 시 입력 데이터로 활용 예정.

        각 문장마다 계산:
          - topic_score   : anchor와의 코사인 유사도 → 0~100
          - coherence_sim : 앞뒤 인접 문장과의 평균 유사도 (흐름 연속성)
                            첫 문장/마지막 문장은 한쪽만 계산
          - quality_score : label_model로 문장 품질 → 0~100
          - flags         : 해당 문장의 문제 유형 목록
            "off_topic"    : topic_score가 off_topic_th(0.35) 미만
            "low_coherence": coherence_sim이 coherence_th(0.40) 미만
            "low_quality"  : quality_score가 quality_th(35) 미만
        """
        sents = split_sentences(doc)
        if not sents:
            return []

        anchor_vec = self.topic_model.encode(
            [anchor], normalize_embeddings=True, convert_to_numpy=True
        )[0]
        sent_vecs  = self.topic_model.encode(
            sents, normalize_embeddings=True, convert_to_numpy=True
        )
        topic_sims = (sent_vecs @ anchor_vec).tolist()

        results = []
        for i, (sent, tvec) in enumerate(zip(sents, sent_vecs)):
            tsim        = topic_sims[i]
            topic_score = score3_to_100(sim_to_topic_score(tsim))

            # 인접 문장과의 평균 유사도 (흐름 연속성 측정)
            neighbors = []
            if i > 0:
                neighbors.append(float(sent_vecs[i - 1] @ tvec))
            if i < len(sents) - 1:
                neighbors.append(float(sent_vecs[i + 1] @ tvec))
            coherence_sim = float(np.mean(neighbors)) if neighbors else None

            # 문장 단위 품질 점수
            q_raw         = self.quality_score(sent)            # 0~2
            quality_score = int(min(100, round(q_raw * 50)))    # 0~100

            flags = []
            if tsim < off_topic_th:
                flags.append("off_topic")
            if coherence_sim is not None and coherence_sim < coherence_th:
                flags.append("low_coherence")
            if quality_score < quality_th:
                flags.append("low_quality")

            results.append({
                "sentence":      sent,
                "topic_score":   topic_score,
                "coherence_sim": round(coherence_sim, 3) if coherence_sim is not None else None,
                "quality_score": quality_score,
                "flags":         flags,
            })

        return results

    def predict(
        self,
        topic_summary: str,
        topic_desc: str,
        topic_tags: List[str],
        doc_text: str,
        debug: bool = False,
    ) -> Dict[str, Any]:
        """
        채점 메인 함수. /score 엔드포인트에서 호출.

        처리 순서:
          1. anchor 텍스트 생성
          2. topic_model로 코사인 유사도 계산 → 0~3 캘리브레이션
          3. label_model로 품질 점수 계산 → 0~3 스케일 맞춤
          4. 가중평균(ALPHA=0.35) → 최종 점수 0~100
          5. on_topic 판정 (False면 final=0 강제)
          6. sentence_evidence: on/off topic 문장 추출
          7. sentence_analysis: 문장별 상세 분석
          8. worst_sentence: 유사도 가장 낮은 문장 1개 추출

        반환 구조:
          on_topic         : 주제 적합 여부 (bool)
          scores           : final/topic/quality 점수 (0~100)
          keywords         : TF-IDF 핵심 키워드 5개
          evidence         : on/off topic 대표 문장
          sentence_analysis: 문장별 분석 리스트
          worst_sentence   : topic_score 최하위 문장 (LLM 피드백용)
          debug            : 내부 계산값 (debug=True일 때만 포함)
        """
        anchor = build_anchor_text(topic_summary, topic_desc, topic_tags)

        # ── 점수 계산 ──────────────────────────────────────────────────────────
        sim          = self.topic_similarity(anchor, doc_text)
        topic_score_3 = sim_to_topic_score(sim)       # 0~3

        q_2   = self.quality_score(doc_text)           # 0~2 연속값
        q_3   = q_2 * 1.5                              # 0~3 스케일로 맞춤 (topic과 동일 범위)
        final_3 = ALPHA * topic_score_3 + (1.0 - ALPHA) * q_3  # 가중평균

        on_topic = sim >= T_OFF_SIM  # 유사도 0.30 미만이면 주제 이탈

        topic_100   = score3_to_100(topic_score_3)
        quality_100 = int(min(100, round(q_2 * 50)))   # 0~2 → 0~100
        final_100   = score3_to_100(final_3)
        if not on_topic:
            final_100 = 0  # 주제 이탈 시 최종 점수 0점 처리

        # ── 부가 정보 생성 ─────────────────────────────────────────────────────
        if on_topic and topic_100 > 45:
            # 주제에 맞는 경우만 on/off 문장 구분
            evidence = self.sentence_evidence(anchor, doc_text)
        else:
            # 전체가 주제 이탈이면 모든 문장을 off_topic으로 표시
            evidence = {"on_topic_sentences": [], "off_topic_sentences": split_sentences(doc_text)}

        sent_analysis = self.sentence_analysis(anchor, doc_text)

        # ── worst_sentence: 유사도 가장 낮은 문장 추출 ────────────────────────
        # LLM 피드백 생성 시 "가장 문제 있는 문장"으로 활용 예정
        worst_sentence = None
        if sent_analysis:
            worst = min(sent_analysis, key=lambda x: x["topic_score"])
            worst_sentence = {
                "sentence":    worst["sentence"],
                "topic_score": worst["topic_score"],
                "flags":       worst["flags"],
            }

        result: Dict[str, Any] = {
            "on_topic": on_topic,
            "scores": {
                "final":   final_100,
                "topic":   topic_100,
                "quality": quality_100,
            },
            "keywords":         extract_keywords(doc_text, top_n=5),
            "evidence":         evidence,
            "sentence_analysis": sent_analysis,
            "worst_sentence":   worst_sentence,
        }

        if debug:
            result["debug"] = {
                "sim_raw":        round(sim, 4),
                "topic_score_3":  round(topic_score_3, 2),
                "quality_class":  int(round(q_2)),   # 0=낮음, 1=보통, 2=높음
                "quality_score_2": round(q_2, 2),
                "final_score_3":  round(final_3, 2),
                "calib": {
                    "SIM_MIN":    SIM_MIN,
                    "SIM_MAX":    SIM_MAX,
                    "T_OFF_SIM":  T_OFF_SIM,
                    "ALPHA":      ALPHA,
                    "ABS_OFF_TH": ABS_OFF_TH,
                    "GAP_TH":     GAP_TH,
                }
            }

        return result


# ── 단독 실행 테스트 ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    scorer = ServiceScorer()

    result = scorer.predict(
        topic_summary="사업 성공 요인",
        topic_desc="사업 아이템을 선택할 때 고려하면 좋은 것들. 현재 유행들.",
        topic_tags=["#사업", "#아이템", "#유행"],
        doc_text="두바이쫀득쿠키는 SNS 바이럴 마케팅과 한정판 전략을 통해 소비자들의 관심을 끌었습니다. "
                 "이러한 전략은 브랜드 가치를 높이고 매출 상승으로 이어졌습니다.",
        debug=True
    )

    print(result)
