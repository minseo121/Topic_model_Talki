# ============================================================
# main.py
# 역할: FastAPI 서버 엔드포인트 정의.
#
# 엔드포인트: POST /score
#   요청을 받아 채점 결과를 JSON으로 반환.
#   is_stt=True이면 STT 보정(stt_normalizer) 먼저 수행 후 채점.
#
# 담당 범위:
#   이 파일은 ML 로직의 진입점으로, 실제 채점은 service_scorer.py가 담당.
#   STT 보정은 stt_normalizer.py가 담당.
#   LLM 요약/피드백 생성은 미구현 (추후 추가 예정).
# ============================================================

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from service_scorer import ServiceScorer
from stt_normalizer import normalize_stt

app    = FastAPI()
scorer = ServiceScorer()  # 앱 시작 시 모델 한 번만 로드 (재사용)


# ── LLM 연동 ──────────────────────────────────────────────────────────────────
# TODO: 사용 중인 LLM API에 맞게 llm_call 함수를 구현할 것.
# 시그니처: (prompt: str) -> str
#
# [GPT 예시]
# import openai
# _client = openai.OpenAI(api_key="...")
# def llm_call(prompt: str) -> str:
#     return _client.chat.completions.create(
#         model="gpt-4o",
#         messages=[{"role": "user", "content": prompt}],
#     ).choices[0].message.content
#
# [Claude 예시]
# import anthropic
# _client = anthropic.Anthropic(api_key="...")
# def llm_call(prompt: str) -> str:
#     return _client.messages.create(
#         model="claude-sonnet-4-6",
#         max_tokens=2048,
#         messages=[{"role": "user", "content": prompt}],
#     ).content[0].text
# ──────────────────────────────────────────────────────────────────────────────

def llm_call(prompt: str) -> str:
    raise NotImplementedError("llm_call 미구현: 위 주석을 참고해서 LLM API에 맞게 구현할 것")


class ScoreRequest(BaseModel):
    """
    /score 엔드포인트 요청 스키마.

    topic_summary : 주제 한 줄 요약 (예: "직업 선택의 중요성")
    topic_desc    : 주제 상세 설명 (예: "직업을 고를 때 고려할 조건들")
    topic_tags    : 관련 태그 목록 (예: ["#직업", "#선택", "#가치관"])
    doc_text      : 평가할 텍스트 (일반 입력 또는 STT raw 텍스트)
    is_stt        : True이면 doc_text를 STT raw로 간주하고 LLM 보정 후 채점
    """
    topic_summary: str
    topic_desc:    str
    topic_tags:    List[str]
    doc_text:      str
    is_stt:        bool = False


@app.post("/score")
def score(req: ScoreRequest):
    """
    채점 엔드포인트.

    처리 흐름:
      1. is_stt=True이면:
           stt_normalizer.normalize_stt()로 구어체 → 문어체 보정
           보정된 normalized_doc을 doc_text로 대체
           보정 내역(stt_normalization)을 결과에 포함
      2. service_scorer.predict()로 채점 수행
      3. 결과 반환

    응답 구조:
      on_topic          : 주제 적합 여부 (bool)
      scores            : { final, topic, quality } (0~100)
      keywords          : TF-IDF 핵심 키워드 5개
      evidence          : { on_topic_sentences, off_topic_sentences }
      sentence_analysis : 문장별 분석 리스트
      worst_sentence    : topic_score 최하위 문장 1개
      stt_normalization : STT 보정 내역 (is_stt=True일 때만 포함)
    """
    doc_text         = req.doc_text
    stt_normalization = None

    if req.is_stt:
        # STT 보정: LLM으로 구어체 → 완전한 문어체 문장으로 변환
        stt_result        = normalize_stt(doc_text, llm_call)
        doc_text          = stt_result["normalized_doc"]   # 보정된 텍스트로 교체
        stt_normalization = stt_result["sentences"]        # 문장별 보정 내역 저장

    result = scorer.predict(
        topic_summary=req.topic_summary,
        topic_desc=req.topic_desc,
        topic_tags=req.topic_tags,
        doc_text=doc_text,
        debug=False,
    )

    if stt_normalization is not None:
        result["stt_normalization"] = stt_normalization

    return result
