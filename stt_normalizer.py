# ============================================================
# stt_normalizer.py
# 역할: STT(음성 인식)로 변환된 구어체 텍스트를 완전한 문어체 문장으로 보정.
#
# 왜 필요하냐:
#   음성 인식 결과에는 필러 단어("어~", "그래서 뭐냐", "막", "그냥"),
#   불완전한 문장, 구어체 표현이 섞여 있음.
#   이 상태로 service_scorer.py에 넣으면 임베딩 품질이 낮아져 채점이 부정확해짐.
#   → LLM으로 문장을 먼저 정제한 뒤 채점에 사용.
#
# 설계 방식 (LLM 주입):
#   이 모듈은 특정 LLM API에 종속되지 않음.
#   normalize_stt(raw_stt, llm_call_fn) 형태로
#   실제 LLM 호출 함수를 외부에서 주입받아 사용.
#   → main.py에서 GPT든 Claude든 동일한 시그니처로 감싸서 넘기면 됨.
#
# llm_call_fn 시그니처: (prompt: str) -> str
#   예) GPT:   lambda p: openai_client.chat.completions.create(...).choices[0].message.content
#   예) Claude: lambda p: anthropic_client.messages.create(...).content[0].text
# ============================================================

import json
import re
from typing import Any, Callable, Dict, List


def _split_rough_segments(text: str) -> List[str]:
    """
    STT 원문을 대략적인 문장 단위로 분리.

    완벽한 문장 분리가 아닌 "대략적인 청크" 수준.
    이후 LLM이 각 청크를 완전한 문장으로 보정하므로 정확도보다 속도 우선.

    분리 기준:
      - 마침표/느낌표/물음표 뒤 공백
      - '다', '요', '죠' 뒤 공백 (한국어 종결어미)
    """
    text  = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+|(?<=다)\s+|(?<=요)\s+|(?<=죠)\s+", text)
    return [p.strip() for p in parts if p.strip()]


def build_normalize_prompt(segments: List[str]) -> str:
    """
    STT 보정용 LLM 프롬프트 생성.

    segments를 JSON 배열로 직렬화해서 프롬프트에 삽입.
    LLM에게 각 segment에 대해:
      1. 불완전 문장 → 완전한 문어체 문장으로 복원
      2. 필러 단어 제거
      3. 문법 오류 여부 및 이슈 서술
    을 요청.

    반환된 JSON 파싱은 normalize_stt()에서 처리.
    """
    segments_json = json.dumps(segments, ensure_ascii=False)
    return f"""다음은 음성 인식(STT)으로 변환된 한국어 텍스트 조각들입니다.
구어체 표현, 불완전한 문장, 필러 단어(어, 그래서, 뭐냐, 막, 그냥 등)가 포함될 수 있습니다.

각 조각에 대해:
1. 불완전한 문장을 문맥에 맞게 완전한 문어체 문장으로 복원
2. 필러 단어 제거
3. 문법 오류 여부 및 이슈 서술

STT 조각 목록 (JSON 배열):
{segments_json}

다음 JSON 형식으로만 응답하세요. 다른 설명 없이 JSON만 출력하세요:
{{
  "sentences": [
    {{
      "original": "원본 조각",
      "normalized": "보정된 완전한 문장",
      "grammar_ok": true,
      "grammar_issues": []
    }}
  ]
}}"""


def normalize_stt(raw_stt: str, llm_call_fn: Callable[[str], str]) -> Dict[str, Any]:
    """
    STT raw 텍스트를 완전한 문장들로 보정.

    Args:
        raw_stt:      STT 원문 텍스트 (구어체, 불완전 문장 포함)
        llm_call_fn:  프롬프트(str)를 받아 응답(str)을 반환하는 함수.
                      GPT, Claude 등 어떤 LLM이든 이 시그니처에 맞게 감싸서 넘기면 됨.
                      구현 예시는 main.py의 llm_call() 주석 참고.

    반환:
      {
        "sentences": [
          {
            "original":       str,        # STT 원문 조각
            "normalized":     str,        # LLM이 보정한 완전한 문장
            "grammar_ok":     bool,       # 문법 이상 없으면 True
            "grammar_issues": List[str]   # 문법 이슈 설명 (없으면 빈 리스트)
          },
          ...
        ],
        "normalized_doc": str  # 보정된 문장들을 이어붙인 전체 텍스트
                               # → service_scorer.predict()의 doc_text로 전달
      }

    LLM 파싱 실패 시:
      원문을 그대로 반환 (서비스 중단 방지용 fallback)
    """
    segments = _split_rough_segments(raw_stt)
    if not segments:
        return {"sentences": [], "normalized_doc": ""}

    prompt       = build_normalize_prompt(segments)
    raw_response = llm_call_fn(prompt).strip()

    # LLM이 ```json ... ``` 형태로 감싸서 반환할 수 있으므로 JSON만 추출
    json_match = re.search(r"\{[\s\S]*\}", raw_response)
    if not json_match:
        # JSON 파싱 실패 시 원문 그대로 반환 (서비스 중단 방지)
        fallback = [
            {
                "original":       seg,
                "normalized":     seg,
                "grammar_ok":     True,
                "grammar_issues": [],
            }
            for seg in segments
        ]
        return {
            "sentences":      fallback,
            "normalized_doc": " ".join(segments),
        }

    result                   = json.loads(json_match.group())
    result["normalized_doc"] = " ".join(
        s["normalized"] for s in result["sentences"]
    )
    return result
