# ============================================================
# eval_anchor2doc_topicacc.py
# 역할: anchor → 문서 주제 분류 정확도(Topic Accuracy)를 평가하는 스크립트.
#
# eval_retrieval.py와의 차이점:
#   - eval_retrieval.py       : Recall@K 방식 (상위 K개 안에 정답이 있는가)
#   - eval_anchor2doc_topicacc: 단순 분류 정확도 (가장 유사한 주제가 정답인가)
#
# 평가 방식: Prototype 기반 주제 분류
#   1. anchors.json에서 topic별 anchor 여러 개의 임베딩 평균 → prototype 벡터
#   2. 문서 임베딩과 모든 prototype의 유사도 계산
#   3. 가장 유사한 prototype의 topic이 정답 topic과 일치하면 정답
#
# prototype이란:
#   각 주제를 대표하는 평균 임베딩 벡터.
#   anchor 여러 개의 임베딩을 평균내서 만들기 때문에
#   특정 표현에 치우치지 않고 주제의 "중심"을 잘 표현함.
#
# valid와 test 양쪽 모두 평가해서 일반화 성능 확인.
# ============================================================

import csv
import json
from pathlib import Path
from collections import Counter

import numpy as np
from sentence_transformers import SentenceTransformer

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent
OUT_DIR      = BASE_DIR / "out"

MODEL_PATH   = BASE_DIR / "topic_model_mnr"
VALID_PATH   = OUT_DIR  / "valid.tsv"
TEST_PATH    = OUT_DIR  / "test.tsv"
ANCHORS_PATH = BASE_DIR / "anchors.json"


def load_rows(path: Path):
    """TSV에서 topic_id, text, label 로드."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append({
                "topic_id": r["topic_id"].strip(),
                "text":     r["text"],
                "label":    int(r["label"]),
            })
    return rows


def format_anchor(summary: str, desc: str, tags: list[str]) -> str:
    """
    서비스 입력 형식의 anchor 텍스트 생성.
    service_scorer.py의 build_anchor_text()와 동일한 포맷 유지.
    """
    tags     = [t.lstrip("#").strip() for t in (tags or []) if t and t.strip()]
    tags     = [f"#{t}" for t in tags][:3]
    tags_str = " ".join(tags) if tags else ""
    return "\n".join([
        f"[요약] {summary.strip()}",
        f"[설명] {desc.strip()}",
        f"[태그] {tags_str}".rstrip(),
    ]).strip()


def build_topic_prototypes(model: SentenceTransformer, anchors_by_topic: dict):
    """
    topic별 prototype 벡터 생성.

    각 topic의 anchor 텍스트들을 임베딩 후 평균 → L2 정규화.
    정규화하는 이유: 코사인 유사도 계산 시 크기 차이 영향 제거.

    반환:
      topics  : 정렬된 topic_id 리스트
      protos  : (num_topics, embedding_dim) 형태의 prototype 행렬
    """
    topics = sorted(anchors_by_topic.keys())
    topic_texts = {}

    for t in topics:
        items = anchors_by_topic.get(t, [])
        if not items:
            # anchors.json에 해당 topic이 없으면 최소 텍스트로 대체
            topic_texts[t] = [format_anchor(f"{t} 관련 발표", f"{t} 주제의 발표 내용이다.", [t])]
        else:
            topic_texts[t] = [
                format_anchor(it.get("summary", ""), it.get("desc", ""), it.get("tags", []))
                for it in items
            ]

    # topic별 anchor 임베딩 평균 → prototype
    protos = []
    for t in topics:
        embs  = model.encode(topic_texts[t], normalize_embeddings=True, convert_to_numpy=True)
        proto = embs.mean(axis=0)
        proto = proto / (np.linalg.norm(proto) + 1e-12)  # L2 정규화
        protos.append(proto)

    protos = np.stack(protos, axis=0)  # (num_topics, dim)
    return topics, protos


def eval_topic_acc(rows, model, topics, protos):
    """
    문서들을 prototype 기반으로 주제 분류 후 정확도 계산.

    동작:
      1. 각 문서를 임베딩
      2. 모든 topic prototype과의 유사도 계산
      3. 가장 유사한 prototype의 topic을 예측 topic으로 선택
      4. 예측 topic == 정답 topic이면 정답

    반환:
      acc     : 전체 정확도
      per     : topic별 정확도 딕셔너리
      counter : (정답, 예측) 쌍의 빈도 (혼동 행렬 분석용)
    """
    texts  = [r["text"]     for r in rows]
    true   = [r["topic_id"] for r in rows]

    doc_embs = model.encode(
        texts, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=True
    )
    # sims: (num_docs, num_topics) 유사도 행렬
    sims     = doc_embs @ protos.T
    pred_idx = np.argmax(sims, axis=1)            # 각 문서에서 가장 유사한 topic index
    pred     = [topics[i] for i in pred_idx]      # index → topic_id 변환

    acc = float(np.mean([p == y for p, y in zip(pred, true)]))

    # topic별 정확도
    per = {}
    for t in sorted(set(true)):
        idx    = [i for i, y in enumerate(true) if y == t]
        per[t] = float(np.mean([pred[i] == true[i] for i in idx])) if idx else 0.0

    return acc, per, Counter(zip(true, pred))


def main():
    model            = SentenceTransformer(str(MODEL_PATH))
    anchors_by_topic = json.loads(ANCHORS_PATH.read_text(encoding="utf-8"))

    # topic별 prototype 벡터 생성
    topics, protos = build_topic_prototypes(model, anchors_by_topic)

    valid_rows = load_rows(VALID_PATH)
    test_rows  = load_rows(TEST_PATH)

    print("\n=== Anchor-Prototype Topic Prediction ===")

    # valid 평가 (과적합 여부 확인용)
    acc_v, per_v, _ = eval_topic_acc(valid_rows, model, topics, protos)
    print(f"[VALID] accuracy: {acc_v:.4f} | per-topic: {per_v}")

    # test 평가 (최종 성능 지표)
    acc_t, per_t, _ = eval_topic_acc(test_rows, model, topics, protos)
    print(f"[TEST ] accuracy: {acc_t:.4f} | per-topic: {per_t}")


if __name__ == "__main__":
    main()
