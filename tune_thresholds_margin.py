# ============================================================
# tune_thresholds_margin.py
# 역할: margin 값을 0~3 점수로 구간화할 최적 threshold를 탐색하는 스크립트.
#
# margin이란:
#   margin = s_pos - s_neg (정답 topic 유사도 - 다른 topic 중 최대 유사도)
#   margin이 클수록 해당 주제에 명확히 부합하는 문서.
#
# 왜 threshold 튜닝이 필요하냐:
#   margin을 0~3 점수로 나눌 때 어느 값을 기준으로 나눌지 정해야 함.
#   단순히 균등 분할하면 실제 데이터 분포를 반영하지 못할 수 있음.
#   → valid 데이터에서 grid search로 최적 threshold(t1, t2, t3)를 찾고
#     test 데이터에 적용해서 성능 확인.
#
# 탐색 방법:
#   - margin 분포의 퍼센타일(5~95) 기준으로 19개 후보값 생성
#   - (t1, t2, t3) 3개 threshold 조합을 grid search
#   - macro F1 최대화 기준으로 best 조합 선택
#     (macro F1을 쓰는 이유: 클래스 불균형에 accuracy보다 강건함)
#
# 참고: 현재 service_scorer.py는 이 방식(margin 기반)이 아닌
#       코사인 유사도 직접 캘리브레이션 방식을 사용하고 있음.
#       이 스크립트는 초기 설계 검증용으로 보존.
# ============================================================

import csv
import json
from pathlib import Path
from collections import defaultdict

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
    """서비스 입력 형식의 anchor 텍스트 생성."""
    tags     = [t.lstrip("#").strip() for t in (tags or []) if t and t.strip()]
    tags     = [f"#{t}" for t in tags][:3]
    tags_str = " ".join(tags) if tags else ""
    return "\n".join([
        f"[요약] {summary.strip()}",
        f"[설명] {desc.strip()}",
        f"[태그] {tags_str}".rstrip(),
    ]).strip()


def build_topic_prototypes(model: SentenceTransformer, anchors_by_topic: dict):
    """topic별 prototype 벡터 생성 (anchor 임베딩 평균 → L2 정규화)."""
    topics = sorted(anchors_by_topic.keys())
    protos = {}

    for t in topics:
        items = anchors_by_topic.get(t, [])
        if not items:
            texts = [format_anchor(f"{t} 관련 발표", f"{t} 주제의 발표 내용이다.", [t])]
        else:
            texts = [format_anchor(it.get("summary", ""), it.get("desc", ""), it.get("tags", [])) for it in items]

        embs  = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        proto = embs.mean(axis=0)
        proto = proto / (np.linalg.norm(proto) + 1e-12)
        protos[t] = proto

    return topics, protos


def margins_for_rows(rows, model, topics, protos):
    """
    각 문서의 margin 계산.
      margin = s_pos - s_neg
      s_pos: 정답 topic prototype과의 유사도
      s_neg: 다른 topic prototype 중 가장 높은 유사도
    """
    texts    = [r["text"] for r in rows]
    doc_embs = model.encode(
        texts, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=True
    )

    proto_mat    = np.stack([protos[t] for t in topics], axis=0)  # (T, D)
    sims         = doc_embs @ proto_mat.T                          # (N, T)
    topic_to_idx = {t: i for i, t in enumerate(topics)}
    y_topic      = np.array([topic_to_idx[r["topic_id"]] for r in rows], dtype=int)

    s_pos      = sims[np.arange(len(rows)), y_topic]
    sims_copy  = sims.copy()
    sims_copy[np.arange(len(rows)), y_topic] = -1e9
    s_neg      = np.max(sims_copy, axis=1)

    margin  = s_pos - s_neg
    y_label = np.array([r["label"] for r in rows], dtype=int)
    return margin, y_label


def bucketize(x, t1, t2, t3):
    """
    margin 값을 threshold 3개로 0~3 구간으로 분류.
      x < t1  → 0점
      t1 ≤ x < t2 → 1점
      t2 ≤ x < t3 → 2점
      x ≥ t3  → 3점
    """
    out      = np.zeros_like(x, dtype=int)
    out[x >= t1] = 1
    out[x >= t2] = 2
    out[x >= t3] = 3
    return out


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def macro_f1(y_true, y_pred, num_classes=4):
    """
    클래스별 F1 평균 계산.
    불균형 데이터에서 accuracy보다 신뢰성 높은 지표.
    threshold 최적화 기준으로 사용.
    """
    f1s = []
    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        if tp == 0 and (fp > 0 or fn > 0):
            f1s.append(0.0)
            continue
        if tp == 0 and fp == 0 and fn == 0:
            f1s.append(1.0)
            continue
        prec = tp / (tp + fp + 1e-12)
        rec  = tp / (tp + fn + 1e-12)
        f1s.append(2 * prec * rec / (prec + rec + 1e-12))
    return float(np.mean(f1s))


def confusion(y_true, y_pred, num_classes=4):
    """혼동 행렬 계산 (행=정답, 열=예측)."""
    mat = np.zeros((num_classes, num_classes), dtype=int)
    for a, b in zip(y_true, y_pred):
        mat[a, b] += 1
    return mat


def tune_thresholds(margin, y_label):
    """
    (t1, t2, t3) threshold 조합 grid search.

    후보 생성: margin 분포의 퍼센타일 5~95 구간에서 19개 균등 추출
    평가 기준: macro F1 최대화
    탐색 범위: 19C3 = 969가지 조합
    """
    qs   = np.linspace(5, 95, 19)
    cand = np.percentile(margin, qs)

    best       = None
    best_score = -1

    for i in range(len(cand)):
        for j in range(i + 1, len(cand)):
            for k in range(j + 1, len(cand)):
                t1, t2, t3 = float(cand[i]), float(cand[j]), float(cand[k])
                pred        = bucketize(margin, t1, t2, t3)
                score       = macro_f1(y_label, pred)
                if score > best_score:
                    best_score = score
                    best       = (t1, t2, t3)

    return best, best_score


def main():
    model            = SentenceTransformer(str(MODEL_PATH))
    anchors_by_topic = json.loads(ANCHORS_PATH.read_text(encoding="utf-8"))
    topics, protos   = build_topic_prototypes(model, anchors_by_topic)

    valid_rows = load_rows(VALID_PATH)
    test_rows  = load_rows(TEST_PATH)

    # ── valid에서 최적 threshold 탐색 ────────────────────────────────────────
    print("\n=== Threshold tuning on VALID (margin = pos - best_other) ===")
    m_valid, yv   = margins_for_rows(valid_rows, model, topics, protos)
    (t1, t2, t3), f1v = tune_thresholds(m_valid, yv)

    pred_v = bucketize(m_valid, t1, t2, t3)
    print(f"best thresholds: t1={t1:.4f}, t2={t2:.4f}, t3={t3:.4f}")
    print(f"[VALID] acc={accuracy(yv, pred_v):.4f} macroF1={macro_f1(yv, pred_v):.4f}")
    print("[VALID] confusion rows=true cols=pred:\n", confusion(yv, pred_v))

    # ── 같은 threshold를 test에 적용해서 일반화 성능 확인 ────────────────────
    print("\n=== Apply thresholds to TEST ===")
    m_test, yt = margins_for_rows(test_rows, model, topics, protos)
    pred_t     = bucketize(m_test, t1, t2, t3)
    print(f"[TEST ] acc={accuracy(yt, pred_t):.4f} macroF1={macro_f1(yt, pred_t):.4f}")
    print("[TEST ] confusion rows=true cols=pred:\n", confusion(yt, pred_t))


if __name__ == "__main__":
    main()
