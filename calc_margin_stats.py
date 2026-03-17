# ============================================================
# calc_margin_stats.py
# 역할: valid 데이터에서 margin 분포 통계를 계산하는 스크립트.
#
# margin이란:
#   margin = s_pos - s_neg
#   s_pos: 문서와 정답 topic prototype의 유사도
#   s_neg: 문서와 가장 유사한 다른 topic prototype의 유사도
#
#   margin이 클수록 정답 topic과 명확히 구분됨 (좋은 신호).
#   margin이 0에 가까우면 두 topic이 헷갈린다는 의미.
#   margin이 음수면 다른 topic으로 잘못 분류됨.
#
# 왜 margin 통계가 필요하냐:
#   service_scorer.py의 SIM_MIN, SIM_MAX 캘리브레이션 값을 설정할 때
#   실제 데이터 분포를 보고 "의미 있는 범위"를 잡아야 함.
#   p05(하위 5%)를 0점 기준, p95(상위 95%)를 최고점 기준으로 삼는 방식.
#
# 출력 예시:
#   p05: 0.325  → 이 값 이하면 주제 적합도 0점에 가까운 수준
#   p95: 0.560  → 이 값 이상이면 주제 적합도 최고 수준
# ============================================================

import csv, json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent
OUT_DIR      = BASE_DIR / "out"
MODEL_PATH   = BASE_DIR / "topic_model_mnr"
ANCHORS_PATH = BASE_DIR / "anchors.json"
VALID_PATH   = OUT_DIR  / "valid.tsv"


def load_rows(path):
    """valid.tsv에서 (topic_id, text) 쌍 로드."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            rows.append((row["topic_id"].strip(), row["text"]))
    return rows


def format_anchor(summary, desc, tags):
    """서비스 입력 형식의 anchor 텍스트 생성."""
    tags     = [t.lstrip("#").strip() for t in (tags or []) if t and t.strip()]
    tags     = [f"#{t}" for t in tags][:3]
    tags_str = " ".join(tags) if tags else ""
    return "\n".join([f"[요약] {summary.strip()}",
                      f"[설명] {desc.strip()}",
                      f"[태그] {tags_str}".rstrip()]).strip()


def build_protos(model, anchors_by_topic):
    """
    topic별 prototype 벡터 생성.
    각 topic의 anchor 임베딩들을 평균 → L2 정규화.
    """
    topics = sorted(anchors_by_topic.keys())
    protos = []
    for t in topics:
        items = anchors_by_topic[t]
        texts = [format_anchor(it["summary"], it["desc"], it.get("tags", [])) for it in items]
        embs  = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        proto = embs.mean(axis=0)
        proto = proto / (np.linalg.norm(proto) + 1e-12)
        protos.append(proto)
    return topics, np.stack(protos, axis=0)


def main():
    topic_model      = SentenceTransformer(str(MODEL_PATH))
    anchors_by_topic = json.loads(Path(ANCHORS_PATH).read_text(encoding="utf-8"))
    topics, proto_mat = build_protos(topic_model, anchors_by_topic)
    topic_to_idx = {t: i for i, t in enumerate(topics)}

    rows = load_rows(VALID_PATH)

    # ── valid 문서 임베딩 ─────────────────────────────────────────────────────
    y_idx = np.array([topic_to_idx[t] for t, _ in rows], dtype=int)
    texts = [txt for _, txt in rows]
    doc_embs = model.encode = topic_model.encode(
        texts, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=True
    )

    # ── margin 계산 ───────────────────────────────────────────────────────────
    # sims: (num_docs, num_topics) 유사도 행렬
    sims  = doc_embs @ proto_mat.T

    s_pos = sims[np.arange(len(rows)), y_idx]          # 정답 topic과의 유사도

    sims2 = sims.copy()
    sims2[np.arange(len(rows)), y_idx] = -1e9          # 정답 topic 유사도 제거
    s_neg = np.max(sims2, axis=1)                      # 나머지 중 가장 높은 유사도

    margin = s_pos - s_neg  # 정답-오답 유사도 차이

    # ── 분포 통계 출력 ────────────────────────────────────────────────────────
    # 이 값들을 service_scorer.py의 SIM_MIN(p05), SIM_MAX(p95) 설정에 활용
    qs = [0, 5, 25, 50, 75, 95, 100]
    print("\n=== VALID margin stats ===")
    print("min/max:", float(margin.min()), float(margin.max()))
    for q in qs:
        print(f"p{q:02d}:", float(np.percentile(margin, q)))


if __name__ == "__main__":
    main()
