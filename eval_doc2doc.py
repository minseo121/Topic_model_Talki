# ============================================================
# eval_doc2doc.py
# 역할: 문서 간 유사도(Doc-to-Doc) 검색 성능을 평가하는 스크립트.
#
# eval_retrieval.py와의 차이점:
#   - eval_retrieval.py : anchor(주제 설명) → document 검색
#   - eval_doc2doc.py   : document → document 검색
#     "이 에세이와 같은 주제의 다른 에세이를 얼마나 잘 찾는가"를 측정
#
# 활용 목적:
#   anchor 없이 문서 자체만으로 주제 클러스터링이 잘 되는지 확인.
#   모델이 주제별 의미 공간을 잘 분리했는지 검증하는 보조 평가.
#
# 평가 지표:
#   - Recall@K  : 상위 K개 유사 문서 안에 같은 주제 문서가 있는 비율
#   - MRR       : 정답 문서가 몇 번째에 등장하는지의 역수 평균
# ============================================================

import csv
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
OUT_DIR    = BASE_DIR / "out"
MODEL_PATH = BASE_DIR / "topic_model_mnr"
TEST_PATH  = OUT_DIR  / "test.tsv"

K_LIST = [1, 5, 10]  # Recall@1, @5, @10 측정


def load_rows(path):
    """test.tsv에서 (topic_id, text) 쌍 로드."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append((r["topic_id"].strip(), r["text"]))
    return rows


def main():
    model = SentenceTransformer(str(MODEL_PATH))
    rows  = load_rows(TEST_PATH)

    topics = [t for t, _ in rows]
    texts  = [x for _, x in rows]

    # ── 전체 문서 임베딩 ──────────────────────────────────────────────────────
    embs = model.encode(
        texts, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=True
    )

    # ── 문서 간 코사인 유사도 행렬 계산 ──────────────────────────────────────
    # sims[i][j]: 문서 i와 문서 j의 유사도
    sims = embs @ embs.T
    np.fill_diagonal(sims, -1e9)  # 자기 자신과의 유사도는 제외 (항상 1.0이므로)

    # ── 각 문서에 대해 Recall@K, MRR 계산 ────────────────────────────────────
    recalls = {k: [] for k in K_LIST}
    mrrs    = []

    for i in range(len(rows)):
        order         = np.argsort(-sims[i])     # 유사도 내림차순 정렬
        ranked_topics = [topics[j] for j in order]

        # 같은 topic의 문서가 처음 등장한 rank 탐색
        first_rank = None
        for rank, t in enumerate(ranked_topics, start=1):
            if t == topics[i]:
                first_rank = rank
                break
        mrrs.append(0.0 if first_rank is None else 1.0 / first_rank)

        for k in K_LIST:
            hit = any(t == topics[i] for t in ranked_topics[:k])
            recalls[k].append(1.0 if hit else 0.0)

    # ── 결과 출력 ─────────────────────────────────────────────────────────────
    print("\n=== Doc-to-Doc Retrieval ===")
    for k in K_LIST:
        print(f"Recall@{k}: {np.mean(recalls[k]):.4f}")
    print(f"MRR: {np.mean(mrrs):.4f}")


if __name__ == "__main__":
    main()
