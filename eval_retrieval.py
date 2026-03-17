# ============================================================
# eval_retrieval.py
# 역할: 파인튜닝된 topic_model_mnr의 검색 성능을 평가하는 스크립트.
#
# 평가 방식: Anchor → Document Retrieval
#   주제 설명(anchor)을 쿼리로, 전체 test 문서들을 검색 대상으로 삼아
#   "같은 주제의 문서가 상위 K개 안에 들어오는가"를 측정.
#
# 평가 지표:
#   - Recall@K  : 상위 K개 결과 안에 정답(같은 주제 문서)이 있는 비율
#                 K=1, 5, 10으로 측정. 현재 Recall@10 약 88% 달성.
#   - MRR       : Mean Reciprocal Rank. 정답이 몇 번째에 등장하는지의 역수 평균.
#                 1.0에 가까울수록 정답이 항상 1위에 등장한다는 의미.
#
# 왜 triplet 파일 대신 test.tsv + anchors.json을 직접 쓰냐:
#   서비스에서 실제로 들어오는 입력 형식(summary/desc/tags)으로 평가해야
#   실제 서비스 성능을 정확히 측정할 수 있음.
#   triplet 포맷으로 평가하면 학습 데이터 형식으로만 검증하는 셈이라 의미가 낮음.
# ============================================================

import csv
import json
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR  = BASE_DIR / "out"

MODEL_PATH   = BASE_DIR / "topic_model_mnr"   # 평가할 파인튜닝 모델
TEST_PATH    = OUT_DIR  / "test.tsv"          # 검색 대상 문서 (id, topic_id, text, label)
ANCHORS_PATH = BASE_DIR / "anchors.json"      # 주제별 anchor 후보 (서비스 입력 형식)

# ── 평가 설정 ─────────────────────────────────────────────────────────────────
K_LIST           = [1, 5, 10]   # Recall@1, Recall@5, Recall@10 측정
ANCHORS_PER_TOPIC = 30          # topic당 anchor를 30개 샘플링해서 평균 성능 측정
                                 # anchors.json에 후보가 적으면 중복 허용해서 30개 맞춤
RANDOM_SEED = 42

# 평가 시 태그 증강 미적용 (학습 때만 증강, 평가는 원본 그대로)
# 평가에서 증강을 적용하면 실행마다 결과가 달라져 재현성이 없어짐
P_DROP_TAGS    = 1.0  # 태그 전부 제거 (증강 없이 태그 없는 버전으로 통일)
P_ONE_TAG      = 0.0
P_SHUFFLE_TAGS = 0.0


def load_rows(path: Path):
    """test.tsv에서 문서 데이터 로드."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append({
                "id":       int(r["id"]),
                "topic_id": r["topic_id"],
                "text":     r["text"],
                "label":    int(r["label"]),
            })
    return rows


def format_anchor(summary: str, desc: str, tags: list[str]) -> str:
    """
    주제 정보를 서비스 입력 형식 텍스트로 변환.
    service_scorer.py의 build_anchor_text()와 동일한 포맷 (학습-평가-서비스 일관성).
    """
    tags     = [t.lstrip("#").strip() for t in (tags or []) if t and t.strip()]
    tags     = [f"#{t}" for t in tags][:3]
    tags_str = " ".join(tags) if tags else ""
    return "\n".join([
        f"[요약] {summary.strip()}",
        f"[설명] {desc.strip()}",
        f"[태그] {tags_str}".rstrip(),
    ]).strip()


def augment_tags(tags):
    """
    평가 시 태그 처리. P_DROP_TAGS=1.0으로 설정되어 있어 태그를 전부 제거.
    (평가 재현성을 위해 증강 비활성화)
    """
    tags = list(tags or [])
    r = random.random()
    if r < P_DROP_TAGS:
        return []
    if tags and random.random() < P_ONE_TAG:
        return [random.choice(tags)]
    if len(tags) >= 2 and random.random() < P_SHUFFLE_TAGS:
        random.shuffle(tags)
    return tags[:3]


def sample_anchors(anchors_by_topic: dict, topic_id: str, n: int):
    """
    anchors.json에서 해당 topic의 anchor를 n개 샘플링.
    후보가 n개보다 적으면 중복 허용해서 n개를 맞춤.
    여러 개의 anchor로 평가하는 이유: 특정 표현에만 잘 작동하는 게 아닌지 확인.
    """
    cands = anchors_by_topic.get(topic_id, [])
    if not cands:
        return ["무작위 텍스트입니다. 의미 없습니다."] * n

    out = []
    for _ in range(n):
        item    = random.choice(cands)
        summary = item.get("summary", "")
        desc    = item.get("desc", "")
        tags    = augment_tags(item.get("tags", []))
        out.append(format_anchor(summary, desc, tags))
    return out


def eval_for_topic(topic_id, anchor_embs, doc_embs, doc_topic_ids, k_list):
    """
    특정 topic에 대해 Recall@K와 MRR을 계산.

    동작:
      1. anchor 임베딩과 전체 문서 임베딩의 코사인 유사도 계산
      2. 유사도 내림차순으로 문서 정렬
      3. 상위 K개 안에 같은 topic 문서가 있으면 hit
      4. 정답이 처음 등장한 순위의 역수 = RR (Reciprocal Rank)

    anchor 30개 각각에 대해 측정 후 평균 반환.
    """
    # sims: (num_anchors, num_docs) 유사도 행렬
    sims = anchor_embs @ doc_embs.T

    recalls = {k: [] for k in k_list}
    mrrs    = []

    for i in range(sims.shape[0]):
        order         = np.argsort(-sims[i])  # 유사도 내림차순 정렬
        ranked_topics = [doc_topic_ids[j] for j in order]

        # 정답(같은 topic)이 처음 등장한 rank 탐색
        first_rank = None
        for rank, t in enumerate(ranked_topics, start=1):
            if t == topic_id:
                first_rank = rank
                break
        mrrs.append(0.0 if first_rank is None else 1.0 / first_rank)

        for k in k_list:
            hit = any(t == topic_id for t in ranked_topics[:k])
            recalls[k].append(1.0 if hit else 0.0)

    return {k: float(np.mean(recalls[k])) for k in k_list}, float(np.mean(mrrs))


def main():
    random.seed(RANDOM_SEED)

    # ── 모델 및 데이터 로드 ───────────────────────────────────────────────────
    model            = SentenceTransformer(str(MODEL_PATH))
    test_rows        = load_rows(TEST_PATH)
    anchors_by_topic = json.loads(ANCHORS_PATH.read_text(encoding="utf-8"))

    # ── 전체 test 문서 임베딩 (한 번만 계산) ─────────────────────────────────
    doc_texts     = [r["text"]     for r in test_rows]
    doc_topic_ids = [r["topic_id"] for r in test_rows]
    doc_embs = model.encode(
        doc_texts, batch_size=32,
        convert_to_numpy=True, normalize_embeddings=True,
        show_progress_bar=True
    )

    # ── topic별 anchor 임베딩 생성 및 평가 ───────────────────────────────────
    topics     = sorted(set(doc_topic_ids))
    per_topic  = {}
    all_recalls = {k: [] for k in K_LIST}
    all_mrr     = []

    for t in topics:
        # topic당 ANCHORS_PER_TOPIC(30)개 anchor 샘플링 후 임베딩
        anchors     = sample_anchors(anchors_by_topic, t, ANCHORS_PER_TOPIC)
        anchor_embs = model.encode(
            anchors, batch_size=32,
            convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False
        )

        recalls_t, mrr_t     = eval_for_topic(t, anchor_embs, doc_embs, doc_topic_ids, K_LIST)
        per_topic[t]          = {"recall": recalls_t, "mrr": mrr_t}

        for k in K_LIST:
            all_recalls[k].append(recalls_t[k])
        all_mrr.append(mrr_t)

    # ── 결과 출력 ─────────────────────────────────────────────────────────────
    print("\n=== Overall (macro over topics) ===")
    for k in K_LIST:
        print(f"Recall@{k}: {np.mean(all_recalls[k]):.4f}")
    print(f"MRR: {np.mean(all_mrr):.4f}")

    print("\n=== Per topic ===")
    for t in topics:
        rs = per_topic[t]["recall"]
        print(f"[{t}] " + " ".join([f"R@{k}={rs[k]:.4f}" for k in K_LIST]) + f"  MRR={per_topic[t]['mrr']:.4f}")


if __name__ == "__main__":
    main()
