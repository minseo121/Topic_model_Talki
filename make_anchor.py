# ============================================================
# make_anchor.py
# 역할: data_making.py에서 생성된 train/valid/test.tsv를 기반으로
#       MNR Loss 학습용 triplet 파일을 재생성하는 스크립트.
#
# data_making.py와의 차이점:
#   - data_making.py: anchor가 ANCHORS 딕셔너리의 고정 문장 하나뿐
#   - make_anchor.py: anchors.json에 저장된 여러 anchor 후보 중 랜덤 선택
#                     + 태그 증강(드랍/1개/셔플)으로 anchor 다양성 확보
#
# 왜 anchor 다양성이 중요하냐:
#   서비스에서는 사용자가 직접 주제 요약/설명/태그를 입력함.
#   학습 시 anchor가 하나의 고정 문장이면 그 표현에만 과적합됨.
#   다양한 표현의 anchor를 써서 학습해야 실서비스 입력에도 잘 일반화됨.
#
# 왜 triplets_train.tsv만 생성하냐 (valid/test triplet 미생성):
#   triplet 포맷은 MNR Loss 학습 전용이라 train에만 필요함.
#   valid/test 평가는 eval_retrieval.py에서 test.tsv + anchors.json을
#   직접 사용하는 방식으로 수행 → 서비스 입력 형식 그대로 평가 가능.
#
#   또한 데이터셋 규모(241개)와 학습 epoch(3회)가 작고,
#   이미 한국어를 충분히 학습한 KoSimCSE 모델을 도메인에 맞게
#   살짝 파인튜닝하는 수준이라 과적합 위험이 낮아
#   학습 중 valid 검증(early stopping)이 따로 필요하지 않음.
#
# 주의: train/valid/test.tsv 분할은 건드리지 않음 (data_making.py 역할).
# ============================================================

import csv
import json
import random
from pathlib import Path
from collections import defaultdict

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR  = BASE_DIR / "out"

ANCHORS_PATH = BASE_DIR / "anchors.json"  # 주제별 anchor 후보 목록 (summary, desc, tags)

TRAIN_PATH = OUT_DIR / "train.tsv"

TRIP_TRAIN = OUT_DIR / "triplets_train.tsv"

# ── 파라미터 ──────────────────────────────────────────────────────────────────
RANDOM_SEED = 42

# topic당 생성할 triplet 최대 수 (train만 생성)
MAX_PER_TOPIC_TRAIN = 800

# positive 샘플링 가중치: 품질 높은 에세이(label 3, 2)를 더 자주 선택
# 이유: 낮은 품질의 에세이를 positive로 학습하면 모델이 잘못된 방향으로 수렴할 수 있음
LABEL_WEIGHTS = {3: 4, 2: 2, 1: 1, 0: 1}

# ── 태그 증강 확률 ────────────────────────────────────────────────────────────
# 서비스에서 사용자가 태그를 아예 안 쓰거나 일부만 쓰는 경우를 학습에 반영
P_DROP_TAGS    = 0.25  # 25% 확률로 태그 전부 제거 (태그 없이 입력하는 경우 대비)
P_ONE_TAG      = 0.20  # 20% 확률로 태그 1개만 사용 (태그 일부만 입력하는 경우 대비)
P_SHUFFLE_TAGS = 0.10  # 10% 확률로 태그 순서 섞기 (태그 순서에 과적합 방지)


def load_rows(path: Path):
    """
    TSV 파일에서 에세이 데이터 로드.
    각 행: id(int), topic_id(str), text(str), label(int)
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            r["id"]    = int(r["id"])
            r["label"] = int(r["label"])
            rows.append(r)
    return rows


def write_triplets(path: Path, triplets):
    """
    (anchor, positive, negative, topic_id) triplet을 TSV로 저장.
    - anchor  : 주제 설명 텍스트 (anchors.json에서 생성)
    - positive: 같은 주제 에세이
    - negative: 다른 주제 에세이
    - topic_id: 학습 시 참고용 (MNR Loss에서 직접 사용하지는 않음)
    """
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["anchor", "positive", "negative", "topic_id"], delimiter="\t")
        w.writeheader()
        w.writerows(triplets)


def weighted_choice(cands):
    """
    label 점수에 비례한 가중 샘플링.
    LABEL_WEIGHTS에 따라 높은 점수의 에세이를 positive로 더 자주 선택.
    """
    weights = [LABEL_WEIGHTS.get(r["label"], 1) for r in cands]
    return random.choices(cands, weights=weights, k=1)[0]


def format_anchor(summary: str, desc: str, tags: list[str]) -> str:
    """
    주제 정보(summary, desc, tags)를 서비스 입력 형식의 텍스트로 변환.
    service_scorer.py의 build_anchor_text()와 동일한 포맷 사용 (학습-서비스 일관성).

    출력 예시:
      [요약] 직업 선택의 중요성
      [설명] 직업을 고를 때 고려해야 할 조건들
      [태그] #직업 #선택 #가치관
    """
    tags = [t.lstrip("#").strip() for t in tags if t and t.strip()]
    tags = [f"#{t}" for t in tags][:3]   # 최대 3개 태그만 사용
    tags_str = " ".join(tags) if tags else ""

    lines = [
        f"[요약] {summary.strip()}",
        f"[설명] {desc.strip()}",
        f"[태그] {tags_str}".rstrip(),
    ]
    return "\n".join(lines).strip()


def augment_tags(tags: list[str]) -> list[str]:
    """
    태그 증강: 서비스에서 사용자가 태그를 다양하게 입력하는 상황을 시뮬레이션.

    순서대로 적용:
      1. P_DROP_TAGS(25%) 확률로 태그 전부 제거
      2. P_ONE_TAG(20%) 확률로 태그 1개만 랜덤 선택
      3. P_SHUFFLE_TAGS(10%) 확률로 태그 순서 섞기
    """
    tags = list(tags or [])

    # 1) 태그 전부 드랍
    if random.random() < P_DROP_TAGS:
        return []

    # 2) 태그 1개만 남기기
    if tags and random.random() < P_ONE_TAG:
        return [random.choice(tags)]

    # 3) 태그 순서 섞기
    if len(tags) >= 2 and random.random() < P_SHUFFLE_TAGS:
        random.shuffle(tags)

    return tags[:3]  # 최대 3개


def pick_anchor_for_topic(anchors_by_topic: dict, topic_id: str) -> str:
    """
    anchors.json에서 해당 topic의 anchor 후보 중 하나를 랜덤 선택 후 포맷.

    data_making.py의 고정 ANCHORS 문장과 달리, 여러 후보 중 랜덤 선택 +
    태그 증강을 적용해서 triplet마다 다른 anchor 표현이 사용됨.

    fallback: anchors.json에 해당 topic이 없으면 topic_id로 최소한의 텍스트 생성
    """
    candidates = anchors_by_topic.get(topic_id, [])
    if not candidates:
        # anchors.json에 없는 topic일 경우 최소 텍스트로 대체
        return format_anchor(
            summary=f"{topic_id} 관련 발표",
            desc=f"{topic_id} 주제에 대한 발표 내용이다.",
            tags=[topic_id]
        )

    item    = random.choice(candidates)           # 후보 중 랜덤 선택
    summary = item.get("summary", "").strip()
    desc    = item.get("desc", "").strip()
    tags    = augment_tags(item.get("tags", []))  # 태그 증강 적용

    return format_anchor(summary, desc, tags)


def make_triplets(rows, anchors_by_topic, max_per_topic: int, seed: int):
    """
    (anchor, positive, negative) triplet 생성.

    MNR Loss 학습 메커니즘:
      - anchor와 positive는 가까워지도록, anchor와 negative는 멀어지도록 학습
      - negative는 배치 내 다른 샘플이 자동으로 활용됨 (MNR Loss의 핵심)
        → triplet 파일에 negative 컬럼이 있어도 train_topic.py에서는 anchor-positive만 사용

    생성 개수:
      - topic당 min(max_per_topic, max(50, 데이터수 × 5))개
        → 소규모 데이터도 최소 50개 triplet 보장
    """
    random.seed(seed)

    by_topic = defaultdict(list)
    for r in rows:
        by_topic[r["topic_id"]].append(r)

    topics   = list(by_topic.keys())
    triplets = []

    for t in topics:
        cands = by_topic[t]
        if len(cands) < 2:
            continue  # 데이터가 1개 이하면 스킵

        n = min(max_per_topic, max(50, len(cands) * 5))

        for _ in range(n):
            # anchor: anchors.json에서 랜덤 선택 + 태그 증강
            anchor = pick_anchor_for_topic(anchors_by_topic, t)

            # positive: 같은 주제, 품질 높은 에세이 우대 샘플링
            pos = weighted_choice(cands)["text"]

            # negative: 완전히 다른 주제의 에세이
            neg_topic = random.choice([x for x in topics if x != t])
            neg = weighted_choice(by_topic[neg_topic])["text"]

            triplets.append({
                "anchor":   anchor,
                "positive": pos,
                "negative": neg,
                "topic_id": t,
            })

    random.shuffle(triplets)  # 주제별로 뭉쳐 있으면 배치 편향 발생 → 셔플
    return triplets


def main():
    """
    triplet 파일 재생성 파이프라인.

    전제조건: data_making.py를 먼저 실행해서 train.tsv가 있어야 함.

    출력:
      out/triplets_train.tsv : MNR Loss 학습용 (train_topic.py에서 사용)

    valid/test 평가는 eval_retrieval.py가 test.tsv + anchors.json을 직접 사용.
    triplet 포맷 변환 없이 서비스 입력 형식 그대로 평가하므로 별도 triplet 불필요.
    """
    random.seed(RANDOM_SEED)

    if not ANCHORS_PATH.exists():
        raise FileNotFoundError(f"anchors.json 못 찾음: {ANCHORS_PATH}")

    # anchors.json 로드: { topic_id: [ {summary, desc, tags}, ... ] }
    anchors_by_topic = json.loads(ANCHORS_PATH.read_text(encoding="utf-8"))

    train_rows = load_rows(TRAIN_PATH)

    trip_train = make_triplets(train_rows, anchors_by_topic, MAX_PER_TOPIC_TRAIN, seed=42)

    write_triplets(TRIP_TRAIN, trip_train)

    print("DONE ✅")
    print("triplets_train:", len(trip_train))
    print("saved to:", TRIP_TRAIN)


if __name__ == "__main__":
    main()
