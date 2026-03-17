# ============================================================
# data_making.py
# 역할: 원본 에세이 데이터(TSV)를 읽어서 학습에 필요한 형태로 가공하는 전처리 스크립트
#
# 실행하면 out/ 폴더에 아래 파일들이 생성됨:
#   - train.tsv / valid.tsv / test.tsv   : 학습/검증/테스트용 분할 데이터
#   - triplets_train.tsv 등              : MNR Loss 학습용 (anchor, positive, negative) 쌍
#
# 주의: 이 스크립트는 모델 학습 전에 딱 한 번 실행하는 전처리용이며,
#       서비스 실행 시에는 관여하지 않음.
# ============================================================

import re
import csv
import random
from collections import defaultdict

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "out"
OUTPUT_DIR.mkdir(exist_ok=True)  # out/ 폴더가 없으면 자동 생성


# ── 1) 주제별 원본 데이터 파일 경로 ──────────────────────────────────────────
# 유학생 에세이 데이터셋. 각 파일은 TSV 형식(id, document, label).
# label은 0~3 점수 (0: 매우 낮음, 3: 매우 높음)
INPUTS = [
    ("job",       str(BASE_DIR / "data" / "job.txt")),        # 직업 관련 에세이 101개
    ("happiness", str(BASE_DIR / "data" / "happiness.txt")), # 행복 관련 에세이 96개
    ("economic",  str(BASE_DIR / "data" / "economic.txt")),  # 경제 관련 에세이 62개
    ("success",   str(BASE_DIR / "data" / "success.txt")),   # 성공 관련 에세이 45개
]

# ── 2) 전처리 파라미터 ────────────────────────────────────────────────────────
MIN_CHARS = 30                # 이 글자 수 미만인 텍스트는 너무 짧아서 학습에 부적합 → 제외
SPLIT = (0.8, 0.1, 0.1)      # train 80% / valid 10% / test 10% 비율로 분할
RANDOM_SEED = 42              # 재현 가능한 결과를 위한 랜덤 시드

# ── 3) 주제별 anchor 텍스트 ───────────────────────────────────────────────────
# MNR Loss 학습 시 anchor로 사용하는 주제 설명 문장.
# 이후 make_anchor.py에서는 anchors.json 기반으로 더 다양한 anchor를 사용하도록 개선됨.
# (이 ANCHORS는 단순 고정 문장이라 다양성이 없음 → make_anchor.py가 최신 버전)
ANCHORS = {
    "job":       "직업 선택과 직업관에 대한 글이다. 직업을 고를 때 중요 조건이나 일의 의미를 다룬다.",
    "happiness": "행복한 삶의 조건과 행복의 의미에 대한 글이다. 개인·사회적 요인과 삶의 태도를 다룬다.",
    "economic":  "경제적 여유와 행복, 물질과 만족의 관계에 대한 글이다. 소득·빈부·삶의 질을 다룬다.",
    "success":   "성공의 의미와 성공을 이루는 과정에 대한 글이다. 목표, 노력, 가치관을 다룬다.",
}


def clean_text(s: str) -> str:
    """
    원본 에세이 텍스트의 노이즈를 제거하고 정규화.
    - "[[문단]]" 태그: 원본 데이터에서 문단 구분자로 쓰인 태그를 공백으로 교체
    - 연속 공백: 여러 개의 공백/개행을 단일 공백으로 압축
    """
    s = s.replace("[[문단]]", " ")   # 문단 구분 태그 제거
    s = re.sub(r"\s+", " ", s).strip()  # 연속 공백 정리
    return s


def read_tsv(path: str):
    """
    TSV 파일을 한 행씩 딕셔너리로 읽어서 yield.
    원본 파일 컬럼: id, document, label
    """
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        yield from reader


def write_rows(path: str, rows):
    """
    가공된 행들을 TSV로 저장.
    출력 컬럼: id, topic_id, text, label
    id 기준으로 정렬해서 저장 (순서 일관성 보장)
    """
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "topic_id", "text", "label"], delimiter="\t")
        w.writeheader()
        for r in sorted(rows, key=lambda x: x["id"]):
            w.writerow(r)


def write_triplets(path: str, triplets):
    """
    MNR Loss 학습용 triplet 데이터를 TSV로 저장.
    출력 컬럼: anchor, positive, negative, topic_id
    - anchor  : 주제 설명 텍스트 (학습 기준점)
    - positive: 해당 주제의 에세이 (anchor와 가까워지도록 학습)
    - negative: 다른 주제의 에세이 (anchor와 멀어지도록 학습)
    """
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["anchor", "positive", "negative", "topic_id"], delimiter="\t")
        w.writeheader()
        w.writerows(triplets)


def weighted_choice(cands):
    """
    label 점수에 비례해서 가중 샘플링.
    높은 점수(label 3, 2)의 에세이를 positive로 더 자주 뽑도록 가중치 부여.
    이유: 품질 낮은 에세이(label 0, 1)를 positive로 쓰면 모델이 잘못된 방향으로 학습될 수 있음.
      label 3 → 가중치 4 (가장 우대)
      label 2 → 가중치 2
      label 0, 1 → 가중치 1
    """
    weights = []
    for r in cands:
        if r["label"] == 3:
            w = 4
        elif r["label"] == 2:
            w = 2
        else:
            w = 1
        weights.append(w)
    return random.choices(cands, weights=weights, k=1)[0]


def make_triplets(rows, seed: int, max_per_topic: int):
    """
    (anchor, positive, negative) triplet 쌍을 생성.

    MNR Loss 학습 방식:
      - anchor  : 주제 설명 고정 문장 (ANCHORS 딕셔너리에서 가져옴)
      - positive: 같은 주제의 에세이 (weighted_choice로 품질 높은 것 우대)
      - negative: 다른 주제의 에세이 (완전히 다른 topic에서 랜덤 선택)

    생성 개수:
      - topic당 min(max_per_topic, max(50, 데이터수 * 5))개 생성
        → 데이터가 적으면 최소 50개 보장, 많으면 max_per_topic 상한 적용
    """
    random.seed(seed)
    by_topic = defaultdict(list)
    for r in rows:
        by_topic[r["topic_id"]].append(r)

    topics = list(by_topic.keys())
    triplets = []

    for t in topics:
        cands = by_topic[t]
        if len(cands) < 2:
            continue  # 데이터가 1개 이하면 negative 뽑을 때 충돌 가능 → 스킵

        # 데이터 크기에 따라 생성 개수 조정
        n = min(max_per_topic, max(50, len(cands) * 5))

        for _ in range(n):
            anchor = ANCHORS[t]                                         # 주제 설명 고정 텍스트
            pos = weighted_choice(cands)["text"]                        # 같은 주제, 품질 우대 샘플링

            neg_topic = random.choice([x for x in topics if x != t])   # 다른 주제 랜덤 선택
            neg = weighted_choice(by_topic[neg_topic])["text"]          # 해당 주제에서 에세이 선택

            triplets.append({
                "anchor":   anchor,
                "positive": pos,
                "negative": neg,
                "topic_id": t,
            })

    random.shuffle(triplets)  # 주제 순서대로 뭉쳐 있으면 배치 편향 발생 → 셔플
    return triplets


def split_by_topic_id_order(rows, split=(0.8, 0.1, 0.1)):
    """
    주제별로 id 오름차순 정렬 후 앞에서부터 비율대로 분할.

    왜 주제별로 따로 나누냐:
      - 전체를 한꺼번에 나누면 특정 주제가 test에만 몰릴 수 있음
      - 각 주제마다 독립적으로 80/10/10 분할해야 모든 주제가 train/valid/test에 균등하게 포함됨

    결과:
      train: 241개 / valid: 29개 / test: 34개 (총 304개)
    """
    by_topic = defaultdict(list)
    for r in rows:
        by_topic[r["topic_id"]].append(r)

    train, valid, test = [], [], []

    for t, lst in by_topic.items():
        lst = sorted(lst, key=lambda x: x["id"])  # id 오름차순 정렬 (재현성 보장)
        n = len(lst)
        n_train = int(n * split[0])
        n_valid = int(n * split[1])

        train.extend(lst[:n_train])
        valid.extend(lst[n_train:n_train + n_valid])
        test.extend(lst[n_train + n_valid:])

    return train, valid, test


def build_all_with_topic():
    """
    4개 주제 파일을 하나로 합치면서 전처리.

    처리 내용:
      1. TSV에서 id, label, text(document 컬럼) 파싱
      2. clean_text()로 노이즈 제거
      3. MIN_CHARS(30자) 미만 텍스트 제외
      4. 중복 id 제거 (seen_ids로 체크)
      5. topic_id 컬럼 추가 (어느 파일에서 왔는지 표시)

    반환: id 기준으로 정렬된 전체 row 리스트
    """
    rows_out = []
    seen_ids = set()

    for topic_id, path in INPUTS:
        for row in read_tsv(path):
            try:
                doc_id = int(row["id"])
                label  = int(row["label"])
                text   = clean_text(row["document"])
            except Exception:
                continue  # 파싱 실패한 행은 스킵

            if len(text) < MIN_CHARS:
                continue  # 너무 짧은 텍스트 제외
            if doc_id in seen_ids:
                continue  # 중복 id 제외

            seen_ids.add(doc_id)
            rows_out.append({
                "id":       doc_id,
                "topic_id": topic_id,
                "text":     text,
                "label":    label,
            })

    rows_out.sort(key=lambda x: x["id"])
    return rows_out


def main():
    """
    전체 전처리 파이프라인 실행.

    Step A: 4개 파일 합치기 → all_with_topic.tsv (304개)
    Step B: 주제별 80/10/10 분할 → train.tsv(241) / valid.tsv(29) / test.tsv(34)
    Step C: train에서만 triplet 생성
            → triplets_train.tsv (학습용, 최대 800/topic)

    valid/test 평가는 eval_retrieval.py가 test.tsv + anchors.json으로 직접 수행.
    """
    random.seed(RANDOM_SEED)

    # A) 전체 데이터 합치기
    rows = build_all_with_topic()
    write_rows(str(OUTPUT_DIR / "all_with_topic.tsv"), rows)

    # B) train / valid / test 분할
    train, valid, test = split_by_topic_id_order(rows, SPLIT)
    write_rows(str(OUTPUT_DIR / "train.tsv"), train)
    write_rows(str(OUTPUT_DIR / "valid.tsv"), valid)
    write_rows(str(OUTPUT_DIR / "test.tsv"),  test)

    # C) MNR Loss 학습용 triplet 생성 (train만 생성)
    # valid/test triplet은 생성하지 않음.
    # 평가는 eval_retrieval.py가 test.tsv + anchors.json을 직접 사용하므로 불필요.
    # 데이터 규모(241개)와 epoch(3회)가 작고 KoSimCSE 파인튜닝 수준이라
    # 과적합 위험이 낮아 학습 중 valid 검증(early stopping)도 필요하지 않음.
    trip_train = make_triplets(train, seed=42, max_per_topic=800)

    write_triplets(str(OUTPUT_DIR / "triplets_train.tsv"), trip_train)

    print("DONE ✅")
    print("rows:", len(rows), "train/valid/test:", len(train), len(valid), len(test))
    print("triplets_train:", len(trip_train))


if __name__ == "__main__":
    main()
