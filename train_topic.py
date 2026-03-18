# ============================================================
# train_topic.py
# 역할: 주제 적합성 판단을 위한 임베딩 모델 파인튜닝 스크립트.
#
# 사용 모델: BM-K/KoSimCSE-roberta-multitask
#   - 한국어 문장 유사도에 특화된 사전학습 모델
#   - 이미 한국어 문장 의미를 충분히 학습한 상태에서 출발
#   - 우리 도메인(한국어 말하기 평가)에 맞게 추가 학습(파인튜닝)
#
# 학습 방식: MultipleNegativesRankingLoss (MNR Loss)
#   - (anchor, positive) 쌍만 입력하면 됨
#   - 배치 내 다른 샘플들이 자동으로 negative로 활용됨
#     예) 배치 크기 8이면 1쌍당 7개의 negative가 자동 생성
#   - triplet 파일에 negative 컬럼이 있어도 여기서는 사용하지 않음
#
# 학습 결과: topic_model_mnr/ 폴더에 저장
#   - service_scorer.py에서 이 모델을 로드해서 서비스에 사용
#   - 평가: eval_retrieval.py 실행 → Recall@10 약 88% 달성
# ============================================================

import csv
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses

# ── 설정 ──────────────────────────────────────────────────────────────────────
MODEL_NAME = "BM-K/KoSimCSE-roberta-multitask"  # 한국어 문장 유사도 사전학습 모델
TRAIN_PATH = "out/triplets_train.tsv"           # make_anchor.py로 생성한 학습용 triplet

BATCH_SIZE    = 8    # RTX 3050 기준 안전한 배치 크기
EPOCHS        = 3    # 데이터가 소규모(241개)이므로 3회로 충분. 이상 늘리면 과적합 우려
WARMUP_STEPS  = 100  # 초반 learning rate를 서서히 올리는 구간 (학습 안정화)


def load_pairs(path):
    """
    triplets_train.tsv에서 (anchor, positive) 쌍만 읽어서 InputExample 리스트 반환.

    MNR Loss는 anchor-positive 쌍만 있으면 학습 가능.
    negative 컬럼은 파일에 존재하지만 여기서는 무시.
    (배치 내 다른 positive들이 자동으로 negative 역할을 함)
    """
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            examples.append(InputExample(texts=[row["anchor"], row["positive"]]))
    return examples


# ── GPU 확인 ──────────────────────────────────────────────────────────────────
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ── 모델 로드 ─────────────────────────────────────────────────────────────────
# HuggingFace에서 사전학습 모델 다운로드 (최초 실행 시 자동 다운로드)
model = SentenceTransformer(MODEL_NAME)

# ── 데이터 로드 ───────────────────────────────────────────────────────────────
train_examples = load_pairs(TRAIN_PATH)
train_loader   = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
# shuffle=True: 매 epoch마다 배치 구성을 다르게 해서 다양한 negative 조합 확보

# ── Loss 설정 ─────────────────────────────────────────────────────────────────
# MultipleNegativesRankingLoss:
#   배치 내 (anchor_i, positive_i) 쌍에서
#   anchor_i와 positive_i의 유사도는 높이고,
#   anchor_i와 다른 positive_j (j≠i)의 유사도는 낮추도록 학습
loss = losses.MultipleNegativesRankingLoss(model)

# ── 학습 ──────────────────────────────────────────────────────────────────────
model.fit(
    train_objectives=[(train_loader, loss)],
    epochs=EPOCHS,
    warmup_steps=WARMUP_STEPS,
    output_path="topic_model_mnr",  # 학습 완료 후 저장 경로
)

print("Saved -> topic_model_mnr")
