# ============================================================
# train_label_classifier.py
# 역할: 에세이 문장 품질을 낮음/보통/높음 3단계로 분류하는 모델 파인튜닝 스크립트.
#
# 사용 모델: klue/roberta-base
#   - 한국어 자연어 이해(NLU)에 특화된 사전학습 모델 (KLUE 벤치마크 기반)
#   - 문장의 자연스러움, 구성 완성도, 표현 품질 등을 학습
#
# 원본 데이터 레이블 (0~3) → 학습용 레이블 (0~2) 리매핑:
#   원본 0, 1 → 0 (낮음)   ← 0과 1을 합친 이유: 샘플 수가 너무 적어 불균형 심함
#   원본 2    → 1 (보통)
#   원본 3    → 2 (높음)
#
# 클래스 불균형 보정 (WeightedTrainer):
#   에세이 데이터 특성상 높은 점수(3점)가 많고 낮은 점수(0,1점)가 적음.
#   그냥 학습하면 모델이 항상 "높음"만 예측하는 문제 발생.
#   → 적은 클래스에 더 높은 loss 가중치를 부여해서 보정.
#   가중치 공식: total / (클래스수 × 해당클래스샘플수)
#              샘플이 적은 클래스일수록 가중치가 커짐
#
# 학습 결과: label_model/ 폴더에 저장
#   - service_scorer.py에서 이 모델을 로드해서 문장 품질 점수 계산에 사용
#   - 출력: [p_낮음, p_보통, p_높음] 확률 → 확률 가중합으로 0~2 연속값 변환
# ============================================================

import csv
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent
OUT_DIR   = BASE_DIR / "out"

TRAIN_PATH = OUT_DIR / "train.tsv"
VALID_PATH = OUT_DIR / "valid.tsv"
TEST_PATH  = OUT_DIR / "test.tsv"

# ── 모델 설정 ─────────────────────────────────────────────────────────────────
MODEL_NAME  = "klue/roberta-base"  # 한국어 분류 태스크에 강한 사전학습 모델
NUM_LABELS  = 3    # 낮음(0) / 보통(1) / 높음(2) 3분류
MAX_LEN     = 256  # 토크나이저 최대 입력 길이 (에세이 길이 고려)


def remap_label(orig: int) -> int:
    """
    원본 0~3 레이블을 0~2 (3단계)로 리매핑.

    원본 데이터의 0점과 1점 샘플이 매우 적어서 4분류로 학습하면
    해당 클래스를 거의 학습하지 못하는 불균형 문제 발생.
    → 0과 1을 "낮음(0)"으로 합쳐서 3분류로 단순화.
    """
    if orig <= 1:
        return 0   # 낮음 (원본 0점, 1점)
    elif orig == 2:
        return 1   # 보통 (원본 2점)
    else:
        return 2   # 높음 (원본 3점)


def load_tsv(path: Path):
    """
    TSV에서 텍스트와 레이블을 읽어 HuggingFace Dataset으로 변환.
    레이블은 remap_label()로 0~2로 변환해서 저장.
    """
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            texts.append(r["text"])
            labels.append(remap_label(int(r["label"])))
    return Dataset.from_dict({"text": texts, "label": labels})


def compute_metrics(eval_pred):
    """
    검증/테스트 시 평가 지표 계산.
    - acc      : 전체 정확도
    - macro_f1 : 클래스별 F1 평균 (불균형 데이터에서 accuracy보다 신뢰성 높음)
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "acc":      accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


class WeightedTrainer(Trainer):
    """
    클래스 불균형 보정을 위해 loss에 가중치를 적용하는 커스텀 Trainer.

    HuggingFace 기본 Trainer는 모든 샘플에 동일한 loss를 적용함.
    → 높음(2) 클래스가 많으면 낮음(0) 클래스를 무시하는 방향으로 학습됨.

    WeightedTrainer는 적은 클래스에 더 높은 loss 가중치를 부여해서
    모든 클래스를 균등하게 학습하도록 보정.
    """
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights  # 클래스별 가중치 텐서

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits

        # 가중치 적용 cross entropy loss
        # 샘플 적은 클래스의 오분류에 더 큰 패널티 부여
        loss = torch.nn.functional.cross_entropy(
            logits, labels, weight=self.class_weights.to(logits.device)
        )
        return (loss, outputs) if return_outputs else loss


def main():
    # ── 데이터 로드 ───────────────────────────────────────────────────────────
    train_ds = load_tsv(TRAIN_PATH)
    valid_ds = load_tsv(VALID_PATH)
    test_ds  = load_tsv(TEST_PATH)

    # ── 클래스 가중치 계산 ────────────────────────────────────────────────────
    # 공식: weight_i = total / (NUM_LABELS × count_i)
    # 샘플이 적은 클래스일수록 가중치가 커져서 loss에 더 큰 영향을 미침
    label_counts  = Counter(train_ds["label"])
    total         = sum(label_counts.values())
    weights       = [total / (NUM_LABELS * label_counts.get(i, 1)) for i in range(NUM_LABELS)]
    class_weights = torch.tensor(weights, dtype=torch.float)
    print("class weights:", {i: round(w, 3) for i, w in enumerate(weights)})
    # 출력 예) {0: 3.2, 1: 1.8, 2: 0.6} → 낮음 클래스에 가장 높은 가중치

    # ── 토크나이저 및 전처리 ──────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tok(batch):
        # 최대 MAX_LEN(256) 토큰으로 truncation
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN)

    train_ds = train_ds.map(tok, batched=True)
    valid_ds = valid_ds.map(tok, batched=True)
    test_ds  = test_ds.map(tok, batched=True)

    # 배치 내 길이가 다른 샘플들을 같은 길이로 패딩
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ── 모델 로드 ─────────────────────────────────────────────────────────────
    # klue/roberta-base의 사전학습 가중치를 불러와서 분류 헤드(NUM_LABELS=3) 추가
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )

    # ── 학습 설정 ─────────────────────────────────────────────────────────────
    args = TrainingArguments(
        output_dir="label_cls_ckpt",          # 체크포인트 저장 경로
        eval_strategy="epoch",                # 매 epoch 끝마다 valid 평가
        save_strategy="epoch",                # 매 epoch 끝마다 체크포인트 저장
        learning_rate=2e-5,                   # 파인튜닝 권장 learning rate
        per_device_train_batch_size=8,        # RTX 3050 기준 안전한 배치 크기
        per_device_eval_batch_size=16,
        num_train_epochs=5,                   # 소규모 데이터이므로 5 epoch
        weight_decay=0.01,                    # L2 정규화 (과적합 방지)
        fp16=True,                            # GPU 사용 시 16bit 연산으로 속도/메모리 절약
        logging_steps=20,
        load_best_model_at_end=True,          # 학습 완료 후 best 체크포인트로 자동 복원
        metric_for_best_model="macro_f1",     # best 기준: macro F1 (불균형 데이터에 적합)
        greater_is_better=True,
    )

    # ── 학습 실행 ─────────────────────────────────────────────────────────────
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,      # epoch마다 valid로 macro_f1 측정 → best model 선택
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # ── 테스트 평가 ───────────────────────────────────────────────────────────
    # 학습이 끝난 best model로 test 성능 최종 확인
    print("\n=== TEST EVAL ===")
    pred   = trainer.predict(test_ds)
    logits = pred.predictions
    y_true = pred.label_ids
    y_pred = np.argmax(logits, axis=1)

    print("acc:      ", accuracy_score(y_true, y_pred))
    print("macro_f1: ", f1_score(y_true, y_pred, average="macro"))
    print("confusion:\n", confusion_matrix(y_true, y_pred))
    # confusion matrix 해석 예시 (행=정답, 열=예측):
    #   [[낮음→낮음, 낮음→보통, 낮음→높음],
    #    [보통→낮음, 보통→보통, 보통→높음],
    #    [높음→낮음, 높음→보통, 높음→높음]]

    # ── 모델 저장 ─────────────────────────────────────────────────────────────
    # service_scorer.py에서 label_model/ 경로로 직접 로드
    trainer.save_model("label_model")
    tokenizer.save_pretrained("label_model")
    print("\nSaved -> label_model")


if __name__ == "__main__":
    main()
