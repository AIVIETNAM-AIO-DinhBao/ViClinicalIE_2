# Implementation Plan

## 0. Mục tiêu

Kế hoạch này dùng để triển khai nhanh hệ thống theo hướng hackathon:

```text
1. Chạy được baseline hiện tại.
2. Tách và kiểm tra NER zero-shot.
3. Tạo output hợp lệ end-to-end.
4. Sinh data và fine-tune GLiNER.
5. Tích hợp fine-tuned model.
6. Tối ưu nhanh các phần có ROI cao.
```

Không làm acceptance gate/ablation phức tạp. Chỉ cần smoke test, validate offset/schema và so sánh score nhanh nếu có gold.

---

## Phase 0 — Smoke test repo

### Mục tiêu

Biết repo hiện tại chạy được tới đâu, thiếu gì.

### Việc cần làm

```bash
cd "D:/Viettel AI Race/training-repo"
pip install -r requirements.txt
pip install -e .
```

Chạy thử sample:

```bash
python run.py --config configs/baseline.yaml --input data/sample_input --output out/sample_baseline
```

Nếu có dev gold:

```bash
python run.py --config configs/baseline.yaml --input data/dev/input --output out/dev_baseline
python score.py --pred out/dev_baseline --gold data/dev
```

### Done khi

```text
- import package không lỗi
- biết GLiNER load được hay chưa
- biết KB/linking có đang chặn pipeline không
- có output JSON hoặc log lỗi rõ ràng
```

---

## Phase 1 — NER zero-shot only

### Mục tiêu

Chạy riêng GLiNER zero-shot, không phụ thuộc linking/KB.

### File dự kiến

```text
scripts/run_ner_zero_shot.py
configs/zero_shot_ner.yaml
src/medextract/ner/gliner_ner.py
```

### Output debug mong muốn

```json
[
  {
    "text": "khó thở",
    "type": "TRIỆU_CHỨNG",
    "position": [16, 23],
    "score": 0.72
  }
]
```

Nếu class NER hiện tại không trả score thì có thể bỏ `score` ở bản đầu.

### Command mong muốn

```bash
python scripts/run_ner_zero_shot.py --config configs/zero_shot_ner.yaml --input data/sample_input --output out/ner_zero_shot
```

### Done khi

```text
- GLiNER chạy được trên sample
- mỗi file có JSON prediction
- mọi span thỏa raw[start:end] == text
- log được số entity theo type
```

---

## Phase 2 — Baseline end-to-end hợp lệ

### Mục tiêu

Có pipeline tạo được submission JSON hợp lệ, dù score chưa cao.

### Pipeline

```text
GLiNER zero-shot
-> clean_spans
-> ConText assertions
-> normalizer nếu KB có sẵn
-> validate_output
-> write JSON
```

### Việc cần làm

1. Chạy `run.py` với `configs/baseline.yaml`.
2. Nếu KB chưa có, cho phép chế độ tạm thời trả `candidates: []` cho diagnosis/drug để pipeline không crash.
3. Kiểm tra output schema.
4. Kiểm tra offset.
5. Tạo zip submission nếu cần.

### Command

```bash
python run.py --config configs/baseline.yaml --input data/sample_input --output out/sample_e2e --zip
```

### Done khi

```text
- output là JSON list
- đúng type/assertion hợp lệ
- candidates chỉ có ở CHẨN_ĐOÁN và THUỐC
- không có offset mismatch
- tạo được zip
```

---

## Phase 3 — Tune nhanh zero-shot

### Mục tiêu

Cải thiện NER zero-shot bằng threshold và cleanup đơn giản.

### Việc cần làm

Thử các threshold:

```text
0.25, 0.30, 0.35, 0.40, 0.45
```

Nếu nhanh, thử threshold theo type:

```yaml
TRIỆU_CHỨNG: 0.40
CHẨN_ĐOÁN: 0.35
THUỐC: 0.30
TÊN_XÉT_NGHIỆM: 0.35
KẾT_QUẢ_XÉT_NGHIỆM: 0.35
```

Log tối thiểu:

```text
file_id, total_entities, num_by_type
```

### File dự kiến

```text
configs/zero_shot_fast.yaml
src/medextract/ner/postprocess.py
src/medextract/ner/gliner_ner.py
```

### Done khi

```text
- chọn được config zero-shot ổn nhất
- không over-detect quá rõ
- không làm giảm chất lượng bằng cleanup nguy hiểm
```

---

## Phase 4 — Synthetic data cho GLiNER

### Mục tiêu

Tạo dataset nhỏ nhưng đúng offset để fine-tune GLiNER.

### File dự kiến

```text
scripts/generate_gliner_synthetic.py
data/gliner_train/train.jsonl
data/gliner_train/dev.jsonl
scripts/validate_gliner_data.py
```

### Nguồn sample

Template diagnosis:

```text
Bệnh nhân có tiền sử {diagnosis}.
Không ghi nhận {diagnosis}.
Được chẩn đoán {diagnosis}.
```

Template symptom:

```text
Bệnh nhân {symptom}.
Không {symptom}.
Có {symptom} khi gắng sức.
```

Template drug:

```text
Đang dùng {drug} {strength} po daily.
Trước nhập viện dùng {drug} {strength} bid.
```

Template lab/result:

```text
{test}: {result}.
{test} là {result}.
{test} bình thường.
```

### Quy mô v1

```text
1k-3k samples tổng
train/dev = 90/10
```

### Done khi

```text
- file JSONL đọc được
- 100% entity offset đúng
- đủ 5 type
- có một ít noise: bỏ dấu, dính chữ, thiếu dấu câu
```

---

## Phase 5 — Fine-tune GLiNER v1

### Mục tiêu

Có model GLiNER fine-tuned đầu tiên.

### File dự kiến

```text
scripts/train_gliner.py
models/gliner_finetuned_v1/
```

### Config gợi ý

```text
base model: urchade/gliner_multi-v2.1
epochs: 3-5
batch size: theo GPU
learning rate: dùng default trước
```

Không grid search nhiều ở bản đầu.

### Done khi

```text
- train chạy xong
- lưu được checkpoint
- checkpoint inference được trên sample
- không over-detect nghiêm trọng so với zero-shot
```

---

## Phase 6 — Tích hợp fine-tuned GLiNER

### Mục tiêu

Pipeline chính dùng được model fine-tuned.

### File dự kiến

```text
configs/finetuned.yaml
src/medextract/ner/gliner_ner.py
```

### Config mẫu

```yaml
extends: baseline.yaml
solution: finetuned

ner:
  model: models/gliner_finetuned_v1
  threshold: 0.35
```

### Command

```bash
python run.py --config configs/finetuned.yaml --input data/sample_input --output out/sample_finetuned
```

Nếu có dev:

```bash
python run.py --config configs/finetuned.yaml --input data/dev/input --output out/dev_finetuned
python score.py --pred out/dev_finetuned --gold data/dev
```

### Done khi

```text
- pipeline chạy với model fine-tuned
- output hợp lệ
- nếu score/manual review tệ hơn zero-shot thì giữ zero-shot làm fallback
```

---

## Phase 7 — Quick ROI improvements

### 7.1 Drug helper

Mục tiêu: span thuốc đầy đủ hơn.

Pattern ưu tiên:

```text
drug + strength + unit + route + frequency
```

Ví dụ:

```text
amlodipine 10 mg po daily
metformin 500 mg bid
aspirin 81 mg
```

### 7.2 Lab/result helper

Mục tiêu: bắt test/result structured.

Pattern:

```text
<test>: <value>
<test> là <value>
<test> tăng/giảm/bình thường/âm tính/dương tính
```

Không emit bare number nếu không có test context.

### 7.3 Linking nhanh

Ưu tiên:

```text
exact alias
no-diacritic alias
lowercase match
RxNorm ingredient + strength
ICD synonym enrichment
```

LLM reranker để sau.

### Done khi

```text
- cải thiện lỗi thấy rõ khi review output
- không làm tăng duplicate/FP quá nhiều
- output vẫn validate pass
```

---

## Phase 8 — Submission hardening

### Mục tiêu

Tạo submission ổn định.

### Checklist tối thiểu

```text
- chạy full input không crash
- đủ file output
- JSON parse được
- offset đúng
- type/assertion hợp lệ
- candidates đúng chỗ
- zip đúng structure
```

### Command

```bash
python run.py --config configs/finetuned.yaml --input <test_input> --output out/submission --zip
```

---

## Thứ tự ưu tiên nếu thiếu thời gian

```text
P0. baseline chạy được
P1. NER zero-shot only
P2. output hợp lệ + offset đúng
P3. assertion rules
P4. linking tối thiểu
P5. threshold tuning
P6. synthetic data
P7. fine-tune GLiNER
P8. tích hợp fine-tuned model
P9. drug/lab helper rules
P10. linking nâng cao
```

Không làm ngay:

```text
learned fusion
assertion classifier
cross-encoder reranker
LLM reranker bắt buộc
external dataset pipeline lớn
ablation matrix phức tạp
```
