# Solution Design Hackathon

## 1. Mục tiêu

Xây hệ thống trích xuất và chuẩn hóa khái niệm y khoa cho Viettel AI Race Track 2 theo hướng **nhanh, gọn, chạy được, dễ cải thiện**.

Định hướng chính:

```text
Training Session làm core.
V2 chỉ bổ sung các phần có lợi rõ trong thời gian ngắn.
Zero-shot NER chạy được trước, fine-tune GLiNER sau.
```

Không cố xây kiến trúc production quá nặng trong giai đoạn hackathon.

---

## 2. Bài toán

Input là clinical note dạng `.txt`. Output là `output/{id}.json`, mỗi file chứa danh sách concept.

5 loại entity:

```text
TRIỆU_CHỨNG
TÊN_XÉT_NGHIỆM
KẾT_QUẢ_XÉT_NGHIỆM
CHẨN_ĐOÁN
THUỐC
```

Assertions áp dụng cho:

```text
TRIỆU_CHỨNG, CHẨN_ĐOÁN, THUỐC
```

Các assertion hợp lệ:

```text
isNegated
isHistorical
isFamily
```

Candidates áp dụng cho:

```text
CHẨN_ĐOÁN -> ICD-10
THUỐC     -> RxNorm RXCUI
```

Bất biến quan trọng nhất:

```python
raw_text[start:end] == entity["text"]
```

---

## 3. Nguyên tắc thiết kế

1. **Chạy được trước, tối ưu sau.**
2. **GLiNER là backbone NER.** Ban đầu dùng zero-shot, sau đó fine-tune.
3. **Không normalize text để lấy offset.** Normalize chỉ dùng cho search/linking.
4. **Rules dùng ở nơi rõ cấu trúc.** Ví dụ thuốc, xét nghiệm, negation cue.
5. **Không union prediction bừa bãi.** Nếu thêm rule thì phải tránh duplicate/overlap.
6. **Không thêm module học máy nếu chưa có pipeline baseline ổn.**
7. **Ưu tiên điểm số thực tế hơn kiến trúc đẹp.**

---

## 4. Core pipeline

Pipeline giai đoạn đầu:

```text
Raw note
  -> GLiNER zero-shot NER
  -> boundary cleanup nhẹ
  -> ConText-style assertion rules
  -> ICD/RxNorm linking
  -> JSON + offset validation
```

Pipeline sau fine-tune:

```text
Raw note
  -> GLiNER fine-tuned
  -> drug/lab helper rules
  -> boundary cleanup + overlap cleanup
  -> ConText-style assertion rules
  -> hybrid linking
  -> JSON + offset validation
```

---

## 5. Module chính

### 5.1 NER bằng GLiNER

Dùng GLiNER để tìm span và type cho cả 5 loại entity.

Config gốc:

```yaml
model: urchade/gliner_multi-v2.1
threshold: 0.35
labels:
  symptom                          -> TRIỆU_CHỨNG
  disease or diagnosis             -> CHẨN_ĐOÁN
  medication or drug               -> THUỐC
  medical test or lab name         -> TÊN_XÉT_NGHIỆM
  test result or measurement value -> KẾT_QUẢ_XÉT_NGHIỆM
```

Việc cần ưu tiên:

- chạy được zero-shot trên sample;
- kiểm tra offset;
- thử threshold nhanh;
- sau đó mới fine-tune.

### 5.2 Boundary cleanup nhẹ

Chỉ làm các xử lý an toàn:

```text
trim whitespace
trim dấu câu đầu/cuối
không lấy cue phủ định vào span
mở rộng span thuốc sang strength/unit/route/frequency nếu rõ
resolve overlap đơn giản
```

Không làm boundary refiner phức tạp ở giai đoạn đầu.

### 5.3 Assertion rules

Dùng ConText-style rules hiện tại.

Ví dụ cue:

```text
không, không có, chưa ghi nhận      -> isNegated
tiền sử, trước nhập viện, đã từng   -> isHistorical
mẹ, bố, cha, anh, chị, em           -> isFamily
```

Không train assertion classifier trước khi NER và linking ổn.

### 5.4 Linking

Giai đoạn đầu dùng baseline:

```text
SapBERT + FAISS retrieval
```

Sau đó bổ sung nhanh:

```text
exact alias
lowercase/no-diacritic matching
char fuzzy/BM25 nếu có sẵn
RxNorm strength constraint
ICD alias enrichment
```

LLM reranker chỉ làm nếu còn thời gian và candidate đúng thường đã nằm trong top-k.

### 5.5 Drug/lab helper rules

Chỉ thêm rule có ROI cao:

Drug:

```text
amlodipine 10 mg po daily
metformin 500 mg bid
aspirin 81 mg
```

Lab/result:

```text
Troponin: 0.10
Glucose máu tăng
Điện tâm đồ bình thường
X-quang ngực âm tính
```

Mục tiêu là hỗ trợ GLiNER, không thay GLiNER hoàn toàn.

---

## 6. Những phần lấy từ V2

Giữ các phần quan trọng:

```text
offset-safe processing
structure-aware chunking
simple boundary cleanup
drug/lab helper rules
overlap cleanup
hybrid linking cơ bản
strict JSON/offset validator
```

Hoãn các phần nặng:

```text
learned fusion
assertion transformer
cross-encoder reranker
LLM reranker bắt buộc
full ablation matrix
full acceptance gates
external dataset pipeline lớn
```

---

## 7. Chiến lược fine-tune GLiNER

Fine-tune chỉ bắt đầu sau khi zero-shot pipeline chạy được.

Nguồn data ưu tiên:

```text
synthetic label-by-construction
```

Tạo câu/note ngắn từ ontology và template:

```text
Bệnh nhân có tiền sử tăng huyết áp.
Không ghi nhận khó thở.
Đang dùng amlodipine 10 mg po daily.
Troponin là 0.10.
```

Mỗi sample phải có offset đúng ngay từ lúc sinh.

Mục tiêu v1:

```text
1k-3k samples
đủ 5 entity types
có negation/history/family context
có thuốc + xét nghiệm + kết quả
có noise nhẹ: bỏ dấu, dính chữ, mixed EN/VI
```

---

## 8. Rủi ro chính và cách xử lý nhanh

| Rủi ro | Cách xử lý nhanh |
|---|---|
| GLiNER over-detect | tăng threshold, rule filter noise, giới hạn density |
| Sai offset | validate bắt buộc, drop span sai |
| Sai symptom/diagnosis | thử label/threshold, thêm head-word hints nếu cần |
| Thuốc bị thiếu liều | drug span extension rule |
| Lab/result bị lẫn | pattern pairing đơn giản |
| Linking không ra code | thêm alias/no-diacritic/exact match |
| Fine-tune làm tệ hơn | giữ zero-shot config làm fallback |

---

## 9. Kết luận

Thiết kế cuối cùng cho hackathon:

```text
Training Session core
+ một số guardrail quan trọng từ V2
+ fine-tune GLiNER khi baseline đã chạy
```

Ưu tiên thực tế:

```text
chạy được -> output hợp lệ -> tune NER -> fine-tune -> cải thiện linking/rules
```
