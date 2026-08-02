# 02. Phương pháp

## Thiết kế chung

Repository cung cấp 3 pipeline dùng chung một codebase. Giá trị `solution` trong file YAML quyết định pipeline được khởi tạo.

```text
note.txt
    → NER
    → xử lý span và type
    → assertion
    → ontology linking
    → validation
    → concepts.json
```

Các component chính nằm trong:

```text
src/medextract/
├── ner/            # GLiNER + post-process (clean_spans, generic prefix)
├── assertions/     # ConText rule engine (baseline / improved)
├── normalization/  # SapBERT retriever, LLM reranker, exact-alias normalizer + lookup
├── llm/            # Qwen engine và next-token logits helper
├── kb/             # build_icd, build_rxnorm, FAISS/SapBERT index
├── scoring/        # host-metric scorer
├── models/         # registry + parameter budget guard
├── config.py       # YAML load/merge + known-key schema
├── io_utils.py     # đọc .txt, ghi JSON, zip submission
├── schema.py       # concept types, assertions, candidates, validate_output
├── selector.py     # two-teacher consensus selector (improved_v2)
└── pipeline.py     # build_pipeline dispatch trên `solution`
```

Mỗi stage có interface riêng. Cấu trúc này giúp thay đổi NER, assertion model hoặc normalizer mà không cần sửa toàn bộ pipeline.

## 1. Baseline

Config:

```text
configs/baseline.yaml
```

Luồng xử lý:

```text
GLiNER
    → span post-processing
    → ConText rules
    → SapBERT retrieval
    → FAISS nearest neighbors
    → schema validation
```

### GLiNER zero-shot NER

Baseline dùng:

```text
urchade/gliner_multi-v2.1
```

Model nhận năm label mô tả bằng tiếng Anh và ánh xạ về type của cuộc thi:

| Label cho GLiNER | Type đầu ra |
|---|---|
| `symptom` | `TRIỆU_CHỨNG` |
| `disease or diagnosis` | `CHẨN_ĐOÁN` |
| `medication or drug` | `THUỐC` |
| `medical test or lab name` | `TÊN_XÉT_NGHIỆM` |
| `test result or measurement value` | `KẾT_QUẢ_XÉT_NGHIỆM` |

Bệnh án dài được chia theo ranh giới dòng. Mỗi chunk là một lát cắt nguyên vẹn của văn bản gốc, sau đó offset được cộng lại theo vị trí toàn cục.

Cách này bảo đảm:

```python
document[start:end] == mention
```

Các span chồng lấn được chọn theo score, sau đó ưu tiên ranh giới gọn hơn.

### Span post-processing

Baseline áp dụng một số rule để sửa lỗi thường gặp:

- Loại khoảng trắng và dấu câu ở hai đầu
- Loại từ phủ định khỏi span khi phù hợp
- Mở rộng drug span qua strength hoặc dosage token
- Loại các section header rõ ràng
- Sửa span bắt đầu hoặc kết thúc giữa một từ

Các rule chỉ sửa offset trên văn bản gốc, không chuẩn hóa trực tiếp nội dung bệnh án.

### ConText assertions

Assertion model là một rule engine tiếng Việt độc lập, không phụ thuộc spaCy.

Model tìm cue (dấu hiệu) trong dòng hoặc section gần concept:

- Negation cue như `không`, `chưa`, `âm tính`
- Family cue như `mẹ`, `bố`, `gia đình`
- Historical cue như `tiền sử`, `trước đây`, `đã từng`

Scope được giới hạn theo dòng, clause boundary và section boundary để giảm over-prediction.

### SapBERT và FAISS linking

Baseline dùng multilingual SapBERT để encode:

- Mention trong bệnh án
- Tên concept trong ICD-10
- Tên concept trong RxNorm

Embedding được chuẩn hóa L2 và truy vấn bằng FAISS `IndexFlatIP`.

Mapping theo type:

```text
CHẨN_ĐOÁN → ICD-10
THUỐC     → RxNorm
```

Đối với thuốc, route và frequency được loại khỏi bản sao dùng cho retrieval, nhưng strength được giữ lại. Candidate được giới hạn ở số lượng nhỏ để tránh làm giảm Jaccard.

## 2. Improved

Config:

```text
configs/improved.yaml
```

Pipeline này giữ nguyên NER, assertion và retrieval của baseline, sau đó thêm Qwen3-8B để rerank candidate.

```text
GLiNER
    → ConText
    → SapBERT top-k
    → Qwen3-8B listwise reranker
    → candidate codes
```

### Retrieve-then-rerank

SapBERT tạo danh sách candidate ban đầu. LLM chỉ được phép chọn mã trong danh sách này.

Prompt cung cấp:

- Mention
- Dòng chứa mention
- Ontology đang sử dụng
- Candidate code
- Candidate name
- RxNorm term type nếu có

LLM trả về JSON array gồm các mã đã có trong candidate list. Mọi mã ngoài danh sách đều bị loại.

Thiết kế này có hai ưu điểm:

1. LLM không thể tạo một mã ontology mới
2. Reasoning chỉ tập trung vào phân biệt các candidate gần nhau

Config mặc định dùng 4-bit để Qwen3-8B có thể chạy trên GPU 16 GB.

## 3. Improved v2

Config:

```text
configs/improved_v2.yaml
```

Đây là pipeline có precision cao nhất trong repository.

```text
GLiNER raw spans (raw_floor = 0.02)
    → per-type thresholds
    → clean_spans
    → primary type corrector (TRIỆU → CHẨN)
    → two-teacher span additions (chỉ type không có candidate)
    → boundary.generic_prefix trim
    → exact-alias lookup → emit mã chỉ khi unique exact alias
    → validation
```

### Per-type thresholds

Một threshold chung không phù hợp với cả năm type vì score distribution của GLiNER khác nhau theo label.

`improved_v2` dùng threshold riêng:

| Type | Threshold |
|---|---:|
| `TRIỆU_CHỨNG` | 0.20 |
| `CHẨN_ĐOÁN` | 0.25 |
| `THUỐC` | 0.30 |
| `TÊN_XÉT_NGHIỆM` | 0.15 |
| `KẾT_QUẢ_XÉT_NGHIỆM` | 0.35 |

GLiNER vẫn thu thập span ở `raw_floor = 0.02`. Các span dưới threshold chính không được emit ngay, nhưng có thể được selector xem xét để bổ sung.

### Span cleanup (clean_spans)

`improved_v2` dùng đúng một cleaner duy nhất là `clean_spans` trong `src/medextract/ner/postprocess.py` — chính là cleaner đã dùng ở baseline và improved, không có cleaner rẽ nhánh khác.

`clean_spans` là destructive theo thiết kế:

- Loại khoảng trắng và dấu câu ở hai đầu
- Sửa span bắt đầu hoặc kết thúc giữa một từ
- Loại span rỗng, offset ngoài phạm vi, span chỉ có khoảng trắng hoặc dấu câu
- Loại duplicate có cùng `start`, `end` và `type`

Sau `clean_spans`, hai bước boundary hẹp còn lại chạy riêng:

- `boundary.generic_prefix` (đang bật) cắt tiền tố chung như `dấu hiệu / biểu hiện / tình trạng / hội chứng` khi phía sau vẫn còn một concept trọn vẹn.
- Header rõ ràng bị loại muộn hơn trong `_build_concepts` qua `is_header_span`, sau khi selector đã chạy, để không xóa nhầm span có type đã được sửa.

Các lever boundary khác (`symptom_boundary_trim`, `drug_cleanup`) từng được thử nhưng không tạo lợi ích ổn định và đã bị loại (xem [`04_findings.md`](04_findings.md)).

### Primary type corrector

Teacher chính:

```text
Qwen/Qwen3-4B-Instruct-2507
```

Model sửa lỗi type phổ biến nhất:

```text
TRIỆU_CHỨNG → CHẨN_ĐOÁN
```

Teacher không sinh câu trả lời tự do. Pipeline đọc next-token logits trên các label cố định và chọn type có score phù hợp.

### Two-teacher additions

Teacher phụ:

```text
Qwen/Qwen3.5-4B
```

Hai teacher cùng xem các span raw có score thấp nhưng chưa giao với span đã chọn.

Một span chỉ được thêm khi:

1. Cả hai teacher chọn cùng type
2. Type thuộc nhóm không có candidate
3. Margin so với nhãn `NONE` đủ lớn

Các type được phép bổ sung:

```text
TRIỆU_CHỨNG
TÊN_XÉT_NGHIỆM
KẾT_QUẢ_XÉT_NGHIỆM
```

`CHẨN_ĐOÁN` và `THUỐC` không được thêm theo cơ chế này vì một concept thừa ở hai type đó còn làm giảm candidate score.

### Exact-alias linking

`improved_v2` không tải SapBERT và không dùng FAISS. Candidate linking là một exact-alias lookup precision-first.

Luồng:

```text
mention → normalize (chỉ bản copy dùng cho retrieval, văn bản gốc không đổi)
       → exact-alias lookup trong hai parquet alias (ICD-10 TT06, RxNorm v2)
       → emit mã CHỈ KHI có đúng một alias trùng khớp duy nhất
       → ICD bare-category (.9 leaf remap khi leaf đó tồn tại)
       → tối đa 1 candidate / concept
```

Hai module chịu trách nhiệm:

- `src/medextract/normalization/lexical_lookup.py` (`ExactAliasLookup`): tra alias exact match trên hai bảng term, có normalize bản mention dùng cho retrieval (ví dụ bỏ route/frequency với thuốc). Tra theo cache `(kb, mention)` vì bệnh án lặp mention.
- `src/medextract/normalization/exact_alias_normalizer.py` (`ExactAliasNormalizer`): quyết định emit — chỉ trả mã khi `len(exact_codes) == 1`, kèm `.9` remap cho bare ICD category.

Retrieval rộng không đồng nghĩa với emit rộng. Các kênh fuzzy rộng hơn từng thử trong quá trình phát triển đều không thay đổi candidate nào sau khi emission đã yêu cầu unique exact alias, nên đã bị loại hoàn toàn (chi tiết ablation ở [`04_findings.md`](04_findings.md)).

### Empty assertions

`improved_v2` trả assertions rỗng.

Đây là lựa chọn theo dữ liệu của cuộc thi. Trong split nội bộ, assertion labels tương đối thưa và các rule thử nghiệm có xu hướng emit thừa.

Quyết định này không phải là khuyến nghị chung cho clinical NLP.

### Validation

Stage cuối kiểm tra:

- `text` khớp với `position`
- Type hợp lệ
- Assertion chỉ xuất hiện trên type cho phép
- Candidate chỉ xuất hiện trên type cho phép
- Output có thứ tự ổn định

## Precision-first design

Metric phạt mạnh concept và candidate thừa. Vì vậy, `improved_v2` ưu tiên:

- Threshold theo từng type
- Type correction có mục tiêu rõ
- Chỉ bổ sung type không có candidate
- Chỉ emit candidate khi match đủ chắc chắn
- Không dùng generation khi next-token logits đã đủ

Thiết kế này khác với pipeline recall-first, nơi hệ thống cố gắng giữ mọi span hoặc mọi candidate có khả năng đúng.

## Models size

| Model | Vai trò | Tham số |
|---|---|---:|
| `urchade/gliner_multi-v2.1` | NER | 0.289B |
| `Qwen/Qwen3-4B-Instruct-2507` | Primary selector | 4.022B |
| `Qwen/Qwen3.5-4B` | Secondary selector | 4.206B |
| **Tổng** | | **8.517B** |

Model registry kiểm tra tổng số tham số đang được load. Quantization làm giảm VRAM cần dùng nhưng không làm giảm số tham số được tính theo giới hạn 9B.

Xem kết quả tại [`03_results.md`](03_results.md).
