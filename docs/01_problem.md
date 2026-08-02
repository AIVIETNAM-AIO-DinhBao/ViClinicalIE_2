# 01. Bài toán

## Mục tiêu

Viettel AI Race 2026, Track 2 yêu cầu trích xuất các khái niệm y khoa từ bệnh án tiếng Việt và chuẩn hóa một số khái niệm về mã ontology.

Có thể xem bài toán gồm 4 tác vụ liên tiếp:

1. Phát hiện đoạn văn bản biểu diễn một khái niệm y khoa
2. Gán đúng type cho đoạn văn bản đó
3. Xác định assertion khi type cho phép
4. Liên kết chẩn đoán với ICD-10 và thuốc với RxNorm

Hệ thống cần trả về danh sách concept theo đúng schema của cuộc thi.

## Đầu vào

Mỗi sample là một file `.txt` chứa bệnh án hoặc ghi chú lâm sàng dạng free-form text.

Ví dụ:

```text
Lý do vào viện: đau đầu, sốt cao 3 ngày.
Tiền sử: đái tháo đường típ 2, tăng huyết áp.
Khám: bệnh nhân tỉnh, không ho, không khó thở.
Điều trị: Paracetamol 500 mg khi sốt.
```

Repository đọc toàn bộ nội dung file và giữ nguyên văn bản gốc. Việc giữ nguyên ký tự, khoảng trắng và xuống dòng là cần thiết để bảo toàn vị trí.

## Đầu ra

Mỗi file input tạo ra một file JSON có cùng tên. Nội dung là một danh sách concept.

```json
[
  {
    "text": "đái tháo đường típ 2",
    "position": [start, end],
    "type": "CHẨN_ĐOÁN",
    "assertions": [],
    "candidates": ["E11.9"]
  }
]
```

### Trường `text`

`text` là đúng đoạn văn bản được trích xuất từ bệnh án.

Điều kiện bắt buộc:

```python
document[start:end] == concept["text"]
```

### Trường `position`

`position` gồm hai character offset tuyệt đối:

```text
[start, end]
```

- `start` là vị trí ký tự đầu tiên
- `end` là vị trí ngay sau ký tự cuối cùng
- Cách biểu diễn này tương đương phép cắt chuỗi `text[start:end]`
- Offset được tính trên toàn bộ file, không tính riêng theo từng dòng

### Trường `type`

Repository hỗ trợ năm type:

| Type | Ý nghĩa | Có assertion | Có candidate |
|---|---|:---:|:---:|
| `TRIỆU_CHỨNG` | Triệu chứng hoặc dấu hiệu lâm sàng | Có | Không |
| `CHẨN_ĐOÁN` | Bệnh hoặc chẩn đoán | Có | ICD-10 |
| `THUỐC` | Thuốc, hoạt chất hoặc chế phẩm | Có | RxNorm |
| `TÊN_XÉT_NGHIỆM` | Tên xét nghiệm | Không | Không |
| `KẾT_QUẢ_XÉT_NGHIỆM` | Giá trị hoặc kết quả xét nghiệm | Không | Không |

Type là một phần của điều kiện matching. Một span đúng nội dung nhưng sai type vẫn được xem là dự đoán sai.

### Trường `assertions`

Assertion mô tả trạng thái của concept trong ngữ cảnh.

Các nhãn được hỗ trợ:

| Assertion | Ý nghĩa |
|---|---|
| `isNegated` | Khái niệm bị phủ định |
| `isFamily` | Khái niệm thuộc tiền sử gia đình hoặc người thân |
| `isHistorical` | Khái niệm thuộc tiền sử hoặc sự kiện trước thời điểm hiện tại |

Assertion chỉ áp dụng cho `TRIỆU_CHỨNG`, `CHẨN_ĐOÁN` và `THUỐC`.

Khi không có assertion phù hợp, trường này là danh sách rỗng:

```json
"assertions": []
```

### Trường `candidates`

Candidate là mã ontology được gán cho concept.

| Type | Ontology |
|---|---|
| `CHẨN_ĐOÁN` | ICD-10 |
| `THUỐC` | RxNorm |

Các type còn lại phải có danh sách candidate rỗng hoặc không chứa trường này, tùy theo schema đầu ra được validator chuẩn hóa.

Repository giới hạn số candidate ở mức thấp vì metric dùng Jaccard. Một mã thừa có thể làm giảm điểm đáng kể ngay cả khi mã đúng cũng đã được dự đoán.

## Matching concept

Local scorer trong repository thực hiện matching 1 đối 1 giữa dự đoán và nhãn tham chiếu.

Hai concept chỉ có thể khớp khi:

1. Cùng `type`
2. Hai span có phần ký tự giao nhau

Khi có nhiều cặp có thể khớp, scorer ưu tiên cặp có phần giao lớn hơn. Mỗi concept chỉ được ghép một lần.

Hệ quả quan trọng:

- Đúng text nhưng sai type không được tính là match
- Offset không cần trùng tuyệt đối nhưng span phải giao nhau
- Span quá rộng vẫn có thể match, tuy nhiên phần text có thể bị phạt qua WER
- Concept thừa không match sẽ làm giảm cả text score và các Jaccard liên quan

## Metric

Repository cài lại cách tính điểm của Ban Tổ chức để so sánh các lần chạy local. Đây không phải official grader của Ban Tổ chức.

Ba thành phần chính:

```text
text_score      = max(0, 1 - WER)
assertion_score = Jaccard(assertions)
candidate_score = Jaccard(candidates)
```

Điểm cuối:

```text
final_score = 100 × (
    0.3 × text_score
  + 0.3 × assertion_score
  + 0.4 × candidate_score
)
```

### Text score

Mỗi concept dự đoán khớp với một concept tham chiếu được so sánh bằng Word Error Rate riêng (WER trên đúng text của concept đó). Các giá trị `max(0, 1 − WER)` được cộng lại trên toàn corpus rồi chia cho mẫu số chung.

WER gồm ba loại lỗi:

- Substitution
- Deletion
- Insertion

WER càng thấp thì text score càng cao.

### Assertion score

Assertion score là Jaccard giữa tập assertion dự đoán và tập assertion tham chiếu trên toàn bộ corpus.

```text
Jaccard = |A ∩ B| / |A ∪ B|
```

### Candidate score

Candidate score dùng Jaccard tương tự assertion score, nhưng phần tử trong tập là cặp concept đã match và mã ontology tương ứng.

Candidate có trọng số lớn nhất trong final score, nhưng việc emit mã không chắc chắn có thể làm mẫu số tăng nhanh.

## Phạt concept thừa

Theo cách cài đặt trong [`src/medextract/scoring/scorer.py`](../src/medextract/scoring/scorer.py), một concept không match bị phạt mạnh hơn một lỗi text thông thường.

Concept thừa làm tăng mẫu số của:

- Text component
- Assertion component nếu type có assertion
- Candidate component nếu type có candidate

Vì vậy, chiến lược có precision cao thường hiệu quả hơn chiến lược thêm nhiều span hoặc nhiều mã với độ tin cậy thấp.

## Ràng buộc chính

- Tổng số tham số của các model đang được sử dụng không vượt quá 9B
- Inference phải chạy offline sau khi model và knowledge base đã được chuẩn bị
- ICD-10 được xây từ bảng tiếng Việt TT06
- RxNorm được xây từ `RXNCONSO`
- Output phải đúng schema và giữ offset chính xác

Xem kiến trúc pipeline tại [`02_method.md`](02_method.md).
