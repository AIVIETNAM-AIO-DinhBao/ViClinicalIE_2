# 04. Findings

## Mục tiêu

File này tổng hợp các bài học rút ra từ ablation và error analysis trong quá trình phát triển repository.

Các kết luận được chia thành hai nhóm:

- **Structural finding** là kết luận gắn với cơ chế metric hoặc thiết kế pipeline
- **Distribution-sensitive finding** là kết luận phụ thuộc mạnh vào dữ liệu đánh giá và cần được kiểm tra lại khi dataset thay đổi

Không nên xem mọi threshold hoặc rule trong file này là giá trị tối ưu chung cho clinical NLP.

## 1. Precision quan trọng hơn việc tăng số lượng dự đoán

Metric phạt concept thừa ở nhiều thành phần cùng lúc.

Một span thừa có thể làm giảm:

- Text score
- Assertion Jaccard
- Candidate Jaccard

Kết quả thực nghiệm cho thấy:

| Thử nghiệm | Kết quả |
|---|---|
| Hạ nhiều threshold để tăng recall | Điểm host giảm khoảng 3.4 |
| Hạ riêng threshold của `THUỐC` | Điểm host giảm khoảng 0.33 |
| Giữ nhiều candidate hơn | Candidate Jaccard giảm |
| Thêm span có candidate | Mẫu số candidate tăng nhanh |

**Kết luận:** Khi chưa có confidence đủ mạnh, không emit thường tốt hơn emit một concept hoặc candidate có xác suất sai cao.

Đây là lý do `improved_v2` dùng threshold theo type, giới hạn candidate và chỉ bổ sung span ở type không có candidate.

## 2. Sửa type có mục tiêu hiệu quả hơn lọc span tổng quát

GLiNER có khả năng phát hiện span tương đối tốt, nhưng thường nhầm bệnh mạn tính thành triệu chứng.

Lỗi quan trọng nhất:

```text
TRIỆU_CHỨNG → CHẨN_ĐOÁN
```

Một type corrector tập trung vào lỗi này có ích hơn một span classifier cố gắng quyết định giữ hoặc bỏ mọi span.

Các span filter tổng quát đạt AUC khoảng 0.70 nhưng vẫn không tạo ra lợi ích ổn định cho final score.

**Kết luận:** Khi NER đã có ranking hợp lý, model phụ nên sửa một lỗi cụ thể và có tác động lớn, thay vì thay thế toàn bộ quyết định của NER.

## 3. Two-teacher consensus phù hợp cho span additions

Một teacher đơn lẻ có thể thêm nhiều span đúng, nhưng cũng dễ thêm span sai.

`improved_v2` yêu cầu hai teacher đồng thuận trước khi bổ sung span và chỉ cho phép ba type không có candidate:

```text
TRIỆU_CHỨNG
TÊN_XÉT_NGHIỆM
KẾT_QUẢ_XÉT_NGHIỆM
```

Cách này giảm rủi ro ở candidate component.

**Kết luận:** Consensus hữu ích khi mục tiêu là tăng recall có kiểm soát. Consensus không cần áp dụng cho mọi quyết định. Trong config hiện tại, type correction chỉ dùng teacher chính, còn span additions mới dùng cả hai teacher.

## 4. Exact lexical linking tốt hơn dense retrieval trong cấu hình này

Các phương pháp đã thử:

- SapBERT nearest-neighbor retrieval
- Dense retrieval
- Qwen3-8B listwise reranking
- Hybrid lexical retrieval
- Unique exact alias emission

Kết quả cho thấy lexical exact match có precision tốt nhất.

Hai nguyên nhân chính:

1. Nhiều target code không thể suy ra chắc chắn chỉ từ mention
2. Dense retriever và reranker có xu hướng chọn một mã ngay cả khi mention chưa đủ cụ thể

Một số ablation:

| Thử nghiệm | Chỉ số liên quan | Kết quả |
|---|---|---|
| Bắt buộc mention có ít nhất 3 từ mới link | Conditional code precision | 0.231 xuống 0.101 |
| Bỏ ICD `.9` remap | Conditional code precision | 0.231 xuống 0.186 |
| Bỏ toàn bộ candidate | Candidate Jaccard | Giảm 0.0036 |
| Candidate confidence gate | Final score | Tăng khoảng 0.0006, dưới noise floor |
| SapBERT và Qwen listwise reranker | Candidate quality | Kém hơn exact lexical policy |

**Kết luận:** Candidate emission phải được xem là một quyết định selective prediction, không phải lúc nào cũng chọn top-1.

## 5. Assertions là điểm khó vì nhãn thưa

Các phương pháp đã thử:

- Vietnamese ConText rules
- LLM assertion classifier
- Chỉ emit khi confidence cao
- Không emit assertion

Kết quả:

| Assertion | AUC |
|---|---:|
| `isNegated` | 0.497 |
| `isHistorical` | 0.727 |

`isNegated` gần mức ngẫu nhiên trong thử nghiệm. `isHistorical` tốt hơn nhưng chưa đủ ổn định để vượt ngưỡng sử dụng an toàn.

**Kết luận:** Với split hiện tại, assertions rỗng cho Jaccard tốt hơn rule-based hoặc LLM-based emission.

Kết luận này phụ thuộc dữ liệu. Trong một dataset có assertion labels dày và nhất quán hơn, assertion model có thể trở thành thành phần quan trọng.

## 6. Mở rộng span một cách mù quáng thường gây hại

Các rule mở rộng span được thử để thêm:

- Dosage và route cho thuốc
- Giá trị và unit cho kết quả xét nghiệm
- Toàn bộ dòng hoặc đoạn cho kết quả
- Các span từ deterministic segmentation

Kết quả:

| Thử nghiệm | Thay đổi |
|---|---:|
| Mở rộng drug span không kiểm soát | Text component giảm khoảng 0.51 |
| Mở rộng result span đến hết dòng | Text component giảm khoảng 0.25 |
| Mở rộng result span theo đoạn | Text component giảm khoảng 0.10 |
| Deterministic segmentation làm span source | Exact-match rate không quá 0.144 |

**Kết luận:** Boundary convention của dữ liệu không thể được suy ra chỉ bằng rule dài hơn. Mỗi rule mở rộng cần có điều kiện chặt và phải được đánh giá bằng paired comparison.

## 7. Merge type sau NER không thay thế được type modeling

Thử nghiệm merge `TÊN_XÉT_NGHIỆM` và `KẾT_QUẢ_XÉT_NGHIỆM` sau NER làm chất lượng text giảm mạnh.

Paired text score giảm từ khoảng 0.65 xuống 0.09 và mất 51 concept đã match.

**Kết luận:** Hai type này có boundary và chức năng khác nhau. Không nên hợp nhất chỉ vì model thường nhầm giữa chúng.

## 8. Feature engineering không tạo ra span verifier đủ mạnh

Một nhóm feature được thử gồm:

- Span length
- Token count
- GLiNER score
- Type one-hot
- Header-like indicator
- Lexical features
- Context features

Phần lớn feature không tạo ra separation đủ rõ. Length feature là nhóm ổn định nhất nhưng vẫn không đủ để quyết định drop span.

Một independent detector dựa trên LLM đạt AUC khoảng 0.50 trong thử nghiệm.

**Kết luận:** Khi negative examples không có cấu trúc rõ, thêm classifier sau NER có thể chỉ học lại score của NER hoặc overfit split nội bộ.

## 9. Một số local gain không đủ để đưa vào config mặc định

Các thay đổi rất nhỏ có thể tăng điểm trên một split nhưng không ổn định khi đổi fold hoặc đổi dữ liệu.

Config public chỉ giữ những thay đổi:

- Có lý do rõ
- Có tác động lặp lại
- Không phá schema
- Không phụ thuộc vào một vài sample riêng biệt

Các lever dưới đây từng cho local gain nhỏ trên split nội bộ nhưng không lặp lại khi đổi fold, nên bị loại khỏi config mặc định (chỉ `boundary.generic_prefix` được giữ bật):

| Lever | Vai trò dự kiến | Bằng chứng loại |
|---|---|---|
| `codedness_gate` | Chỉ emit candidate khi mention đủ "có mã" | Final score thay đổi dưới noise floor, không ổn định khi đổi fold |
| `drug_cleanup` | Làm sạch lại span thuốc sau NER | Text component giảm khi override cleaner chung |
| `boundary.symptom_boundary_trim` | Cắt rìa span triệu chứng | Paired text score giảm, quy ước boundary không suy ra được bằng rule |
| `consensus_selector.corrector_consensus` | Yêu cầu hai teacher đồng thuận khi sửa type | Cost tăng, lợi ích sát noise floor, corrector dùng teacher chính là đủ |

**Kết luận:** Chỉ số tăng dưới noise floor không nên được xem là cải tiến thực sự.

## 10. Bài học thiết kế pipeline

### Nên làm

- Giữ offset trên văn bản gốc
- Tách retrieval và emission policy
- Dùng threshold riêng theo type
- Giới hạn candidate
- Dùng LLM cho quyết định có label space nhỏ
- Đánh giá từng thay đổi bằng paired comparison
- Giữ lại negative results để tránh lặp lại thử nghiệm

### Không nên làm

- Hạ threshold đồng loạt
- Bắt reranker luôn chọn một mã
- Mở rộng span đến hết dòng chỉ bằng rule
- Merge type sau NER mà không có evidence
- Dùng local distribution như prior cố định
- Xem quantization là tương đương hoàn toàn với full precision

## Kết luận

Cải tiến lớn nhất của repository không đến từ việc thêm nhiều model hoặc tăng recall tối đa.

Kết quả tốt hơn đến từ ba quyết định:

1. Sửa đúng lỗi type có tác động lớn
2. Chỉ bổ sung span khi rủi ro thấp
3. Không emit candidate khi mention chưa đủ chắc chắn
