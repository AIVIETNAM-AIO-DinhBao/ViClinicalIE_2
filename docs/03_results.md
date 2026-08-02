# 03. Kết quả

## Phạm vi báo cáo

File này tổng hợp kết quả chính thức của `improved` và `improved_v2` (hai pipeline có điểm trên bảng xếp hạng Ban Tổ chức). `baseline` không có điểm chính thức và chỉ đóng vai trò tham chiếu dễ hiểu.

## Điểm theo pipeline

| Pipeline | Điểm | Ghi chú |
|---|---:|---|---|
| `improved` | 19.7343 | SapBERT retrieval và Qwen3-8B reranking |
| **`improved_v2`** | **27.8786** | Per-type thresholds, Qwen selector và precision-first linking |


## Phân tích thành phần của improved v2

| Thành phần | Giá trị |
|---|---:|
| Text score | 32.1820 |
| Assertion Jaccard | 35.2687 |
| Candidate Jaccard | 19.1084 |
| **Final score** | **27.8786** |

Kiểm tra phép tính:

```text
0.3 × 32.1820
+ 0.3 × 35.2687
+ 0.4 × 19.1084
= 27.8786
```

### Text score

Text score là thành phần đóng góp lớn thứ hai vào final score.

Pipeline dùng threshold theo từng type và hạn chế thêm span có độ tin cậy thấp. Cách này làm giảm insertion error, dù recall chưa đạt mức cao.

### Assertion score

`improved_v2` để assertions rỗng.

Assertion Jaccard vẫn đạt 35.2687 vì assertion labels trong dữ liệu đánh giá tương đối thưa. Các rule assertion thử nghiệm có xu hướng emit nhiều nhãn sai hơn số nhãn đúng bổ sung được.

### Candidate score

Candidate Jaccard là thành phần thấp nhất nhưng có trọng số lớn nhất.

Pipeline chỉ emit mã trong trường hợp unique exact alias match. Chính sách này giảm số mã sai, nhưng cũng bỏ qua nhiều concept không thể liên kết bằng exact lexical match.

## Phân bố concept

Bộ output `improved_v2` được báo cáo có 2,944 concept.

| Type | Số lượng | Tỷ lệ |
|---|---:|---:|
| `TRIỆU_CHỨNG` | 1,405 | 47.7% |
| `CHẨN_ĐOÁN` | 668 | 22.7% |
| `THUỐC` | 346 | 11.8% |
| `TÊN_XÉT_NGHIỆM` | 402 | 13.7% |
| `KẾT_QUẢ_XÉT_NGHIỆM` | 123 | 4.2% |
| **Tổng** | **2,944** | **100%** |

Phân bố này chỉ mô tả một bộ output cụ thể. Không nên dùng trực tiếp như prior cố định cho dữ liệu mới.


## Giới hạn của kết quả

- Quantization có thể làm thay đổi logits gần decision boundary
- Các threshold được tối ưu cho dữ liệu cuộc thi và có thể không phù hợp với bệnh án ngoài benchmark
- Repository không công bố dữ liệu đánh giá của Ban Tổ chức

Kết quả ablation và các hướng không hiệu quả được tổng hợp trong [`04_findings.md`](04_findings.md).
