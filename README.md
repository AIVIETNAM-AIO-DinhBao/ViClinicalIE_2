# medextract

Baseline không chính thức cho bài toán trích xuất và chuẩn hóa khái niệm y khoa từ bệnh án tiếng Việt trong **Viettel AI Race 2026, Track 2**.

Repository cung cấp ba pipeline từ cơ bản đến nâng cao. Người mới nên bắt đầu với `baseline` để hiểu toàn bộ quy trình; `improved_v2` là phiên bản đạt điểm cao nhất.

## Tổng quan

Hệ thống nhận đầu vào là các file bệnh án dạng `.txt` và trả về một file JSON cho mỗi bệnh án. Mỗi concept có thể gồm span, type, assertion và mã ontology tương ứng.

```text
Bệnh án .txt
    → phát hiện span (NER)
    → phân loại type
    → xác định assertion
    → liên kết ICD-10 hoặc RxNorm
    → kiểm tra schema
    → concepts.json
```

## Các pipeline

| Pipeline          | Config                     |        Điểm | Thành phần chính                                      | Mục đích                         |
| ----------------- | -------------------------- | ----------: | ----------------------------------------------------- | -------------------------------- |
| `baseline`        | `configs/baseline.yaml`    |           — | GLiNER, ConText rules, SapBERT, FAISS                 | Baseline dễ hiểu, không dùng LLM |
| `improved`        | `configs/improved.yaml`    |     19.7343 | Baseline và Qwen3-8B listwise reranker                | Thử nghiệm retrieve-then-rerank  |
| **`improved_v2`** | `configs/improved_v2.yaml` | **27.8786** | GLiNER theo từng type, Qwen selector, exact-alias linking | Pipeline có điểm cao nhất        |

Điểm lấy từ bảng xếp hạng của Ban Tổ chức. Chi tiết kết quả và phân tích nằm trong [`docs/03_results.md`](docs/03_results.md).

## Colab notebooks

Mỗi notebook chạy được trên một Colab **T4 (16 GB)**.

| Pipeline    | Mở notebook                                                                                                                                                                                                          |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Baseline    | [![Mở Baseline trên Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/duongtruongbinh/viettel_ai_race_task2/blob/main/notebooks/colab_baseline.ipynb)       |
| Improved    | [![Mở Improved trên Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/duongtruongbinh/viettel_ai_race_task2/blob/main/notebooks/colab_improved.ipynb)       |
| Improved v2 | [![Mở Improved v2 trên Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/duongtruongbinh/viettel_ai_race_task2/blob/main/notebooks/colab_improved_v2.ipynb) |

## Quick Start

```bash
git clone https://github.com/duongtruongbinh/viettel_ai_race_task2.git
cd viettel_ai_race_task2
conda create -n medextract python=3.10 -y && conda activate medextract
pip install -e .                     # cài core; dùng pip install -e ".[quant]" cho 4/8-bit
python scripts/selfcheck.py         # CONFIG / IMPORTS / SCHEMA / PATHS phải PASS cả bốn
# chuẩn bị knowledge base (xem INSTALL.md), rồi chạy:
python run.py --config configs/improved_v2.yaml \
              --input examples/input --output out/improved_v2 --zip
```

`examples/input/` đã kèm hai bệnh án mẫu `001.txt` và `002.txt` để chạy thử ngay. Kết quả ghi vào `out/improved_v2/`, và `submission.zip` (các file JSON nằm phẳng) được tạo khi có cờ `--zip`.

## Documents

| Tài liệu                                   | Nội dung                                   |
| ------------------------------------------ | ------------------------------------------ |
| [`INSTALL.md`](INSTALL.md)                 | Cài đặt, chuẩn bị knowledge base, chạy, đóng gói, đánh giá local |
| [`docs/01_problem.md`](docs/01_problem.md) | Bài toán, schema, matching và metric       |
| [`docs/02_method.md`](docs/02_method.md)   | Kiến trúc và rationale của ba pipeline     |
| [`docs/03_results.md`](docs/03_results.md) | Kết quả và phân tích thành phần            |
| [`docs/04_findings.md`](docs/04_findings.md) | Ablation study và các kết quả âm         |

## Cấu trúc repository

```text
.
├── configs/             # Cấu hình của ba pipeline (baseline / improved / improved_v2)
├── docs/                # Tài liệu kỹ thuật
├── examples/input/      # Hai bệnh án mẫu dùng cho notebook và chạy thử
├── notebooks/           # Google Colab notebooks
├── scripts/             # scripts/selfcheck.py
├── src/medextract/      # Mã nguồn chính
├── INSTALL.md
├── README.md
├── run.py               # Điểm vào inference theo config
├── score.py             # Đánh giá local theo công thức Ban Tổ chức
└── pyproject.toml       # Lệnh cài đặt duy nhất: pip install -e .
```
