# Cài đặt và vận hành

Đây là tài liệu duy nhất về triển khai: cài đặt, chuẩn bị knowledge base, chạy inference, đóng gói bản nộp, đánh giá local và xử lý lỗi thường gặp.

## 1. Môi trường

Yêu cầu:

- Python 3.10 trở lên
- Git
- GPU được khuyến nghị (CUDA tương thích bản PyTorch đang dùng)
- Ổ đĩa đủ cho ba model và knowledge base

Tạo môi trường và cài đặt. `pyproject.toml` là nguồn phụ thuộc duy nhất — một lệnh `pip install -e .` là đủ, không cần file phụ thuộc riêng.

```bash
git clone https://github.com/duongtruongbinh/viettel_ai_race_task2.git
cd viettel_ai_race_task2
conda create -n medextract python=3.10 -y
conda activate medextract
pip install -e .                     # core (torch, gliner, faiss-cpu, transformers, ...)
```

Khi cần 4-bit hoặc 8-bit (improved / improved_v2):

```bash
pip install -e ".[quant]"            # bitsandbytes + accelerate
```

Kiểm tra sau khi cài (không cần GPU, không cần knowledge base):

```bash
python scripts/selfcheck.py          # CONFIG / IMPORTS / SCHEMA / PATHS phải PASS cả bốn
```

## 2. Knowledge base

Bước linking so khớp đề cập với hai ontology: **RxNorm** (thuốc) và **ICD-10** (chẩn đoán).

> Tên nguồn ICD-10 dùng trong tài liệu này là **TT06/2026/TT-BYT (Phụ lục danh mục mã ICD-10 tiếng Việt)**. File
> `data/kb/raw/Phu_luc_Bang_danh_muc_ICD10_FINAL_TT06_2026.xlsx` **đã đi kèm repository**, nên bạn chỉ cần tự tải
> RxNorm. TT06 thay thế danh mục cũ QĐ 4469/QĐ-BYT nhưng giữ nguyên bố trí cột, nên script đọc hai nguồn giống nhau;
> tên viết tắt trong code và cờ dòng lệnh là `TT06`.

### Nguồn RxNorm (cần tự tải)

RxNorm có licence UMLS nên **không** đi kèm repo. Đặt file nguồn tại `data/kb/raw/`. Ba tuỳ chọn, theo thứ tự khuyến nghị:

- **(a) `RXNCONSO.RRF` từ bản release RxNorm đầy đủ (dòng monthly) của NLM.** Cần tài khoản UMLS/UTS (miễn phí). Đặt tại `data/kb/raw/RXNCONSO.RRF`. Đây là nguồn dùng cho bộ nộp đã nộp.
- **(b) RxNorm Current Prescribable Content.** Không cần giấy phép, tải trực tiếp từ trang RxNorm. Bản này dạng CSV/TSV (có header), **không** phải pipe — đặt tại `data/kb/raw/rxnorm_rxnconso.csv` (đuôi `.csv`/`.tsv`). Đừng dùng đuôi `.rrf`: build script phân biệt pipe-RRF và CSV/TSV theo đuôi file, nên đặt sai đuôi sẽ parse nhầm. Phù hợp để chạy thử nhanh, nhưng tập mã hẹp hơn bản đầy đủ nên kết quả sẽ khác bộ nộp đã nộp.
- **(c) CSV cùng bố trí cột với `RXNCONSO`** (ví dụ export từ Kaggle). Đặt tại `data/kb/raw/rxnorm_rxnconso.csv`. Script tự nhận diện RRF so với CSV.

### Build theo pipeline

Đặt các file nguồn RxNorm vào `data/kb/raw/` rồi build theo pipeline định chạy. File TT06 `.xlsx` đã có sẵn trong repo tại cùng thư mục.

**`baseline` và `improved`** — dùng SapBERT retrieval qua FAISS nên cần cả parquet lẫn index:

```bash
python -m medextract.kb.build_rxnorm          # -> data/kb/processed/rxnorm_terms.parquet
python -m medextract.kb.build_icd             # -> data/kb/processed/icd_terms.parquet
python -m medextract.kb.index --device auto   # -> data/kb/processed/{RXNORM,ICD10}.faiss
```

`--device auto` tự chọn GPU còn trống, nếu không có GPU thì dùng CPU (chậm hơn nhưng vẫn chạy).

**`improved_v2`** — exact-alias lookup trên hai bảng parquet giàu alias, **không** cần SapBERT và **không** cần FAISS index, nên chỉ hai lệnh là đủ:

```bash
python -m medextract.kb.build_icd    --tt06     # -> data/kb/processed/icd_terms_v2.parquet
python -m medextract.kb.build_rxnorm --v2       # -> data/kb/processed/rxnorm_terms_v2.parquet
```

`build_icd` không có cơ chế tải tự động danh mục ICD-10 tiếng Anh: nó chỉ đọc file `*.xlsx`/`*.xls` trong `data/kb/raw/` (chính là file TT06 đã commit trong repo).

## 3. Chạy inference

Mỗi lần chạy đều cần KB đã build (xem mục 2). Đặt các file `.txt` trong một thư mục input (ví dụ `data/input/` hoặc dùng sẵn `examples/input/`).

```bash
# Baseline
python run.py --config configs/baseline.yaml    --input examples/input --output out/baseline    --zip

# Improved
python run.py --config configs/improved.yaml    --input examples/input --output out/improved    --zip

# Improved v2
python run.py --config configs/improved_v2.yaml --input examples/input --output out/improved_v2 --zip
```

Mỗi lệnh tạo các file JSON trong thư mục output, mỗi file `.txt` một file `.json` cùng stem (`001.txt → 001.json`).

### Đóng gói bản nộp

Khi có cờ `--zip`, file nộp được lưu tại `<output>/submission.zip`, các file JSON nằm phẳng ở cấp root, không có thư mục con:

```text
out/improved_v2/
├── 001.json
├── 002.json
└── submission.zip
```

### Offline mode

Lần chạy đầu tải model từ Hugging Face Hub. Khi model đã có trong cache, thêm `--offline` để chạy hoàn toàn offline:

```bash
python run.py --config configs/improved_v2.yaml --input examples/input --output out/improved_v2 --zip --offline
```

## 4. Đánh giá local

`score.py` là bản đọc lại công thức chấm của Ban Tổ chức để so sánh các lần chạy local, **không** phải official grader. Chuẩn bị thư mục nhãn dạng `<thư mục nhãn>/{stem}.json` cùng schema với bản nộp (ví dụ `data/dev/`):

```bash
python score.py --pred out/improved_v2 --gold data/dev -v
```

In điểm text / assertions / candidates / FINAL theo công thức `final = 0.3·text + 0.3·assertions + 0.4·candidates` (×100).

## 5. Quantization

Giới hạn 9B của cuộc thi tính trên **số tham số**, không phải bộ nhớ, nên quantize các teacher không thay đổi việc tuân thủ luật. Cài backend rồi chọn chế độ:

```bash
pip install -e ".[quant]"      # bitsandbytes + accelerate
```

| Phiên bản    | Config key / flag                                | Cái được quantize               |
| ------------ | ------------------------------------------------ | ------------------------------- |
| `improved`   | `llm.load_in_4bit: true`                         | listwise reranker Qwen3-8B      |
| `improved_v2`| `quantization.mode` / `--quantize {8bit,4bit}`   | hai teacher của selector        |

VRAM ước tính cho hai teacher của `improved_v2` (hai teacher có thể chia sẻ cùng một GPU):

| Chế độ                  | VRAM (2 teacher) | Ghi chú                       |
| ----------------------- | ---------------: | ----------------------------- |
| `none` (bfloat16)       |          ~18 GB  | lần chạy tham chiếu           |
| `8bit`                  |          ~10 GB  | có thể lệch logits nhẹ        |
| `4bit` (nf4)            |           ~6 GB  | hai teacher vừa một GPU T4    |

Config tham chiếu (mặc định trong `configs/improved_v2.yaml`):

```yaml
quantization:
  mode: none
  dtype: bfloat16
```

Đổi sang 4-bit qua CLI mà không sửa YAML:

```bash
python run.py --config configs/improved_v2.yaml --quantize 4bit \
              --input examples/input --output out/improved_v2_4bit --zip
```
