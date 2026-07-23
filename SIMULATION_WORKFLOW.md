# جریان شبیه‌سازی ExU-Trans

## 1) نمای کلی

```text
HDF5 Dataset
    ↓
Dataset Loader
    ↓
Train / Validation / Test Split
    ↓
ExUTransLite Model
    ↓
Loss
    ↓
Training
    ↓
Best Checkpoint
    ↓
Evaluation
    ↓
Metrics
    ↓
Figures / Tables
    ↓
Paper Comparison
```

## 2) Dataset Loader

منطق اصلی در `scripts/dataset.py` قرار دارد. هر sample از فایل HDF5 خوانده می‌شود و به‌صورت یک نمونه‌ی یکتا با اطلاعات volume و slice برگردانده می‌شود.

```python
sample = {
    "image": image,
    "mask": mask,
    "volume_id": volume_id,
    "slice_id": slice_id,
}
```

loader روی split سطح volume کار می‌کند تا leakage بین train/validation/test رخ ندهد. در این نسخه، image چهار کانال و mask سه کانال دارد.

## 3) Configuration

تنظیمات در `scripts/config.py` تعریف می‌شوند. مهم‌ترین گزینه‌ها:

- `seed`
- `learning_rate`
- `epochs`
- `batch_size`
- `device`
- `dataset path`
- `label mode`
- `Transformer layers`

بعضی مقدارها با environment variable قابل override هستند.

## 4) Model

مدل اصلی در `scripts/model.py` است. `ExUTransLite` یک تقریب پژوهشی از ایده‌های مقاله است و اجزای اصلی زیر را دارد:

- encoder سبک شبیه U-Net
- attention / transformer blocks
- DAE approximation
- fusion
- decoder
- segmentation head

`ExUTransLite` ادعای بازتولید دقیق پارامترها و widths مقاله را ندارد.

## 5) Loss

منطق loss در `scripts/losses.py` است. بخش‌های مهم:

- pixel segmentation loss
- Dice-related component
- alignment / attribute component
- boundary component

جزئیات دقیق loss به `label_mode` وابسته است و باید با target فعلی هم‌خوان باشد.

## 6) Training

Training در `scripts/train.py` و `scripts/main.py` اجرا می‌شود.

```text
batch
→ forward
→ loss
→ backward
→ optimizer step
→ validation
→ save best checkpoint
```

بهترین checkpoint با validation انتخاب می‌شود و در `best_model.pt` ذخیره می‌گردد.

## 7) Evaluation

Evaluation روی volume/sliceهای validation و test انجام می‌شود و metricهای معمول را گزارش می‌کند:

- Dice
- IoU
- Precision
- Recall
- F1
- HD95_px اگر spacing فیزیکی قابل اتکا نباشد

واحد pixel را نباید با millimetre اشتباه گرفت.

## 8) Export

`scripts/export.py` خروجی‌های measured را به شکل CSV و figure و table می‌سازد. همه‌ی شکل‌ها باید از داده‌ی واقعی بیایند، نه از مقدارهای ساختگی.

## 9) Paper Comparison

`reference/paper_targets.json` اعداد منتشرشده‌ی مقاله را نگه می‌دارد و `scripts/compare_with_paper.py` اختلاف measured و reference را محاسبه می‌کند.

```text
Measured Result
+
Published Reference
↓
Difference Report
```

Comparator فقط اختلاف را گزارش می‌کند؛ هیچ metricی را تغییر نمی‌دهد.

## 10) خروجی‌های تاریخی

برخی خروجی‌های قدیمی مخزن از نظر موضوع کلی به بخش ارزیابی مقاله شبیه‌اند، اما شماره‌گذاری، نوع نمودار و منبع داده در همه‌ی موارد با شکل‌های مقاله یکسان نیستند. بنابراین باید آن‌ها را `supporting visualization` یا `historical / paper-style outputs` دانست، نه reproduction exact.

## 11) اجرای مرحله‌به‌مرحله

```bash
source .venv/bin/activate

python run_full_pipeline.py preflight
python run_full_pipeline.py smoke
python run_full_pipeline.py train
python run_full_pipeline.py evaluate
python run_full_pipeline.py export
python run_full_pipeline.py reference
python run_full_pipeline.py compare \
  outputs/measured/metrics/metrics_test.csv \
  --target brats2020
```

## 12) یادداشت علمی کوتاه

- این repository یک تقریب پژوهشی است.
- کد رسمی نویسندگان نیست.
- معماری کامل مقاله به‌صورت دقیق و کامل منتشر نشده است.
- naming دقیق سه کانال HDF5 هنوز قطعی نیست.
- خروجی‌ها نباید به‌زور با مقاله یکسان شوند.
