# جریان شبیه‌سازی ExU-Trans

## 1) نمای کلی

```text
HDF5 Dataset
    ↓
Volume-safe Split
    ↓
Real Training
    ↓
Best Validation Checkpoint
    ↓
Real Test Evaluation
    ↓
Measured Metrics CSV
    ↓
Measured-only Figures / Tables
    ↓
Paper Reference Comparison
```

هیچ figure یا table نباید `Measured` نامیده شود مگر اینکه از اجرای واقعی مدل ساخته شده باشد.

## 2) Dataset Loader

منطق اصلی در `scripts/dataset.py` است.

هر sample شامل:

```python
sample = {
    "image": image,
    "mask": mask,
    "volume_id": volume_id,
    "slice_id": slice_id,
}
```

split در سطح `volume` انجام می‌شود، نه slice؛ بنابراین sliceهای یک بیمار بین train/validation/test پخش نمی‌شوند.

Debug/subset mode نیز volume انتخاب می‌کند، نه چند فایل تصادفی.

## 3) Target

HDF5 فعلی:

```text
image = 240 × 240 × 4
mask  = 240 × 240 × 3
```

سه mask plane به‌صورت binary و mutually-exclusive دیده شده‌اند، اما نام دقیق آن‌ها از provenance به‌طور قطعی اثبات نشده است. بنابراین در کد از:

```text
Region_0
Region_1
Region_2
```

استفاده می‌شود.

در `regions` mode، prediction سه کانال با sigmoid threshold حفظ می‌شود و دیگر با `argmax` به یک channel collapse نمی‌شود.

## 4) Model

مدل در `scripts/model.py` قرار دارد.

`ExUTransLite` شامل تقریب پژوهشی از:

- U-Net-like encoder
- Transformer / attention blocks
- DAE approximation
- fusion
- decoder
- segmentation head

است و ادعای بازتولید دقیق معماری 50.3M پارامتری مقاله را ندارد.

## 5) Loss

در `scripts/losses.py`:

- BCE / segmentation term
- Dice term
- alignment term
- boundary term در modeهای مربوط

برای 3-channel regions، loss به‌صورت multilabel-compatible اجرا می‌شود و channelها حذف نمی‌شوند.

## 6) Training

در `scripts/train.py` و `scripts/main.py`:

```text
batch
→ forward
→ loss
→ backward
→ optimizer step
→ validation
→ save best checkpoint
```

`best_model.pt` فقط با validation انتخاب می‌شود، نه test.

## 7) Evaluation

Evaluation در سطح volume بازسازی می‌شود و metricهای زیر را می‌دهد:

- Dice
- IoU
- Precision
- Recall
- F1
- HD95_px

برای `regions` mode، metric هر `Region_0/1/2` جداگانه ذخیره می‌شود و aggregate نیز محاسبه می‌شود.

## 8) اجرای حدود دو ساعته‌ی واقعی

فایل اصلی:

```text
scripts/budget_run.py
```

Command:

```bash
python run_full_pipeline.py budget-run \
  --hours 2 \
  --epochs 2 \
  --baseline-seconds-per-batch 6.8
```

این mode:

1. dataset را دوباره دانلود نمی‌کند.
2. split اصلی volume-level را حفظ می‌کند.
3. یک subset deterministic از train/validation/test انتخاب می‌کند.
4. training واقعی انجام می‌دهد.
5. best checkpoint را ذخیره می‌کند.
6. validation/test واقعی انجام می‌دهد.
7. CSV metric واقعی تولید می‌کند.
8. figure/table واقعی تولید می‌کند.

با benchmark فعلی CPU، target تقریبی برای ۲ ساعت حدود `14–15 train volume + 3 validation + 3 test` و حداکثر `2 epoch` است. تعداد واقعی در `run_plan.json` ثبت می‌شود.

این اجرا `subset experiment` است، نه full-paper reproduction.

## 9) Measured-only Export

فایل:

```text
scripts/export_measured_paper_style.py
```

ورودی آن باید CSV واقعی باشد:

```bash
python run_full_pipeline.py export-measured \
  outputs/measured_budget_120min/metrics/metrics_test.csv \
  --output-root outputs/measured_budget_120min
```

خروجی‌های اصلی:

```text
figure_measured_metric_distributions.png
figure_measured_per_region.png
figure_measured_vs_paper.png
figure_real_qualitative_examples.png
figure_training_history.png

table_measured_summary.csv
table_measured_per_region.csv
table_measured_vs_paper.csv
```

هیچ synthetic fallback وجود ندارد. اگر metrics CSV واقعی وجود نداشته باشد exporter باید fail کند.

## 10) Reference مقاله

```text
reference/paper_targets.json
scripts/export_paper_results.py
```

Command:

```bash
python run_full_pipeline.py reference
```

این خروجی‌ها فقط `REFERENCE_ONLY` هستند.

## 11) Paper Comparison

```text
Measured Result
+
Published Reference
↓
Difference Report
```

Comparison برای metricهای هم‌نام مثل Dice/IoU/Precision/Recall/F1 قابل نمایش است، اما subset run نباید به‌عنوان result کامل مقاله معرفی شود.

HD95 تا وقتی spacing فیزیکی وجود ندارد با `px` گزارش می‌شود و مستقیماً با `mm` مقاله مقایسه نمی‌شود.

## 12) چرا خروجی‌های قدیمی سریع بودند؟

مسیر قدیمی training واقعی انجام نمی‌داد. در نبود metrics واقعی، exporter می‌توانست synthetic distribution بسازد و table/figureها را از arrayهای hard-coded صادر کند.

مسیر جدید:

```text
real data
→ real optimization
→ real prediction
→ real metrics
→ figure/table
```

بنابراین زمان بیشتر است، ولی provenance علمی دارد.

## 13) دستورات اصلی

```bash
source .venv/bin/activate

python run_full_pipeline.py preflight
python run_full_pipeline.py smoke

# اجرای واقعی کوتاه با بودجه زمانی
python run_full_pipeline.py budget-run --hours 2 --epochs 2 --baseline-seconds-per-batch 6.8

# اجرای واقعی کامل/user-configured
python run_full_pipeline.py train

# reference paper only
python run_full_pipeline.py reference

# measured vs paper
python run_full_pipeline.py compare \
  outputs/measured/metrics/metrics_test.csv \
  --target brats2020
```

## 14) محدودیت علمی

- repository کد رسمی نویسندگان نیست.
- full architecture مقاله دقیقاً قابل بازسازی نیست.
- نام دقیق channelهای mask هنوز قطعی نیست.
- subset run برای evidence واقعی سریع مناسب است، نه ادعای full reproduction.
- measured و reference همیشه جدا نگه داشته می‌شوند.
