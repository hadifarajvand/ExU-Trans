# ExU-Trans — بازسازی پژوهشی برای BraTS2020

این مخزن یک بازسازی پژوهشی / شبیه‌سازی از مقاله‌ی **ExU-Trans: a self-explanatory transformer with U-Net based hybrid model for brain tumor segmentation using MR imaging** با DOI `10.1007/s40747-026-02279-3` است.

این پروژه **کد رسمی نویسندگان نیست**. خروجی‌ها سه دسته‌ی کاملاً جدا دارند:

- `Measured`: حاصل اجرای واقعی مدل روی داده
- `Reference`: اعداد منتشرشده در مقاله
- `Smoke/Debug`: فقط تست pipeline

## وضعیت

| بخش | وضعیت |
|---|---|
| HDF5 loader | PASS |
| Dataset audit | PASS |
| Leakage check | PASS |
| 3-channel real training path | PASS |
| Real-data mini-run | PASS |
| Measured-only exporter | PASS |
| Full long training | Not completed |

## دیتاست

- Kaggle: `awsaf49/brats2020-training-data`
- URL: <https://www.kaggle.com/datasets/awsaf49/brats2020-training-data>
- Format: `HDF5`
- مسیر محلی: `dataset/fulldataset/BraTS2020_training_data/content/data`
- `57,195` فایل
- `369` volume
- `155` slice برای هر volume
- image: `240 × 240 × 4`
- mask: `240 × 240 × 3`

دیتاست داخل GitHub commit نمی‌شود و pipeline آن را دوباره دانلود نمی‌کند.

## نصب

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## تست سریع

```bash
python run_full_pipeline.py preflight
python run_full_pipeline.py smoke
```

## اجرای واقعی با بودجه‌ی حدود ۲ ساعت

برای گرفتن خروجی‌های **واقعی** بدون اجرای چندروزه‌ی کل دیتاست:

```bash
python run_full_pipeline.py budget-run \
  --hours 2 \
  --epochs 2 \
  --baseline-seconds-per-batch 6.8
```

این mode به‌صورت deterministic بخشی از volumeهای train/validation/test را از split اصلی انتخاب می‌کند. برای benchmark فعلی CPU، برنامه تقریباً در محدوده‌ی زیر هدف‌گذاری می‌شود:

- حدود `14–15` volume برای train
- حدود `3` volume برای validation
- حدود `3` volume برای test
- حداکثر `2` epoch، مگر اینکه برای حفظ time budget بعد از یک epoch متوقف شود

خروجی‌ها در مسیری شبیه زیر ذخیره می‌شوند:

```text
outputs/measured_budget_120min/
├── run_plan.json
├── run_manifest.json
├── checkpoints/
├── training/
├── metrics/
├── figures/
└── tables/
```

این خروجی‌ها واقعی هستند اما **full-paper reproduction نیستند**.

## metricهای واقعی

برای `Region_0/1/2` و aggregate، pipeline metricهای زیر را محاسبه می‌کند:

- Dice
- IoU
- Precision
- Recall
- F1
- HD95_px

`HD95_px` به معنی pixel/grid unit است؛ تا وقتی spacing فیزیکی اثبات نشده، نباید `mm` نوشته شود.

نام دقیق سه کانال HDF5 هنوز به‌صورت قطعی WT/TC/ET اثبات نشده است، بنابراین نام‌های امن `Region_0/1/2` استفاده می‌شوند.

## Export واقعی شبیه ساختار مقاله

بعد از هر اجرای measured می‌توان فقط از CSV واقعی figure/table ساخت:

```bash
python run_full_pipeline.py export-measured \
  outputs/measured_budget_120min/metrics/metrics_test.csv \
  --output-root outputs/measured_budget_120min
```

Exporter جدید:

- synthetic fallback ندارد
- metric ساختگی تولید نمی‌کند
- measured value را hard-code نمی‌کند
- distribution figure می‌سازد
- per-region table/figure می‌سازد
- measured-vs-paper comparison می‌سازد

## اجرای کامل

```bash
python run_full_pipeline.py train
```

اجرای کامل روی CPU بسیار کند است. برای مقایسه‌ی سریع و واقعی، `budget-run` توصیه می‌شود.

## Reference مقاله

```bash
python run_full_pipeline.py reference
```

خروجی‌های این command فقط از `reference/paper_targets.json` می‌آیند و **simulation result نیستند**.

## Compare

```bash
python run_full_pipeline.py compare \
  outputs/measured/metrics/metrics_test.csv \
  --target brats2020
```

Comparator فقط اختلاف را گزارش می‌کند و measured result را تغییر نمی‌دهد.

## ساختار خروجی‌ها

- `outputs/preflight/` — audit و preflight
- `outputs/smoke/` — smoke مصنوعی
- `outputs/measured/` — اجرای واقعی کامل یا user-configured
- `outputs/measured_budget_*` — اجرای واقعی time-budgeted subset
- `outputs/reference/` — اعداد مقاله
- `outputs/comparison/` — اختلاف measured/reference

## نکته‌ی مهم درباره‌ی خروجی‌های قدیمی

نسخه‌ی قدیمی repository می‌توانست خیلی سریع PNG/CSV بسازد چون exporter آن در نبود metric واقعی، synthetic data و arrayهای hard-coded استفاده می‌کرد. آن مسیر دیگر مبنای measured output نیست.

در نسخه‌ی فعلی:

```text
Real HDF5 data
→ Real training
→ Best validation checkpoint
→ Real test evaluation
→ Real metrics CSV
→ Measured-only figures/tables
```

## محدودیت علمی

- این implementation یک تقریب پژوهشی است، نه کد رسمی نویسندگان.
- معماری کامل مقاله دقیقاً منتشر نشده است.
- channel naming دقیق HDF5 هنوز قطعی نیست.
- time-budgeted subset برای تولید evidence واقعی سریع مناسب است، ولی جای full-paper experiment را نمی‌گیرد.
- هیچ خروجی نباید به‌زور با اعداد مقاله برابر شود.

برای توضیح workflow کد: `SIMULATION_WORKFLOW.md`.
