# ExU-Trans — بازسازی پژوهشی برای BraTS2020

این مخزن یک بازسازی پژوهشی / شبیه‌سازی از مقاله‌ی **ExU-Trans: a self-explanatory transformer with U-Net based hybrid model for brain tumor segmentation using MR imaging** با DOI `10.1007/s40747-026-02279-3` است.

این پروژه **کد رسمی نویسندگان** نیست. هدف آن اجرای قابل‌ممیزی، ثبت فرض‌ها، و نگه‌داری خروجی‌های `Measured` / `Reference` / `Debug` به‌صورت جداگانه است.

## وضعیت فعلی

| بخش | وضعیت |
|---|---|
| HDF5 loader | PASS |
| Dataset audit | PASS |
| Leakage check | PASS |
| Smoke test | PASS |
| Real-data mini-run | PASS |
| Evaluate / Compare | PASS |
| Full long training | Not completed |

## دیتاست

دیتاست منتخب:

- Kaggle: `awsaf49/brats2020-training-data`
- URL: <https://www.kaggle.com/datasets/awsaf49/brats2020-training-data>
- Format: `HDF5`
- مسیر محلی: `dataset/fulldataset/BraTS2020_training_data/content/data`

ویژگی‌های تأییدشده:

- `57,195` فایل `HDF5`
- `369` volume
- `155` slice برای هر volume
- تصویر: `240 × 240 × 4`
- mask: `240 × 240 × 3`

دیتاست داخل GitHub commit نمی‌شود و دانلود مجدد خودکار هم غیرفعال است.

## نصب

```bash
python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows:

```bash
.venv\Scripts\activate
```

## اجرای سریع

```bash
python run_full_pipeline.py preflight
python run_full_pipeline.py smoke
```

## تست واقعی کوچک

```bash
EPOCHS=1 \
USE_DEBUG_SUBSET=1 \
DEBUG_NUM_CASES=2 \
python run_full_pipeline.py train
```

این اجرا فقط برای sanity check است و نتیجه‌ی نهایی پژوهشی نیست.

## آموزش

```bash
python run_full_pipeline.py train
```

زمان اجرا به سخت‌افزار وابسته است. در همین محیط، اجرای کامل CPU بسیار کند است.

## ارزیابی و خروجی

```bash
python run_full_pipeline.py evaluate
python run_full_pipeline.py export
python run_full_pipeline.py reference
python run_full_pipeline.py compare \
  outputs/measured/metrics/metrics_test.csv \
  --target brats2020
```

## ساختار خروجی‌ها

- `outputs/preflight/` — گزارش‌های پیش‌اجرایی و audit
- `outputs/smoke/` — خروجی smoke مصنوعی
- `outputs/debug_real_data/` — خروجی debug واقعی برچسب‌خورده
- `outputs/measured/` — خروجی اجرای واقعی مدل
- `outputs/reference/` — اعداد منتشرشده‌ی مقاله
- `outputs/comparison/` — گزارش اختلاف measured و reference

## Measured / Reference / Debug

| نوع | معنا |
|---|---|
| Measured | اجرای واقعی مدل روی داده |
| Reference | عدد منتشرشده در مقاله |
| Debug | فقط تست pipeline |

## محدودیت علمی

- این implementation یک تقریب پژوهشی است، نه کد رسمی نویسندگان.
- معماری کامل مقاله به‌صورت دقیق و کامل منتشر نشده است.
- نام دقیق سه کانال HDF5 هنوز از provenance محلی به‌صورت قطعی اثبات نشده است.
- full long-run در محیط CPU فعلی تکمیل نشده است.
- خروجی‌ها نباید به‌زور با مقاله برابر شوند.

برای جزئیات بیشتر:

- `REPRODUCIBILITY_AUDIT.md`
- `SIMULATION_WORKFLOW.md`

## یادداشت

اگر فقط می‌خواهی pipeline را سریع چک کنی، از `smoke` استفاده کن. اگر می‌خواهی اجرای واقعی کوچک ببینی، از `train` با `USE_DEBUG_SUBSET=1` استفاده کن.
