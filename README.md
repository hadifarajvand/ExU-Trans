# بازتولید پژوهشی ExU-Trans برای قطعه‌بندی تومور مغزی

این مخزن یک **پیاده‌سازی پژوهشی قابل اجرا و قابل ممیزی** از ایده‌های مقاله‌ی ExU-Trans است. در این نسخه، سه نوع خروجی کاملاً از هم جدا شده‌اند:

- **Measured**: خروجی واقعی اجرای مدل روی داده.
- **Reference**: عددی که مستقیماً از مقاله رونویسی شده است.
- **Smoke/Demo**: فقط تست صحت اجرای کد؛ نتیجه‌ی پژوهشی نیست.

> **وضعیت علمی:** کد فعلی یک تقریب سبک به نام `ExUTransLite` است و بازتولید دقیق بیت‌به‌بیت مدل منتشرشده محسوب نمی‌شود. مقاله جزئیات کافی برای بازسازی دقیق معماری، splitها، اندازه‌ی resize، seed و همه‌ی تنظیمات آموزش را منتشر نکرده است. بنابراین این مخزن خروجی‌ها را به‌زور به اعداد مقاله نزدیک نمی‌کند.

## مقاله‌ی مرجع

- عنوان: *ExU-Trans: a self-explanatory transformer with U-Net based hybrid model for brain tumor segmentation using MR imaging*
- نویسندگان: G. M. Sasikala و K. Anand
- سال: 2026
- DOI: `10.1007/s40747-026-02279-3`

اعداد ارزیابی مقاله در فایل زیر نگهداری می‌شوند:

```text
reference/paper_targets.json
```

این فایل **نتیجه‌ی اجرای مدل نیست**؛ فقط رجیستری نتایج منتشرشده است.

---

## چرا این نسخه بازطراحی شد؟

نسخه‌ی قبلی اگر فایل metric واقعی را پیدا نمی‌کرد، در `scripts/export_paper_results.py` داده‌ی تصادفی تولید می‌کرد و سپس چند جدول و شکل hard-coded را به‌عنوان خروجی «paper-like» صادر می‌کرد. در نتیجه، اجرای سریع قبلی اثبات‌کننده‌ی اجرای dataset یا بازتولید مقاله نبود.

اصلاحات اصلی:

1. fallback تصادفی و مخفی حذف شده است.
2. خروجی واقعی مدل و اعداد مقاله در پوشه‌های جدا ذخیره می‌شوند.
3. Smoke Test با برچسب صریح `synthetic_smoke_only` ذخیره می‌شود.
4. آموزش واقعی بدون وجود BraTS2020 شروع نمی‌شود.
5. بهترین checkpoint بر اساس validation Dice ذخیره و قبل از test بازیابی می‌شود.
6. validation/test در سطح **case** و پس از بازسازی حجم از sliceها محاسبه می‌شوند.
7. HD95 بدون spacing فیزیکی با واحد `px` گزارش می‌شود، نه `mm`.
8. نام اجزای مقاله اصلاح شده است:
   - SE-MHA = **Self-Explanatory Multi-Head Attention**
   - DAE = **Discriminative Attribute Explainer**
   - BFM = **Bivariate Fusion Module**
9. learning rate و تعداد لایه‌های Transformer با بهترین تنظیم Table 13 هماهنگ شده‌اند: `5e-4` و `4` لایه.
10. تمام فرض‌های غیرمنتشرشده در `run_manifest.json` ثبت می‌شوند.

---

## ساختار مهم مخزن

```text
.
├── README.md
├── REPRODUCIBILITY_AUDIT.md
├── run_full_pipeline.py
├── requirements.txt
├── reference/
│   └── paper_targets.json
├── dataset/
│   └── dataset-downloader.py
├── scripts/
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── losses.py
│   ├── train.py
│   ├── main.py
│   ├── export.py
│   ├── export_paper_results.py
│   ├── compare_with_paper.py
│   └── smoke_test.py
└── outputs/
    ├── smoke/
    ├── reference/
    ├── measured/
    └── comparison/
```

---

## 1) نصب

```bash
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

کد در زمان اجرا dependency نصب نمی‌کند.

---

## 2) سریع‌ترین تست صحت اجرا

```bash
python run_full_pipeline.py smoke
```

خروجی:

```text
outputs/smoke/smoke_result.json
```

این تست forward، loss، backward و یک optimizer step را روی داده‌ی مصنوعی اجرا می‌کند. **Dice و IoU این تست نباید در مقاله، پایان‌نامه یا گزارش ارزیابی استفاده شوند.**

---

## 3) آماده‌سازی BraTS2020

اسکریپت موجود برای دانلود:

```bash
python dataset/dataset-downloader.py
```

ساختار مورد انتظار:

```text
dataset/
├── BraTS2020_TrainingData/
│   └── MICCAI_BraTS2020_TrainingData/
│       ├── BraTS20_Training_001/
│       │   ├── *_flair.nii(.gz)
│       │   ├── *_t1.nii(.gz)
│       │   ├── *_t1ce.nii(.gz)
│       │   ├── *_t2.nii(.gz)
│       │   └── *_seg.nii(.gz)
│       └── ...
└── BraTS2020_ValidationData/
    └── MICCAI_BraTS2020_ValidationData/
```

برای مسیر سفارشی:

```bash
export DATA_ROOT=/absolute/path/to/dataset
```

یا:

```bash
export DATA_ROOT_TRAIN=/absolute/path/to/MICCAI_BraTS2020_TrainingData
export DATA_ROOT_OFFICIAL_VAL=/absolute/path/to/MICCAI_BraTS2020_ValidationData
```

### درباره‌ی dataset snippet

هیچ patient data واقعی به‌طور ساختگی در Git قرار داده نشده است. `.gitignore` از commit شدن تصادفی NIfTIها جلوگیری می‌کند. Smoke Test بدون dataset کار می‌کند؛ اما برای بازتولید عددی، باید داده‌ی واقعی و مجاز BraTS یا snippet واقعی موردنظر کاربر در ساختار بالا موجود باشد.

---

## 4) آموزش واقعی

```bash
python run_full_pipeline.py train
```

اجرای کوچک برای بررسی:

```bash
EPOCHS=2 USE_DEBUG_SUBSET=1 DEBUG_NUM_CASES=2 python run_full_pipeline.py train
```

پیش‌فرض‌های مهم:

```text
seed               = 42
split              = 70/15/15   # فرض مخزن؛ split IDs مقاله منتشر نشده‌اند
image_size         = 128        # فرض مخزن؛ اندازه دقیق resize مقاله منتشر نشده
learning_rate      = 0.0005     # بهترین مقدار Table 13
transformer_layers = 4          # بهترین مقدار Table 13
label_mode         = whole_tumor
```

خروجی واقعی:

```text
outputs/measured/
├── run_manifest.json
├── checkpoints/best_model.pt
├── metrics/metrics_validation.csv
├── metrics/metrics_test.csv
└── figures/
```

`run_manifest.json` را همراه نتیجه نگه دارید؛ تنظیمات و فرض‌ها در آن ثبت می‌شوند.

---

## 5) خروجی اعداد خود مقاله

```bash
python run_full_pipeline.py reference
```

خروجی‌ها:

```text
outputs/reference/
```

تمام فایل‌های این پوشه **REFERENCE_ONLY** هستند و نتیجه‌ی simulation نیستند.

---

## 6) مقایسه‌ی measured با paper

Table 1:

```bash
python run_full_pipeline.py compare \
  outputs/measured/metrics/metrics_validation.csv \
  --target table1
```

BraTS2020 در Table 3:

```bash
python run_full_pipeline.py compare \
  outputs/measured/metrics/metrics_test.csv \
  --target brats2020
```

خروجی:

```text
outputs/comparison/paper_alignment.json
```

Comparator فقط اختلاف را گزارش می‌کند؛ metric واقعی را تغییر نمی‌دهد.

---

## اعداد مرجع اصلی مقاله

### Table 1

| Metric | مقدار |
|---|---:|
| DSC | 90.2% |
| IoU | 84.1% |
| HD | 3.0 mm |
| Precision | 89.6% |
| Recall | 87.7% |
| F1 | 88.6% |

### Table 3 — BraTS2020

| Metric | مقدار |
|---|---:|
| DSC | 90.6% |
| IoU | 84.5% |
| HD | 2.8 mm |
| Precision | 89.8% |
| Recall | 88.1% |
| F1 | 88.9% |

### ناسازگاری مهم داخل مقاله

- Table 1: DSC = `90.2%`
- Table 3 برای BraTS2020: DSC = `90.6%`
- Tables 11–13 برای Full/Optimized model: DSC = `92.0%`
- متن مقاله برای WT/ET/TC: `0.962 / 0.941 / 0.944`

بنابراین یک «عدد واحد» برای همه‌ی بخش‌های مقاله وجود ندارد. target مقایسه باید صریح انتخاب شود.

---

## تفاوت معماری فعلی با مقاله

ممیزی نسخه‌ی قبلی:

```text
Transformer blocks: 2
Parameters: 2,719,948 (~2.72M)
```

نسخه‌ی بازطراحی‌شده با چهار block:

```text
Transformer blocks: 4
Parameters: 3,149,516 (~3.15M)
```

Table 5 مقاله برای ExU-Trans مقدار `50.3M` پارامتر گزارش می‌کند. این اختلاف بزرگ است و نشان می‌دهد implementation موجود **معادل دقیق معماری منتشرشده نیست**. چون مقاله تمام channel widthها و جزئیات لازم را منتشر نکرده، افزایش مصنوعی تعداد پارامترها فقط برای رسیدن به `50.3M` کار علمی قابل دفاعی نیست.

---

## مواردی که در این مخزن ممنوع‌اند

- ساخت metric تصادفی وقتی نتیجه‌ی واقعی موجود نیست؛
- hard-code کردن عدد مقاله و نامیدن آن به‌عنوان simulation output؛
- نوشتن `HD95 (mm)` بدون استفاده از spacing فیزیکی معتبر؛
- مقایسه‌ی slice-level با case/volume-level بدون هشدار؛
- ادعای «100% match» صرفاً به دلیل شباهت ظاهر figure یا CSV؛
- تنظیم مدل فقط برای نزدیک کردن اعداد به مقاله بدون protocol معتبر.

---

## تعریف Full Reproduction

برای ادعای بازتولید کامل، حداقل باید موارد زیر فراهم باشند:

1. dataset کامل و مجاز؛
2. split IDs دقیق و ثبت‌شده؛
3. preprocessing/augmentation با provenance؛
4. معماری و تعداد پارامتر قابل تطبیق با مقاله؛
5. checkpoint بهترین validation؛
6. test فقط بعد از انتخاب checkpoint؛
7. metricهای case/volume-level؛
8. HD95 با spacing فیزیکی و واحد mm؛
9. اجرای واقعی baselineها؛
10. p-value از paired observations واقعی؛
11. اجرای واقعی ablation/noise/generalization؛
12. traceability هر figure/table به metric خام، config و checkpoint.

تا قبل از تکمیل این موارد، عبارت صحیح **research approximation / partial reproduction** است، نه exact reproduction.

برای جزئیات ممیزی، فایل `REPRODUCIBILITY_AUDIT.md` را بخوانید.
