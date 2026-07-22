# Run Guide — راهنمای اجرا

راهنمای اصلی و به‌روز پروژه در `README.md` است.

## اجرای سریع

```bash
pip install -r requirements.txt
python run_full_pipeline.py smoke
```

Smoke Test فقط صحت اجرای کد را با داده‌ی مصنوعی بررسی می‌کند و نتیجه‌ی پژوهشی نیست.

## اجرای واقعی با BraTS2020

بعد از قرار دادن dataset در مسیر توضیح‌داده‌شده در `README.md`:

```bash
python run_full_pipeline.py train
```

خروجی واقعی فقط در `outputs/measured/` ذخیره می‌شود.

## اعداد مقاله

```bash
python run_full_pipeline.py reference
```

فایل‌های `outputs/reference/` فقط اعداد رونویسی‌شده از مقاله هستند و simulation output نیستند.

## مقایسه

```bash
python run_full_pipeline.py compare outputs/measured/metrics/metrics_test.csv --target brats2020
```

برای محدودیت‌ها، ناسازگاری‌های داخل مقاله و ممیزی نسخه‌ی قبلی، `REPRODUCIBILITY_AUDIT.md` را بخوانید.
