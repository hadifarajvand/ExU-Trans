# Get Started — superseded

راهنمای canonical پروژه اکنون `../README.md` است.

اجرای پایه:

```bash
pip install -r requirements.txt
python run_full_pipeline.py smoke
```

برای آموزش واقعی، dataset را طبق `README.md` آماده کنید و سپس:

```bash
python run_full_pipeline.py train
```

اعداد مقاله فقط با دستور زیر به‌عنوان `REFERENCE_ONLY` صادر می‌شوند:

```bash
python run_full_pipeline.py reference
```

برای ممیزی علمی و محدودیت‌های بازتولید، `../REPRODUCIBILITY_AUDIT.md` را ببینید.
