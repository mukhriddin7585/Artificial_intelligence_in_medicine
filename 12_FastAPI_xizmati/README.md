# 12-loyiha — O‘qitilgan model uchun tekshiriladigan FastAPI xizmati

Qo‘llanmaning 12-amaliy mashg‘ulotiga mos mustaqil bajariladigan loyiha.

## Ma’lumot

api_training.csv: sun’iy age, bmi, systolic_bp belgilaridan hosil qilingan o‘quv target. Natija klinik xavf ehtimoli emas.

Standart ma’lumotlar `data/` ichida berilgan. Fayl bo‘lmasa kod uni qayta
yaratadi yoki Scikit-learn tarkibidagi ochiq datasetni eksport qiladi.
Sun’iy ma’lumotlar bemorlarning haqiqiy shaxsiy ma’lumotlarini saqlamaydi.

## Ishga tushirish

Python 3.12 tavsiya etiladi. Terminalni shu loyiha papkasida oching.

```bash
python -m venv .venv
```

Windows:

```bat
.venv\Scripts\activate
python -m pip install -r requirements.txt
python main.py
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

Jupyter/Colab uchun `12_FastAPI_xizmati.ipynb` faylini ochib, barcha kataklarni tartib
bilan bajaring. Daftar to‘liq kodni o‘zida saqlaydi; `main.py`ni import
qilish talab etilmaydi. Colabda kerakli paketlarni `!pip install` orqali
o‘rnating. Kiritilgan `data/` fayllarini ishlatish uchun ularni daftar
ishlayotgan papkaga yuklang; aks holda namuna qayta yaratiladi.

## Fayllar va natija

- `main.py` — to‘liq Python kodi.
- `12_FastAPI_xizmati.ipynb` — bir xil kod va natijalar uchun Jupyter daftari.
- `requirements.txt` — bog‘liqliklar versiyalari.
- `data/` — kirish ma’lumotlari va ularning tavsifi.
- `outputs/` — CSV/JSON, grafik, hisobot yoki saqlangan model.
- `DATA_DICTIONARY.md` — kirish fayllaridagi ustunlar va turlar.

Standart `SEED=42`. O‘qitish, preprocessing va threshold tanlashda
test ma’lumotlari ishlatilmaydi. HMAC tokenlari yangi maxfiy kalit bilan
qayta yaratilgani sababli 14-loyihadagi tokenlar o‘zgarishi mumkin.

`python main.py` API uchun avtomatik 200/401/422 tekshiruvlarini bajaradi.
Mahalliy xizmatni ishga tushirish:

```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Brauzerda `http://127.0.0.1:8000/docs` manzilini oching.
`POST /predict`: `{"age":60,"bmi":27.5,"systolic_bp":135.0}`.
`X-API-Key` sarlavhasi kerak. Standart demo kalit:
`local-demo-key-change-me`. `TSI_API_KEY` muhit o‘zgaruvchisi bilan
almashtiriladi. Xizmat o‘quv prototipi, klinik tizimga tayyor mahsulot emas.

## Natijani talqin qilish

Bu o‘quv loyihasi. Sun’iy namuna yoki ochiq datasetdagi texnik ko‘rsatkich
real bemor uchun klinik samaradorlikni tasdiqlamaydi. Ma’lumot kelib chiqishi,
sinflar ma’nosi, o‘lchov birligi va cheklovlar hisobotda saqlanadi.
