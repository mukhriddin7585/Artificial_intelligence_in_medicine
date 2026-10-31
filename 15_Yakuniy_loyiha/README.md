# 15-loyiha — Qayta murojaat uchun to‘liq mashinali o‘qitish loyihasi

Qo‘llanmaning 15-amaliy mashg‘ulotiga mos mustaqil bajariladigan loyiha.

## Ma’lumot

medical_project.csv: sun’iy takroriy tashriflar. Target 30 kunlik simulyatsiya natijasi; klinik samaradorlik haqidagi da’vo berilmaydi.

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

Jupyter/Colab uchun `15_Yakuniy_loyiha.ipynb` faylini ochib, barcha kataklarni tartib
bilan bajaring. Daftar to‘liq kodni o‘zida saqlaydi; `main.py`ni import
qilish talab etilmaydi. Colabda kerakli paketlarni `!pip install` orqali
o‘rnating. Kiritilgan `data/` fayllarini ishlatish uchun ularni daftar
ishlayotgan papkaga yuklang; aks holda namuna qayta yaratiladi.

## Fayllar va natija

- `main.py` — to‘liq Python kodi.
- `15_Yakuniy_loyiha.ipynb` — bir xil kod va natijalar uchun Jupyter daftari.
- `requirements.txt` — bog‘liqliklar versiyalari.
- `data/` — kirish ma’lumotlari va ularning tavsifi.
- `outputs/` — CSV/JSON, grafik, hisobot yoki saqlangan model.
- `DATA_DICTIONARY.md` — kirish fayllaridagi ustunlar va turlar.

Standart `SEED=42`. O‘qitish, preprocessing va threshold tanlashda
test ma’lumotlari ishlatilmaydi. HMAC tokenlari yangi maxfiy kalit bilan
qayta yaratilgani sababli 14-loyihadagi tokenlar o‘zgarishi mumkin.

## Natijani talqin qilish

Bu o‘quv loyihasi. Sun’iy namuna yoki ochiq datasetdagi texnik ko‘rsatkich
real bemor uchun klinik samaradorlikni tasdiqlamaydi. Ma’lumot kelib chiqishi,
sinflar ma’nosi, o‘lchov birligi va cheklovlar hisobotda saqlanadi.
