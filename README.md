# Tibbiyotda sun’iy intellekt — 15 ta amaliy mashg‘ulot

“Tibbiyotda sun’iy intellekt” o‘quv qo‘llanmasidagi 15 ta amaliy mashg‘ulotning
Python kodlari. Har bir amaliyot **bitta mustaqil `.py` fayl** sifatida berilgan.
Fayllar raqamlanishi qo‘llanmaning I–XV boblariga mos keladi.

To‘plam ma’lumot tayyorlash, modellashtirish, tibbiy tasvirlar bilan ishlash,
elektron tibbiy kartalar, API, risklarni tahlil qilish va yakuniy loyiha
jarayonini qamrab oladi. Kodlar o‘quv va tadqiqot namunalari hisoblanadi;
ularning natijalari real bemor uchun klinik validatsiya dalili emas.

## Tarkib

| № | Amaliy mashg‘ulot | Python fayl | Asosiy natija |
|---|---|---|---|
| 1 | Muhit, kutubxonalar va bazaviy model | `01_muhit_va_bazaviy_model.py` | Muhit versiyalari, tasnif metrikalari, grafik |
| 2 | Bemor oqimi va telemeditsina | `02_bemor_oqimi_va_telemeditsina.py` | Sun’iy tashriflar CSV fayli, kanal hisoboti va dashboard |
| 3 | Tibbiy ma’lumotlar ekotizimi va EHR auditi | `03_ehr_csv_auditi.py` | Missing value hisoboti va sifat muammolari |
| 4 | Healthcare Data Science | `04_tibbiy_data_cleaning.py` | Xom/tozalangan ma’lumotlar, karantin va tozalash auditi |
| 5 | Mashinali o‘qitish asoslari | `05_diabetes_model.py` | Sintetik diabetes tasnifi, confusion matrix, koeffitsiyentlar |
| 6 | Tibbiy diagnostika va prognoz | `06_diagnostika_va_prognoz.py` | Validation taqqoslash, threshold tahlili, alohida test metrikalari |
| 7 | Tibbiy tasvirlarni CNN orqali tahlil qilish | `07_cnn_tibbiy_tasvirlar.py` | Sintetik fantomlar, train/validation/test, CNN modeli |
| 8 | Raqamli salomatlik va monitoring | `08_wearable_monitoring.py` | Sensor vaqt qatorlari va anomaliya ro‘yxati |
| 9 | Klinik qarorlarni qo‘llab-quvvatlash | `09_cdss_risk_scoring.py` | O‘quv qoidalari, risk ballari va qaror daraxti |
| 10 | Elektron tibbiy kartalar va SQL | `10_ehr_sql_feature_extraction.py` | SQLite bazasi va vaqt chegarasiga mos belgilar |
| 11 | Innovatsion texnologiya va startap prototipi | `11_medical_startup_prototype.py` | Qayta qabul modeli, natijalar va loyiha KPI fayli |
| 12 | Modelni amaliy xizmat sifatida taqdim etish | `12_fastapi_tibbiy_model.py` | O‘qitilgan model, tekshirilgan FastAPI endpointlari |
| 13 | Sertifikatlash va tartibga solish | `13_regulatory_risk_checklist.py` | Ta’limiy checklist va hayotiy sikl sxemasi |
| 14 | Etika, adolatlilik va maxfiylik | `14_fairness_va_privacy.py` | Subguruh metrikalari, tafovutlar va psevdonim eksport |
| 15 | Yakuniy loyiha | `15_yakuniy_tibbiy_loyiha.py` | Sifat auditi, model tanlash, threshold, test va model card |

Qo‘shimcha fayllar: `requirements.txt`, `requirements-cnn.txt`, `.gitignore`
va ushbu `README.md`. Repozitoriyda jami **15 ta Python fayli** mavjud.

## Muhitni tayyorlash

Tavsiya etilgan muhit: **Python 3.12, 64 bit**. Asosiy amaliyotlar CPU’da
bajariladi. 7-amaliyot kichik CNN bilan ishlaydi; GPU talab etilmaydi.
Paketlarni o‘rnatish uchun internet zarur. Standart namunalarda amaliyotlarning
o‘zi tashqi dataset yuklab olmaydi.

Arxivni oching va terminalda loyihaning ildiz papkasiga o‘ting:

```bash
cd Tibbiyotda_SI_15_Python_GitHub
python -m venv .venv
```

Windows PowerShell’da virtual muhitni faollashtirish:

```powershell
.\.venv\Scripts\Activate.ps1
```

Windows CMD’da:

```bat
.venv\Scripts\activate.bat
```

Linux yoki macOS’da:

```bash
source .venv/bin/activate
```

Asosiy paketlarni o‘rnating:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

7-amaliyot uchun qo‘shimcha o‘rnating:

```bash
python -m pip install -r requirements-cnn.txt
```

Versiyalar Python 3.12/Linux muhitidagi tekshiruvga mos qayd etilgan.
Boshqa operatsion tizimlarda TensorFlow distributivi va platformaga mos
wheel mavjudligi o‘rnatilayotgan muhitga bog‘liq.

## Alohida amaliyotni ishga tushirish

```bash
python 01_muhit_va_bazaviy_model.py
python 05_diabetes_model.py
python 07_cnn_tibbiy_tasvirlar.py --epochs 5
python 15_yakuniy_tibbiy_loyiha.py
```

Har bir fayl mustaqil ishlaydi. Boshqa amaliyotni oldindan bajarish talab
qilinmaydi. Tasvirlar alohida oynada ochilmaydi: grafiklar PNG ko‘rinishida
saqlanadi, hisob-kitoblar esa terminalda chiqariladi.

Ma’lumotlar va natija yo‘llari skript turgan joyga nisbatan aniqlanadi:

```text
Tibbiyotda_SI_15_Python_GitHub/
  01_muhit_va_bazaviy_model.py
  ...
  15_yakuniy_tibbiy_loyiha.py
  README.md
  requirements.txt
  requirements-cnn.txt
  .gitignore
  data/       # ayrim amaliyotlarning avtomatik yaratiladigan kirish fayllari
  outputs/    # har bir amaliyot uchun alohida natija papkasi
```

Masalan, yakuniy loyiha natijalari
`outputs/15_yakuniy_tibbiy_loyiha/` ichida saqlanadi.
`data/` va `outputs/` kataloglari birinchi ishga tushirishda yaratiladi va
`.gitignore` orqali Git indeksiga avtomatik qo‘shilmaydi. Qayta ishga
tushirish tegishli natija fayllarini yangilaydi. 2-, 3- va 7-amaliyotlar
mavjud kirish fayllarini qayta ishlatadi; yangi standart namuna uchun faqat
tegishli amaliyotning `data/<fayl_nomi>/` papkasini tozalash yetarli.

## Barcha 15 ta amaliyotni bajarish

Avval ikkala requirements faylidagi paketlarni o‘rnating. Linux/macOS Bash’da:

```bash
for script in [0-9][0-9]_*.py; do
  python "$script" || break
done
```

Windows PowerShell’da:

```powershell
Get-ChildItem -File | Where-Object { $_.Name -match '^\d{2}_.*\.py$' } | Sort-Object Name | ForEach-Object {
    python $_.FullName
    if ($LASTEXITCODE -ne 0) { throw "Amaliyot bajarilmadi: $($_.Name)" }
}
```

12-amaliyot standart rejimda serverni ochmaydi, ichki API tekshiruvlarini
bajarib yakunlanadi. Shu sababli barcha fayllarni ketma-ket bajarish mumkin.

## 7-amaliyot: tasvirlar va CNN

Standart rejimda 100 ta sun’iy bemor uchun 200 ta kulrang **sintetik fantom**
yaratiladi. `label=1` sun’iy marker borligini, `label=0` marker yo‘qligini
anglatadi. Bu tasvirlar haqiqiy rentgen, MRT yoki kasallik tasviri emas.

Train/validation/test qismlari `patient_id` bo‘yicha 60/20/20 nisbatda
ajratiladi. Bir bemorning ikki tasviri turli qismlarga tushmaydi. Standart
o‘qitish 5 epoxa; epoxalarni buyruq bilan o‘zgartirish mumkin:

```bash
python 07_cnn_tibbiy_tasvirlar.py --epochs 20
```

Model validation F1 ko‘rsatkichi bo‘yicha thresholdni tanlaydi va test
qismida baholanadi. O‘qitish tarixi, split, metrikalar, PNG va `.keras`
modeli `outputs/07_cnn_tibbiy_tasvirlar/` ichiga yoziladi. 5 epoxa o‘quv
namoyishi uchun belgilangan; aniqlikning muayyan qiymati kafolatlanmaydi.

O‘z tasvirlaringiz uchun `data/07_cnn_tibbiy_tasvirlar/manifest.csv` yarating:

```csv
patient_id,path,label
P001,images/p001_1.png,0
P001,images/p001_2.png,0
P002,images/p002_1.png,1
```

`path` shu `data/07_cnn_tibbiy_tasvirlar/` papkasiga nisbatan beriladi.
Har bir bemor uchun yagona label kerak. Har ikki sinfda train, validation
va testga ajratish uchun yetarli mustaqil bemor bo‘lishi zarur. Yuklashda
rasm kulrang 32×32 formatiga keltiriladi; bu kichik o‘quv CNN’i, real
tasvirlar uchun sifat va rezolyutsiya strategiyasi alohida tanlanadi.

## 12-amaliyot: FastAPI xizmati

Qo‘llanmadagi model o‘qitish, API va test mijozining vazifalari **bitta
faylga** birlashtirilgan. Standart buyruq modelni o‘qitadi va server portini
ochmasdan `/health` hamda `/predict` endpointlarini tekshiradi:

```bash
python 12_fastapi_tibbiy_model.py
```

To‘g‘ri so‘rov uchun `200`, manfiy qiymat, noma’lum maydon va yetishmayotgan
maydon uchun `422` kutiladi. Hisobot `outputs/12_fastapi_tibbiy_model/`
ichidagi `12_api_checks.json` fayliga saqlanadi.

Mahalliy serverni ochish:

```bash
python 12_fastapi_tibbiy_model.py --serve --port 8000
```

Brauzerda `http://127.0.0.1:8000/docs` manzilini oching. `POST /predict`
uchun namuna JSON:

```json
{
  "mean_radius": 14.2,
  "mean_texture": 19.0,
  "mean_perimeter": 92.0,
  "mean_area": 655.0,
  "mean_smoothness": 0.10
}
```

Bu belgilar WDBC to‘plamidagi hujayra yadrolari tasvirlaridan hisoblangan
sonli tavsiflardir. Ular bemorning yoshi yoki odatiy laboratoriya tahlillari
emas. Xizmat har ishga tushganda modelni o‘qitadi va lokal namoyish uchun
`127.0.0.1` interfeysiga ulanadi. Serverni `Ctrl+C` bilan to‘xtating.

## Ma’lumotlar va natijalarni talqin qilish

- **1, 6, 12 va 15-amaliyotlar:** scikit-learn bilan birga keladigan WDBC
  ochiq o‘quv to‘plami. 1-amaliyot asl kodlashni saqlaydi: `0=malignant`,
  `1=benign`; 6, 12 va 15-amaliyotlarda `1=malignant` qilib qayta kodlanadi.
- **2–5, 7–11 va 14-amaliyotlar:** kod bilan yaratiladigan sun’iy ma’lumotlar.
  5-amaliyotdagi diabetes nomli target ta’limiy sinfdir; u haqiqiy diabet
  kohortasidan olingan tashxis emas. 11-amaliyotdagi KPI tavsiflari loyiha
  rejasiga misol, amalda o‘lchangan klinik samaradorlik emas.
- **6 va 15-amaliyotlar:** model tanlash validation qismida bajariladi;
  yakuniy test alohida qoladi. 15-amaliyot threshold tanlangach modelni
  qayta o‘qitmaydi. Preprocessing modelning train qismida o‘rganiladi.
- **8-amaliyot:** butun vaqt qatori bo‘yicha retrospektiv anomaliya tahlili.
  Markazlashtirilgan rolling oyna va butun qatorga moslangan Isolation
  Forest real vaqt prognozi sifatida talqin qilinmaydi.
- **9-amaliyot:** qoidalar va risk ballari o‘quv misoli, tasdiqlangan klinik
  protokol emas. **13-amaliyot:** ball ichki o‘quv ustuvorligi bo‘lib,
  huquqiy qurilma sinfi, sertifikat yoki FDA/CE qarorini bildirmaydi.
- **14-amaliyot:** HMAC kaliti muhitdan olinadi; berilmasa tasodifiy yangi
  kalit yaratiladi va saqlanmaydi. Bir xil tokenlarni qayta olish uchun
  `TSI_PSEUDONYM_KEY` muhit o‘zgaruvchisiga o‘zingiz boshqaradigan kalitni
  kiriting. Psevdonimlashtirish anonimlashtirish degani emas.

Haqiqiy bemor ma’lumotlari va maxfiy kalitlarni ommaviy repozitoriyga
joylashtirmang. `data/`, `outputs/` va `.env` avtomatik chetlatilgan.

## GitHub’ga joylash

GitHub’da yangi bo‘sh repozitoriy yarating. ZIP’ni ochib, **ichidagi
fayllarni** repozitoriy ildiziga joylang; ZIP faylining o‘zini kod o‘rnida
yuklash shart emas. Terminaldan foydalanilsa:

```bash
git init
git add .
git commit -m "Add 15 AI in medicine practical lessons"
git branch -M main
git remote add origin https://github.com/USERNAME/REPOSITORY.git
git push -u origin main
```

`USERNAME` va `REPOSITORY` o‘rniga o‘z GitHub ma’lumotlaringizni yozing.
Yangi repozitoriyda README yoki boshqa boshlang‘ich fayl yaratilgan bo‘lsa,
avval uni klonlab, ushbu to‘plam fayllarini o‘sha papkaga ko‘chiring.

## Kodlarning moslashtirilishi

Asos: foydalanuvchi taqdim etgan “Tibbiyotda sun’iy intellekt” o‘quv
qo‘llanmasining 15 dasturiy ilovasi va avvalgi to‘liq loyihalar to‘plami.
GitHub uchun yo‘llar amaliyotlar bo‘yicha ajratildi, grafiklar faylga
saqlashga moslandi, 12-amaliyot bitta skriptga birlashtirildi. 7-amaliyotda
bemor guruhlari bo‘yicha ajratish, 10-amaliyotda vaqt chegarasi,
14-amaliyotda kalitli psevdonimlashtirish hamda 6 va 15-amaliyotlarda
validation/test ajratilishi saqlandi yoki aniqlashtirildi.

## Bajarish tekshiruvi

Python 3.12/Linux muhitida **15 ta skriptning barchasi xatosiz bajarildi**.
CNN standart 5 epoxa bilan tekshirildi. 44 ta natija tekshiruvi, jumladan
bemorlar bo‘yicha splitlarning ajralganligi, EHR vaqt chegarasi, eksport
ustunlari, HMAC tokenlari, API javoblari va PNG fayllarining hosil bo‘lishi
muvaffaqiyatli o‘tdi. Bu tekshiruvlar dasturlarning texnik bajarilishiga
taalluqli; klinik samaradorlikni tasdiqlamaydi.

Dataset va uchinchi tomon kutubxonalarining foydalanish shartlari ularning
o‘z manbalariga tegishli. Kodni qayta tarqatish litsenziyasi repozitoriy
egasi tomonidan alohida belgilanadi.
