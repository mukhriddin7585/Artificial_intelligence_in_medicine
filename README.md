# Tibbiyotda sun’iy intellekt — 15 to‘liq o‘quv loyiha

Muhammad al-Xorazmiy nomidagi TATU “Tibbiyotda sun’iy intellekt” o‘quv
qo‘llanmasining 15 amaliy mashg‘uloti uchun kodlar to‘plami.

Har bir papka mustaqil: to‘liq `main.py`, Jupyter daftari, ma’lumotlar,
versiyalar, yo‘riqnoma va bajarilganda hosil bo‘ladigan natijalar mavjud.

| Ilova | Loyiha | Papka |
|---|---|---|
| 1 | Python va Jupyter muhitida dastlabki model | `01_Python_Jupyter/` |
| 2 | Bemor oqimi va telemeditsina xizmatini tahlil qilish | `02_Bemor_oqimi/` |
| 3 | EHR ekotizimi va ma’lumot sifati auditi | `03_EHR_sifat_auditi/` |
| 4 | Tibbiy ma’lumotlarni tozalash va vizuallashtirish | `04_Healthcare_Data_Science/` |
| 5 | Diabetes datasetida regressiya va Pipeline | `05_Diabetes_regressiya/` |
| 6 | Logistik regressiya va Random Forest taqqoslash | `06_Diagnostika_modellari/` |
| 7 | TensorFlow va Keras yordamida tasvirlarni tasniflash | `07_CNN_tibbiy_tasvirlar/` |
| 8 | Sensor vaqt qatorlari va anomaliyalarni aniqlash | `08_Wearable_monitoring/` |
| 9 | Qoidaga asoslangan yordamchi tizim va qaror daraxti | `09_CDSS_qoidalar/` |
| 10 | EHR tarixidan vaqtga mos belgilar ajratish | `10_EHR_SQL_Pandas/` |
| 11 | Tibbiy startap prototipi va pilot natijalari | `11_Tibbiy_startup_pilot/` |
| 12 | O‘qitilgan model uchun tekshiriladigan FastAPI xizmati | `12_FastAPI_xizmati/` |
| 13 | Tibbiy dastur uchun dalillar va risklar inventari | `13_Risk_va_hujjatlar/` |
| 14 | Subguruhlar bo‘yicha baholash va psevdonimlashtirish | `14_Fairness_va_maxfiylik/` |
| 15 | Qayta murojaat uchun to‘liq mashinali o‘qitish loyihasi | `15_Yakuniy_loyiha/` |

## Boshlash

Python 3.12 da virtual muhit yarating va faollashtiring. Bitta loyiha
uchun uning `requirements.txt` faylidan foydalaning.

```bash
python -m pip install -r requirements.txt
python run_all.py
```

Faqat 7-loyiha TensorFlow talab qiladi. 12-loyiha FastAPI, Pydantic,
HTTPX va Uvicorn talab qiladi. Qolgan loyihalar asosiy paketlarda ishlaydi.
CPU yetarli; GPU shart emas. Paketlarni dastlab o‘rnatishda internet kerak.
Namuna ma’lumotlari bilan loyihalarni bajarish vaqtida internet kerak emas.

Jupyter uchun `requirements-notebook.txt` ham berilgan. Notebooklarni
loyiha papkasidan oching. Arxivdagi `verification_report.json` haqiqiy
bajarish tekshiruvlarini qayd qiladi.

## Ma’lumotlar kelib chiqishi

1-loyiha WDBC, 5-loyiha diabetes ochiq datasetini Scikit-learn orqali
ishlatadi. 5-loyiha regressiya: target kasallik rivojining miqdoriy
ko‘rsatkichi, diabet tashxisi emas. Qolgan loyihalarda aniq belgilangan
sun’iy o‘quv ma’lumotlari berilgan. 7-loyiha haqiqiy rentgen/MRI to‘plami
bilan ishlash uchun manifest interfeysiga ega; uning standart fantomlari
haqiqiy tibbiy tasvir emas.

13-loyiha hujjat va dalillar inventarini tekshiradi, avtomatik FDA/CE
tasnifi bermaydi. 9-loyihadagi qoida o‘quv misoli, klinik protokol emas.
Hech bir natija real bemor uchun klinik validatsiya o‘rnini bosmaydi.

## Manbalar

- https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
- https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html
- https://www.tensorflow.org/guide/keras
- https://fastapi.tiangolo.com/tutorial/testing/
- https://www.fda.gov/medical-devices/digital-health-center-excellence/software-medical-device-samd
- https://health.ec.europa.eu/medical-devices-sector/new-regulations/guidance-mdcg-endorsed-documents-and-other-guidance_en

Ma’lumotlar va kutubxonalar o‘z litsenziyalariga ega. Asosiy manba:
foydalanuvchi taqdim etgan “Tibbiyotda sun’iy intellekt” qo‘llanmasi.

## Natijalarni qayta tekshirish

`python verify_outputs.py` tayyor namuna ma’lumotlari va natijalar ustida 59 ta mazmuniy shartni tekshiradi. O‘z ma’lumotlaringizni ulasangiz, namuna sonlariga oid tekshiruvlarni moslashtiring.
