# Kirish ma’lumotlari lug‘ati
ehr_sample.csv va ehr_messages.json: xatolari ataylab kiritilgan sun’iy EHR; note matni tuzilmaviy maydon emas.

## ehr_sample.csv
Qatorlar soni: 181.
| Ustun | Turi | Ma’nosi |
|---|---|---|
| record_id | int64 | Yozuv identifikatori |
| patient_id | object | Sun’iy bemor identifikatori yoki ochiq dataset qatori |
| age | object | Yosh, yil |
| visit_date | object | Tashrif vaqti |
| glucose | float64 | Xom glyukoza, birlik glucose_unit da |
| glucose_unit | object | mmol/L yoki mg/dL; noma’lum birlik audit qilinadi |
| note | object | Sun’iy erkin matnli qayd |
