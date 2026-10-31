# Kirish ma’lumotlari lug‘ati
patient_flow.csv: takrorlanuvchi sun’iy tashriflar. Kanallar orasidagi farq sababiy ta’sir isboti emas.

## patient_flow.csv
Qatorlar soni: 600.
| Ustun | Turi | Ma’nosi |
|---|---|---|
| visit_id | int64 | Tashrifning yagona identifikatori |
| department | object | Sun’iy bo‘lim kodi |
| channel | object | Xizmat kanali |
| arrival_time | object | Kelish vaqti |
| service_start | object | Xizmat boshlanish vaqti |
| service_end | object | Xizmat tugash vaqti |
