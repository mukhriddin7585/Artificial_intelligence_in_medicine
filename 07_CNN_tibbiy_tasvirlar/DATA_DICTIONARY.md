# Kirish ma’lumotlari lug‘ati
Standart demo: sun’iy tomografik fantomlar, haqiqiy rentgen/MRI emas. Haqiqiy PNG/JPG tasvirlar data/manifest.csv orqali ulanadi. patient_id, path, label (0 yoki 1) ustunlari majburiy.

## manifest.csv
Qatorlar soni: 200.
| Ustun | Turi | Ma’nosi |
|---|---|---|
| patient_id | object | Sun’iy bemor identifikatori yoki ochiq dataset qatori |
| path | object | data/ ga nisbatan PNG/JPG yo‘li |
| label | int64 | 0/1; demo: sun’iy marker yo‘q/bor |
