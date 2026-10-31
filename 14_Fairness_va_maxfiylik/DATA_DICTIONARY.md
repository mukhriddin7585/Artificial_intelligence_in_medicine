# Kirish ma’lumotlari lug‘ati
fairness_audit.csv: sun’iy guruhlar va oldindan yaratilgan ballar. HMAC psevdonimlashtirish anonimlashtirish kafolati emas.

## fairness_audit.csv
Qatorlar soni: 700.
| Ustun | Turi | Ma’nosi |
|---|---|---|
| patient_id | object | Sun’iy bemor identifikatori yoki ochiq dataset qatori |
| sex_group | object | Sun’iy A/B guruh; haqiqiy demografik ma’lumot emas |
| age_group | object | Sun’iy yosh guruhi |
| y_true | int64 | Sun’iy haqiqiy yorliq 0/1 |
| probability | float64 | Oldindan yaratilgan model balli [0,1] |
