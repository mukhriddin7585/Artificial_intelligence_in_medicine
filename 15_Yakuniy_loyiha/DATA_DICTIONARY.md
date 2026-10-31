# Kirish ma’lumotlari lug‘ati
medical_project.csv: sun’iy takroriy tashriflar. Target 30 kunlik simulyatsiya natijasi; klinik samaradorlik haqidagi da’vo berilmaydi.

## medical_project.csv
Qatorlar soni: 905.
| Ustun | Turi | Ma’nosi |
|---|---|---|
| patient_id | object | Sun’iy bemor identifikatori yoki ochiq dataset qatori |
| encounter | int64 | Sun’iy takroriy tashrif raqami |
| age | int64 | Yosh, yil |
| bmi | float64 | Tana massasi indeksi, kg/m² |
| previous_visits | int64 | Indeks sanasidan oldingi tashriflar soni |
| comorbidity | int64 | Sun’iy hamroh holatlar soni |
| sex_group | object | Sun’iy A/B guruh; haqiqiy demografik ma’lumot emas |
| target | int64 | Sun’iy hosil qilingan target, 0/1; klinik tashxis emas |
