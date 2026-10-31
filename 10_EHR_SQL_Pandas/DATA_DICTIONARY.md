# Kirish ma’lumotlari lug‘ati
patients.csv, ehr_visits.csv, labs.csv: sun’iy EHR jadvallari. 2026-01-15 indeks vaqtidan keyingi ma’lumotlar ishlatilmaydi.

## ehr_visits.csv
Qatorlar soni: 180.
| Ustun | Turi | Ma’nosi |
|---|---|---|
| visit_id | int64 | Tashrifning yagona identifikatori |
| patient_id | int64 | Sun’iy bemor identifikatori yoki ochiq dataset qatori |
| visit_date | object | Tashrif vaqti |
| diagnosis | object | Sun’iy kod SYN_A/SYN_B |

## labs.csv
Qatorlar soni: 180.
| Ustun | Turi | Ma’nosi |
|---|---|---|
| patient_id | int64 | Sun’iy bemor identifikatori yoki ochiq dataset qatori |
| lab_time | object | Laboratoriya vaqti |
| glucose | float64 | Xom glyukoza, birlik glucose_unit da |

## patients.csv
Qatorlar soni: 60.
| Ustun | Turi | Ma’nosi |
|---|---|---|
| patient_id | int64 | Sun’iy bemor identifikatori yoki ochiq dataset qatori |
| birth_date | object | Tug‘ilgan sana |
