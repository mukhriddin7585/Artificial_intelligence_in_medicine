# Kirish ma’lumotlari lug‘ati
cdss_patients.csv: sun’iy holatlar. Qoidalar qo‘llanmadagi o‘quv misolini bajaradi, klinik protokol yoki tavsiya emas.

## cdss_patients.csv
Qatorlar soni: 500.
| Ustun | Turi | Ma’nosi |
|---|---|---|
| patient_id | int64 | Sun’iy bemor identifikatori yoki ochiq dataset qatori |
| age | int64 | Yosh, yil |
| temperature | float64 | Harorat, °C |
| heart_rate | float64 | Yurak urishi, bpm |
| spo2 | float64 | O‘quv simulyatsiyasidagi SpO₂, foiz |
| comorbidity | int64 | Sun’iy hamroh holatlar soni |
| confusion | bool | Sun’iy holat belgisi |
| outcome | int64 | Sun’iy simulyatsiya natijasi, 0/1 |
