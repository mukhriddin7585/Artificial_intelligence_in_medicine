# Kirish ma’lumotlari lug‘ati
wearable_hr.csv: uchta sun’iy sensor oqimi. Anomaliya belgisi simulyatsiyaga tegishli; z-score klinik diagnostika mezoni emas.

## wearable_hr.csv
Qatorlar soni: 1440.
| Ustun | Turi | Ma’nosi |
|---|---|---|
| patient_id | object | Sun’iy bemor identifikatori yoki ochiq dataset qatori |
| timestamp | object | Sensor vaqt belgisi |
| heart_rate | float64 | Yurak urishi, bpm |
| steps | int64 | Bir daqiqadagi qadam |
| quality | float64 | Signal sifati, 0–1 |
| device_on | bool | Qurilma holati |
| injected_event | bool | Simulyatsiyada ataylab kiritilgan hodisa |
