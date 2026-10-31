# Kirish ma’lumotlari lug‘ati
startup_pilot.csv: sun’iy ekspert yorliqlari, model ballari va vaqt o‘lchovlari. KPIlar haqiqiy klinik yoki biznes natijasi emas.

## startup_pilot.csv
Qatorlar soni: 400.
| Ustun | Turi | Ma’nosi |
|---|---|---|
| case_id | int64 | Pilot holat identifikatori |
| y_true | int64 | Sun’iy haqiqiy yorliq 0/1 |
| ai_probability | float64 | Sun’iy model balli [0,1] |
| turnaround_min | float64 | Yordamchi rejim vaqti, daqiqa |
| baseline_min | float64 | Bazaviy rejim vaqti, daqiqa |
| override | bool | Sun’iy ekspert modelni o‘zgartirgan holat |
| site | object | Sun’iy markaz kodi |
