# Kirish ma’lumotlari lug‘ati
regulatory_inventory.csv va hazards.csv: o‘quv hujjatlari. Dastur FDA/CE tasnifi, ruxsat yoki muvofiqlik xulosasini bermaydi.

## hazards.csv
Qatorlar soni: 3.
| Ustun | Turi | Ma’nosi |
|---|---|---|
| hazard | object | Xavf manbai |
| harm | object | Ehtimoliy zarar |
| severity | int64 | Ichki o‘quv shkalasi 1–5 |
| likelihood | int64 | Ichki o‘quv ehtimollik shkalasi 1–5 |
| proposed_control | object | Taklif etilgan nazorat chorasi |

## regulatory_inventory.csv
Qatorlar soni: 10.
| Ustun | Turi | Ma’nosi |
|---|---|---|
| item | object | Dalil yoki hujjat elementi |
| status | object | Inventar holati |
| evidence_file | float64 | Loyiha ichidagi dalil fayli yo‘li |
| owner | object | O‘quv mas’uli |
