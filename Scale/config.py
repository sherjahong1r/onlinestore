"""
Loyiha sozlamalari.
Haqiqiy muhitda bu qiymatlarni .env fayl yoki environment variable orqali olish tavsiya etiladi.
"""

DB_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "dbname": "chem_scada",
    "user": "postgres",
    "password": "jahongir",
}

# PLC ogohlantirish va avariya chegaralari (operator SCADA orqali o'zgartirishi mumkin)
LIMITS = {
    "temperature": {"warn": 90.0, "trip": 100.0},
    "pressure":    {"warn": 18.0, "trip": 20.0},
    "ph_low":      {"warn": 5.5,  "trip": 4.0},
    "ph_high":     {"warn": 8.5,  "trip": 10.0},
    "vibration":   {"warn": 6.0,  "trip": 9.0},
}

SIMULATION_INTERVAL_SEC = 2      # har necha soniyada bitta o'lchov
AI_ANALYSIS_EVERY_N_CYCLES = 15  # AI tahlili har 15 sikldan keyin ishga tushadi
