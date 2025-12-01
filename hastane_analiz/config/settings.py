import os
from dotenv import load_dotenv
from pathlib import Path

# Proje kökünü bul (pyproject.toml'un olduğu yer)
BASE_DIR = Path(__file__).resolve().parents[2]

# .env dosyasını yükle
env_path = BASE_DIR / ".env"
if env_path.exists():
    load_dotenv(env_path)

# DB ayarları
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "hastane_analiz"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "deneme"),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
}

# Excel giriş klasörü
INPUT_FOLDER = os.getenv("INPUT_FOLDER", r"C:\veri")

# Log klasörü
LOG_FOLDER = os.getenv("LOG_FOLDER", str(BASE_DIR / "logs"))