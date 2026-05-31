import os
from pathlib import Path
from dotenv import load_dotenv

# .env dosyasını yükle (Proje kök dizinindeki .env dosyasını arar)
load_dotenv()

# .env içerisinden veri setinin ana yolunu çek, eğer .env yoksa veya silindiyse hata ver
ENV_DATASET_DIR = os.getenv("DATASET_ROOT_DIR")

if not ENV_DATASET_DIR:
    raise ValueError("KRİTİK HATA: .env dosyasında DATASET_ROOT_DIR bulunamadı! Lütfen .env dosyanızı kontrol edin.")

DATASET_ROOT_DIR = Path(ENV_DATASET_DIR)

# Alt klasör yollarını tek merkezden dağıtıyoruz
RAW_DATA_DIR = DATASET_ROOT_DIR / "raw"
PROCESSED_DATA_DIR = DATASET_ROOT_DIR / "processed_faces"
STATS_FILE = DATASET_ROOT_DIR / "stats.txt"

# Sistem başlatıldığında bu klasörlerin var olduğundan emin ol, yoksa oluştur
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)