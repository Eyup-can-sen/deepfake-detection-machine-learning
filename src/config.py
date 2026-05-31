import os

# Artık test için json dosyasının da içinde bulunduğu bu klasörü kullanıyoruz
RAW_DATA_DIR = r"D:\Deepfake_Data\train_sample_videos" 

# C Diskindeki projemizin ana dizini (Otomatik bulur)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Kırpılmış yüzleri ve işlenmiş ufak verileri kaydedeceğimiz proje içi klasör
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

# İşlenmiş veri klasörü yoksa otomatik oluştur
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)