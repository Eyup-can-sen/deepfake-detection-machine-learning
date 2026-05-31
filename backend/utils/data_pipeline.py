import os
from env import load_env_variables
import zipfile
import shutil
import subprocess
import cv2
import torch
from pathlib import Path
from tqdm import tqdm
from facenet_pytorch import MTCNN

# Yolları senin yapına göre dinamik ayarlıyoruz
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = Path(r"C:\Users\ramaz\Desktop\Dataset")
RAW_DIR = DATASET_DIR / "raw"
PROCESSED_DIR = DATASET_DIR / "processed_faces"
STATS_FILE = DATASET_DIR / "stats.txt"

COMPETITION_NAME = "deepfake-detection-challenge"
# Kaggle'daki zip isimleri genelde 00.zip, 01.zip şeklindedir (Örn: 50 parça)
TOTAL_CHUNKS = 50 

class FileManager:
    """İndirme, arşivden çıkarma ve silme işlemlerini yönetir."""
    
    @staticmethod
    def ensure_directories():
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        if not STATS_FILE.exists():
            STATS_FILE.touch()

    @staticmethod
    def download_chunk(chunk_name: str) -> bool:
        zip_path = RAW_DIR / chunk_name
        if zip_path.exists():
            print(f"[BİLGİ] {chunk_name} zaten mevcut. İndirme atlanıyor...")
            return True
        
        print(f"[İNDİRİLİYOR] {chunk_name} Kaggle'dan çekiliyor...")
        try:
            # Kaggle API'sini terminal komutu olarak çağırıyoruz
            subprocess.run([
                "kaggle", "competitions", "download", 
                "-c", COMPETITION_NAME, 
                "-f", chunk_name, 
                "-p", str(RAW_DIR)
            ], check=True)
            return True
        except subprocess.CalledProcessError:
            print(f"[HATA] {chunk_name} indirilemedi. Bağlantıyı veya kaggle.json'ı kontrol et.")
            return False

    @staticmethod
    def extract_and_delete_zip(chunk_name: str) -> Path:
        zip_path = RAW_DIR / chunk_name
        extract_path = RAW_DIR / chunk_name.replace(".zip", "")
        
        print(f"[ÇIKARTILIYOR] {chunk_name} arşivden çıkarılıyor...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
            
        print(f"[TEMİZLİK] {chunk_name} (.zip dosyası) siliniyor...")
        os.remove(zip_path) # Hafızayı korumak için zip'i hemen siliyoruz
        
        return extract_path

    @staticmethod
    def cleanup_raw_folder(folder_path: Path):
        print(f"[TEMİZLİK] İşlenmiş ham videolar siliniyor: {folder_path}")
        shutil.rmtree(folder_path, ignore_errors=True)


class FaceExtractor:
    """RTX 5070 kullanarak videolardan yüz kırpma işlemlerini yapar."""
    
    def __init__(self):
        # CUDA kontrolü ve MTCNN yüz tespit modelinin yüklenmesi
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"[SİSTEM] Yüz çıkarıcı {self.device} üzerinde çalışıyor.")
        
        # margin: yüzün etrafında biraz boşluk bırakır, select_largest: sadece ana karakteri alır
        self.mtcnn = MTCNN(margin=20, keep_all=False, select_largest=True, post_process=False, device=self.device)

    def process_video(self, video_path: Path, output_dir: Path) -> int:
        """Videoyu okur, saniyede 1 kare (FPS) alarak yüzleri kaydeder."""
        cap = cv2.VideoCapture(str(video_path))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0: fps = 30 # Fallback
        
        frame_count = 0
        saved_faces = 0
        video_name = video_path.stem
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Sadece saniyede 1 kare alarak veri tekrarını önlüyor ve hafıza kurtarıyoruz
            if frame_count % fps == 0:
                # OpenCV BGR okur, MTCNN RGB bekler
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Yüzü tespit et ve kaydet
                face_filename = output_dir / f"{video_name}_frame_{frame_count}.jpg"
                
                try:
                    # MTCNN doğrudan tensör veya imaj döndürür, save_path verirsek kendi kaydeder
                    face = self.mtcnn(frame_rgb, save_path=str(face_filename))
                    if face is not None:
                        saved_faces += 1
                except Exception as e:
                    pass # Tespit edilemeyen bulanık kareleri atla
                    
            frame_count += 1
            
        cap.release()
        return saved_faces


class PipelineOrchestrator:
    """Tüm süreci yöneten ana sınıf."""
    
    def __init__(self):
        FileManager.ensure_directories()
        self.extractor = FaceExtractor()
        
    def log_stats(self, chunk_name: str, face_count: int):
        """Hocaya sunulacak istatistikleri txt dosyasına yazar."""
        with open(STATS_FILE, "a", encoding="utf-8") as f:
            f.write(f"{chunk_name} = {face_count} görsel\n")
        print(f"[BAŞARILI] {chunk_name} işlendi. Toplam {face_count} yüz kırpıldı ve kaydedildi.")

    def run(self):
        print("\n--- DEEPFAKE VERİ İŞLEME MOTORU BAŞLATILDI ---\n")
        
        for i in range(TOTAL_CHUNKS):
            # 00.zip, 01.zip formatını oluşturur
            chunk_name = f"{i:02d}.zip"
            
            # 1. İndirme (Eğer zaten varsa atlar)
            success = FileManager.download_chunk(chunk_name)
            if not success:
                continue
                
            # 2. Arşivden çıkarma ve .zip'i anında silme
            extract_folder = FileManager.extract_and_delete_zip(chunk_name)
            
            # 3. Klasördeki tüm .mp4 dosyalarını bul ve işle
            videos = list(extract_folder.rglob("*.mp4"))
            chunk_face_count = 0
            
            print(f"[İŞLENİYOR] {chunk_name} içindeki {len(videos)} video analiz ediliyor...")
            for video_path in tqdm(videos, desc="Videolar"):
                faces_saved = self.extractor.process_video(video_path, PROCESSED_DIR)
                chunk_face_count += faces_saved
                
            # 4. İstatistikleri kaydet
            self.log_stats(chunk_name, chunk_face_count)
            
            # 5. Ham videoları tamamen sil (Diski rahatlat)
            FileManager.cleanup_raw_folder(extract_folder)
            
        print("\n[BİTTİ] Tüm veri seti başarıyla işlendi ve temizlendi.")

if __name__ == "__main__":
    pipeline = PipelineOrchestrator()
    pipeline.run()