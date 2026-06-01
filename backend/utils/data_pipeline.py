import os
import sys
import json
import zipfile
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed  # PARALEL İŞLEME İÇİN EKLENDİ

# Proje ana dizinini yollara ekliyoruz
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from dotenv import load_dotenv
load_dotenv()

# Yeni Kaggle Token Tabanlı Kimlik Doğrulama (KGAT_ formatı)
KAGGLE_API_TOKEN = os.getenv("KAGGLE_API_TOKEN")

if not KAGGLE_API_TOKEN:
    raise ValueError("KRİTİK HATA: .env dosyasında KAGGLE_API_TOKEN bulunamadı!")

# Kaggle kütüphanesi bu ortam değişkenini okur
os.environ['KAGGLE_API_TOKEN'] = KAGGLE_API_TOKEN

# Token'ı ~/.kaggle/access_token dosyasına da yaz (yedek olarak)
_kaggle_dir = Path.home() / ".kaggle"
_kaggle_dir.mkdir(parents=True, exist_ok=True)
_access_token_path = _kaggle_dir / "access_token"
_access_token_path.write_text(KAGGLE_API_TOKEN)

# KAGGLE API'YI BAŞLATIYORUZ
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image

try:
    from facenet_pytorch import MTCNN
    _HAS_MTCNN = True
except ImportError:
    MTCNN = None
    _HAS_MTCNN = False

from backend.src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

COMPETITION_NAME = "deepfake-detection-challenge"
FRAMES_PER_VIDEO = 5 

# Kaggle artık DFDC verilerini Dataset olarak sunuyor
DFDC_DATASETS = [
    {
        "slug": "pranay22077/dfdc-10",
        "label": "DFDC Part 0-9",
    },
    {
        "slug": "pranay22077/dfdc-10-deepfake-detection-challenge-pt-2-10-19",
        "label": "DFDC Part 10-19",
    },
]

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

_FACE_CASCADE = None
if not _HAS_MTCNN:
    _FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def _detect_face_box(frame_rgb, mtcnn):
    if _HAS_MTCNN and mtcnn is not None:
        boxes, _ = mtcnn.detect(frame_rgb)
        if boxes is None: return None
        return boxes[0]

    if _FACE_CASCADE is None or _FACE_CASCADE.empty(): return None
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0: return None
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    return [x, y, x + w, y + h]

class DatasetDownloader:
    """Kaggle Dataset API üzerinden DFDC verilerini indirir."""
    @staticmethod
    def download_dataset(dataset_slug: str, target_dir: Path) -> bool:
        """Bir Kaggle dataset'ini toplu olarak indirir ve çıkartır."""
        marker_file = target_dir / f".downloaded_{dataset_slug.replace('/', '_')}"
        if marker_file.exists():
            print(f"[BİLGİ] {dataset_slug} zaten indirilmiş! Atlanıyor...")
            return True

        print(f"[İNDİRİLİYOR] {dataset_slug} Kaggle Dataset API ile çekiliyor...")
        try:
            api.dataset_download_files(
                dataset_slug,
                path=str(target_dir),
                unzip=True,
                force=True,
                quiet=False
            )
            marker_file.write_text("ok")
            print(f"\n[BAŞARILI] {dataset_slug} başarıyla indirildi ve çıkartıldı!")
            return True
        except Exception as e:
            print(f"\n[HATA] Kaggle Dataset İndirme Başarısız: {e}")
            return False

class FileManager:
    @staticmethod
    def extract_and_delete_zip(zip_path: Path) -> Path:
        extract_path = zip_path.with_suffix('')
        print(f"[ÇIKARTILIYOR] {zip_path.name} arşivden çıkarılıyor...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"[TEMİZLİK] {zip_path.name} siliniyor (Disk tasarrufu)...")
        os.remove(zip_path) 
        return extract_path

    @staticmethod
    def cleanup_raw_folder(folder_path: Path):
        print(f"[TEMİZLİK] İşlenmiş ham videolar siliniyor: {folder_path}")
        shutil.rmtree(folder_path, ignore_errors=True)

class FaceExtractor:
    def __init__(self):
        if _HAS_MTCNN:
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
                print(f"[SİSTEM] MTCNN Yüz Kırpıcı Donanımı: {torch.cuda.get_device_name(0)} (cuda:0)")
            else:
                self.device = torch.device('cpu')
                print(f"[UYARI] CUDA bulunamadı. CPU üzerinde çalışıyor.")
                
            self.mtcnn = MTCNN(
                margin=20, keep_all=False, select_largest=True,
                post_process=False, device=self.device,
            )
        else:
            self.device = None
            self.mtcnn = None
            print("[UYARI] MTCNN bulunamadı. OpenCV Haar Cascade kullanılacak.")

    def process_video_to_array(self, video_path: Path):
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0: return None

        step = max(1, total_frames // FRAMES_PER_VIDEO)
        
        valid_frames = []
        for i in range(FRAMES_PER_VIDEO):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
            success, frame = cap.read()
            if success:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                valid_frames.append(frame_rgb)
            else: 
                break
        cap.release()

        if len(valid_frames) == 0: return None

        all_boxes, _ = self.mtcnn.detect(valid_frames)
        cropped_faces = []

        for idx, frame_rgb in enumerate(valid_frames):
            boxes = all_boxes[idx]
            box = boxes[0] if (boxes is not None and len(boxes) > 0) else None

            if box is not None:
                x1, y1, x2, y2 = [int(b) for b in box]
                h, w, _ = frame_rgb.shape
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                face_crop = frame_rgb[y1:y2, x1:x2]
                if face_crop.size != 0 and face_crop.shape[0] > 0 and face_crop.shape[1] > 0:
                    face_resized = cv2.resize(face_crop, (224, 224))
                    cropped_faces.append(face_resized)

        if len(cropped_faces) == 0: return None
        
        while len(cropped_faces) < FRAMES_PER_VIDEO: 
            cropped_faces.append(cropped_faces[-1])
            
        return np.array(cropped_faces, dtype=np.uint8)

class PipelineOrchestrator:
    def __init__(self):
        self.extractor = FaceExtractor()
        self.metadata_path = PROCESSED_DATA_DIR / "metadata.json"
        
    def update_central_metadata(self, new_records):
        metadata = {}
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except Exception as e:
                print(f"[UYARI] Merkezi metadata.json okunamadı, yeniden oluşturuluyor: {e}")
                
        metadata.update(new_records)
        
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4)
        print(f"[METADATA] {len(new_records)} video verisi merkezi metadata.json'a eklendi/güncellendi.")

    def _find_chunk_folders(self, base_dir: Path) -> list:
        chunk_folders = []
        for d in sorted(base_dir.rglob("dfdc_train_part_*")):
            if d.is_dir():
                has_videos = any(d.rglob("*.mp4"))
                if has_videos:
                    chunk_folders.append(d)
        
        filtered = []
        for folder in chunk_folders:
            is_child = False
            for other in chunk_folders:
                if folder != other and str(folder).startswith(str(other) + os.sep):
                    is_child = True
                    break
            if not is_child:
                filtered.append(folder)
        
        return filtered

    def _process_single_video(self, video_path: Path, metadata: dict):
        """Tek bir videoyu işleyen Thread işçisi."""
        if video_path.name not in metadata: 
            return None
            
        label_str = metadata[video_path.name]["label"]
        label_val = 1.0 if label_str == "FAKE" else 0.0
        
        face_sequence = self.extractor.process_video_to_array(video_path)
        if face_sequence is not None:
            video_dest_dir = PROCESSED_DATA_DIR / label_str / video_path.stem
            video_dest_dir.mkdir(parents=True, exist_ok=True)
            
            frame_filenames = []
            for idx, face_frame in enumerate(face_sequence):
                frame_filename = f"frame_{idx}.jpg"
                frame_path = video_dest_dir / frame_filename
                Image.fromarray(face_frame).save(frame_path, quality=95)
                frame_filenames.append(frame_filename)
            
            return video_path.name, {
                "label": label_str,
                "label_val": label_val,
                "folder": f"{label_str}/{video_path.stem}",
                "frames": frame_filenames
            }
        return None

    def _process_chunk_folder(self, chunk_folder: Path) -> bool:
        chunk_name = chunk_folder.name
        
        metadata = {}
        print(f"\n[ARANIYOR] {chunk_name} içinde metadata.json taranıyor...")
        json_files = list(chunk_folder.rglob("*.json"))
        
        for j_file in json_files:
            if "metadata" in j_file.name.lower():
                try:
                    with open(j_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    print(f"[BULUNDU] Metadata başarıyla okundu: {j_file.name}")
                    break
                except Exception as e:
                    print(f"[HATA] Metadata okunamadı: {e}")
                    
        videos = list(chunk_folder.rglob("*.mp4"))
        if not videos:
            print(f"[UYARI] {chunk_name} içinde hiç video bulunamadı.")
            return True
            
        if not metadata:
            print(f"\n[KRİTİK HATA] {chunk_name} içinde metadata.json BULUNAMADI!")
            print(f"-> 100GB'lık veri ÇÖPE GİTMESİN diye silme işlemi iptal edildi.")
            return False 
            
        chunk_metadata_updates = {}
        processed_video_count = 0
        
        # --- PARALEL THREAD HAVUZU KURULUMU ---
        # İşlemcinin 4.76 GHz gücünü kullanmak için aynı anda 6 video işliyoruz.
        # RTX 5070 bu yükü taşırken ekran kartı kullanımı tavan yapacaktır.
        NUM_WORKERS = 6 
        
        print(f"[İŞLENİYOR] {chunk_name} içindeki {len(videos)} video PARALEL olarak ({NUM_WORKERS} Thread) analiz ediliyor...")
        
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            # Tüm videolar için işleri thread havuzuna fırlatıyoruz
            futures = {executor.submit(self._process_single_video, video_path, metadata): video_path for video_path in videos}
            
            # Sonuçlar tamamlandıkça tqdm barda ilerleme görülecek
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Videolar ({chunk_name})"):
                try:
                    result = future.result()
                    if result is not None:
                        video_name, update_info = result
                        chunk_metadata_updates[video_name] = update_info
                        processed_video_count += 1
                except Exception as e:
                    video_file = futures[future]
                    print(f"\n[HATA] {video_file.name} işlenirken beklenmedik hata oluştu: {e}")
        
        if chunk_metadata_updates:
            self.update_central_metadata(chunk_metadata_updates)
            
        print(f"[KAYDEDİLDİ] {processed_video_count} video için yüz dizisi başarıyla çıkarıldı.")
        return True

    def run(self):
        print(f"\n--- DEEPFAKE ARDIŞIK YÜZ GÖRÜNTÜLERİ İŞLEME MOTORU ---\n")
        print("Lütfen bir çalışma modu seçin:")
        print("  [1] Otomatik Mod (Kaggle'dan 500GB İndir, İşle ve Temizle)")
        print("  [2] Manuel Mod (Bilgisayarda Zaten İndirilmiş Olan Verileri İşle)")
        
        secim = input("\nSeçiminiz (1 veya 2): ").strip()
        
        if secim == "2":
            yerel_yol = input("Lütfen '.mp4' ve 'metadata.json' dosyalarının bulunduğu klasör yolunu girin: ").strip()
            yerel_yol = yerel_yol.replace('"', '').replace("'", "")
            hedef_klasor = Path(yerel_yol)
            
            if not hedef_klasor.exists():
                print(f"\n[HATA] Belirtilen yol bulunamadı: {hedef_klasor}")
                return
                
            print(f"\n[BİLGİ] Yerel klasör taranıyor: {hedef_klasor}")
            chunk_folders = self._find_chunk_folders(hedef_klasor)
            
            if not chunk_folders:
                chunk_folders = [hedef_klasor]
                
            for chunk_folder in chunk_folders:
                self._process_chunk_folder(chunk_folder)
                print(f"[GÜVENLİK] Manuel mod kullanıldığı için '{chunk_folder.name}' klasörü silinmedi, diskte bırakıldı.")
                
            print(f"\n[BİTTİ] Yerel veriler başarıyla yüz tensörlerine dönüştürüldü: {PROCESSED_DATA_DIR}")

        elif secim == "1":
            for ds_info in DFDC_DATASETS:
                slug = ds_info["slug"]
                label = ds_info["label"]
                
                print(f"\n{'='*60}")
                print(f"[DATASET] {label} ({slug})")
                print(f"{'='*60}\n")
                
                success = DatasetDownloader.download_dataset(slug, RAW_DATA_DIR)
                if not success:
                    print(f"[HATA] {slug} indirilemedi, sonraki dataset'e geçiliyor...")
                    continue
                
                chunk_folders = self._find_chunk_folders(RAW_DATA_DIR)
                
                if not chunk_folders:
                    print(f"[UYARI] {slug} içinde işlenecek chunk klasörü bulunamadı!")
                    continue
                
                print(f"[BİLGİ] {len(chunk_folders)} chunk klasörü bulundu:")
                for cf in chunk_folders:
                    print(f"  → {cf.name}")
                
                for chunk_folder in chunk_folders:
                    is_success = self._process_chunk_folder(chunk_folder)
                    if is_success:
                        FileManager.cleanup_raw_folder(chunk_folder)
                    else:
                        print(f"\n[GÜVENLİK PROTOKOLÜ DEVREDE]")
                        print(f"-> {chunk_folder.name} klasörü silinmekten kurtarıldı ve diskte bırakıldı.\n")
            
            print(f"\n[BİTTİ] Tüm dataset işlendi. Sonuçlar: {PROCESSED_DATA_DIR}")
        
        else:
            print("\n[HATA] Geçersiz seçim yaptınız. Lütfen programı yeniden başlatıp 1 veya 2'yi tuşlayın.")

if __name__ == "__main__":
    pipeline = PipelineOrchestrator()
    pipeline.run()