import os
import sys
import json
import zipfile
import shutil
from pathlib import Path

# Proje ana dizinini yollara ekliyoruz
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from dotenv import load_dotenv
load_dotenv()

# Yeni Kaggle Token Tabanlı Kimlik Doğrulama (KGAT_ formatı)
KAGGLE_API_TOKEN = os.getenv("KAGGLE_API_TOKEN")

if not KAGGLE_API_TOKEN:
    raise ValueError("KRİTİK HATA: .env dosyasında KAGGLE_API_TOKEN bulunamadı!")

# Kaggle kütüphanesi bu ortam değişkenini veya ~/.kaggle/access_token dosyasını okur
os.environ['KAGGLE_API_TOKEN'] = KAGGLE_API_TOKEN

# Token'ı ~/.kaggle/access_token dosyasına da yaz (yedek olarak)
_kaggle_dir = Path.home() / ".kaggle"
_kaggle_dir.mkdir(parents=True, exist_ok=True)
_access_token_path = _kaggle_dir / "access_token"
_access_token_path.write_text(KAGGLE_API_TOKEN)

# KAGGLE API'YI DOĞRUDAN PYTHON İÇİNDE BAŞLATIYORUZ
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

import cv2
import torch
import numpy as np
from tqdm import tqdm

try:
    import h5py
    _HAS_H5PY = True
except ImportError:
    h5py = None
    _HAS_H5PY = False

try:
    from facenet_pytorch import MTCNN
    _HAS_MTCNN = True
except ImportError:
    MTCNN = None
    _HAS_MTCNN = False

from backend.src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

COMPETITION_NAME = "deepfake-detection-challenge"
FRAMES_PER_VIDEO = 5 

# Kaggle artık DFDC verilerini Dataset olarak sunuyor (eski zip yapısı kaldırıldı)
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

H5_FILE_PATH = PROCESSED_DATA_DIR / "deepfake_dataset.h5"
NPZ_OUTPUT_DIR = PROCESSED_DATA_DIR / "npz_chunks"

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
            # İndirme başarılı işareti bırak
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
                print(f"[SİSTEM] Yüz Çıkarıcı Donanımı: {torch.cuda.get_device_name(0)}")
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

    def process_video_to_array(self, video_path: Path):
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0: return None

        step = max(1, total_frames // FRAMES_PER_VIDEO)
        cropped_faces = []

        for i in range(FRAMES_PER_VIDEO):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
            success, frame = cap.read()
            if success:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                box = _detect_face_box(frame_rgb, self.mtcnn)

                if box is not None:
                    x1, y1, x2, y2 = [int(b) for b in box]
                    h, w, _ = frame_rgb.shape
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    face_crop = frame_rgb[y1:y2, x1:x2]
                    if face_crop.size != 0 and face_crop.shape[0] > 0 and face_crop.shape[1] > 0:
                        face_resized = cv2.resize(face_crop, (224, 224))
                        cropped_faces.append(face_resized)
            else: break

        cap.release()
        if len(cropped_faces) == 0: return None
        while len(cropped_faces) < FRAMES_PER_VIDEO: cropped_faces.append(cropped_faces[-1])
        return np.array(cropped_faces, dtype=np.uint8)

class PipelineOrchestrator:
    def __init__(self):
        self.extractor = FaceExtractor()
        self.init_h5_file()
        
    def init_h5_file(self):
        if not _HAS_H5PY:
            NPZ_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            return

        if not H5_FILE_PATH.exists():
            print("[BİLGİ] Yeni HDF5 Veri Seti dosyası oluşturuluyor...")
            with h5py.File(H5_FILE_PATH, 'w') as hf:
                hf.create_dataset('X', shape=(0, FRAMES_PER_VIDEO, 224, 224, 3), 
                                  maxshape=(None, FRAMES_PER_VIDEO, 224, 224, 3), dtype='uint8')
                hf.create_dataset('y', shape=(0,), maxshape=(None,), dtype='float32')

    def _write_npz_chunk(self, X_batch, y_batch, chunk_name=None):
        if len(X_batch) == 0: return None
        label = Path(chunk_name).stem if chunk_name else "chunk"
        out_path = NPZ_OUTPUT_DIR / f"chunk_{label}.npz"
        np.savez_compressed(out_path, X=np.asarray(X_batch, dtype=np.uint8), y=np.asarray(y_batch, dtype=np.float32))
        return out_path

    def append_to_h5(self, X_batch, y_batch, chunk_name=None):
        if len(X_batch) == 0: return None
        if not _HAS_H5PY: return self._write_npz_chunk(X_batch, y_batch, chunk_name)

        with h5py.File(H5_FILE_PATH, 'a') as hf:
            current_size = hf['X'].shape[0]
            new_size = current_size + len(X_batch)
            hf['X'].resize((new_size, FRAMES_PER_VIDEO, 224, 224, 3))
            hf['X'][current_size:new_size] = X_batch
            hf['y'].resize((new_size,))
            hf['y'][current_size:new_size] = y_batch
        return H5_FILE_PATH

    def _find_chunk_folders(self, base_dir: Path) -> list:
        """İndirilen dataset içindeki dfdc_train_part_XX klasörlerini bulur."""
        chunk_folders = []
        # Dataset yapısı: base_dir/dfdc_train_part_XX/dfdc_train_part_X/*.mp4 + metadata.json
        for d in sorted(base_dir.rglob("dfdc_train_part_*")):
            if d.is_dir():
                # metadata.json varsa veya mp4 dosyaları varsa bu bir chunk klasörü
                has_videos = any(d.rglob("*.mp4"))
                if has_videos:
                    chunk_folders.append(d)
        
        # Tekrarlı alt klasörleri filtrele (üst klasörü tut)
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

    def _process_chunk_folder(self, chunk_folder: Path, output_label: str):
        """Bir chunk klasörünü işler (yüz çıkarma + kaydetme)."""
        chunk_name = chunk_folder.name
        
        # metadata.json'u bul (aynı dizinde veya üst dizinde olabilir)
        metadata = {}
        for meta_candidate in [chunk_folder / "metadata.json", 
                                chunk_folder.parent / "metadata.json"]:
            if meta_candidate.exists():
                with open(meta_candidate, 'r') as f:
                    metadata = json.load(f)
                break
        
        videos = list(chunk_folder.rglob("*.mp4"))
        if not videos:
            print(f"[UYARI] {chunk_name} içinde video bulunamadı, atlanıyor...")
            return
            
        X_chunk, y_chunk, processed_video_count = [], [], 0
        
        print(f"[İŞLENİYOR] {chunk_name} içindeki {len(videos)} video analiz ediliyor...")
        for video_path in tqdm(videos, desc=f"Videolar ({chunk_name})"):
            # metadata varsa etiket al, yoksa atla
            if metadata:
                if video_path.name not in metadata: continue
                label_str = metadata[video_path.name]["label"]
                label_val = 1.0 if label_str == "FAKE" else 0.0
            else:
                print(f"[UYARI] metadata.json bulunamadı, {chunk_name} atlanıyor...")
                return
            
            face_sequence = self.extractor.process_video_to_array(video_path)
            if face_sequence is not None:
                X_chunk.append(face_sequence)
                y_chunk.append(label_val)
                processed_video_count += 1
        
        print(f"[KAYDEDİLİYOR] {processed_video_count} video verisi {output_label} dosyasına yazılıyor...")
        self.append_to_h5(X_chunk, y_chunk, chunk_name)

    def run(self):
        output_label = ".h5" if _HAS_H5PY else ".npz"
        print(f"\n--- DEEPFAKE {output_label} VERI ISLEME MOTORU BASLATILDI ---\n")
        
        for ds_info in DFDC_DATASETS:
            slug = ds_info["slug"]
            label = ds_info["label"]
            
            print(f"\n{'='*60}")
            print(f"[DATASET] {label} ({slug})")
            print(f"{'='*60}\n")
            
            # Dataset'i indir
            success = DatasetDownloader.download_dataset(slug, RAW_DATA_DIR)
            if not success:
                print(f"[HATA] {slug} indirilemedi, sonraki dataset'e geçiliyor...")
                continue
            
            # İndirilen klasördeki chunk'ları bul
            chunk_folders = self._find_chunk_folders(RAW_DATA_DIR)
            
            if not chunk_folders:
                print(f"[UYARI] {slug} içinde işlenecek chunk klasörü bulunamadı!")
                continue
            
            print(f"[BİLGİ] {len(chunk_folders)} chunk klasörü bulundu:")
            for cf in chunk_folders:
                print(f"  → {cf.name}")
            
            # Her chunk'ı sırayla işle
            for chunk_folder in chunk_folders:
                self._process_chunk_folder(chunk_folder, output_label)
                # İşlenen chunk'ı sil (disk tasarrufu)
                FileManager.cleanup_raw_folder(chunk_folder)
        
        if _HAS_H5PY: print("\n[BİTTİ] Bulunan tüm dosyalar başarıyla .h5 formatına sıkıştırıldı.")
        else: print(f"\n[BİTTİ] Bulunan tüm dosyalar .npz parçalarına yazıldı: {NPZ_OUTPUT_DIR}")

if __name__ == "__main__":
    pipeline = PipelineOrchestrator()
    pipeline.run()