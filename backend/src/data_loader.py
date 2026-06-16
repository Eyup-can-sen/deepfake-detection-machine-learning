import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
# Görüntü zenginleştirme (Data Augmentation) için Albumentations kütüphanesi
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import json

# HDF5 formatını okumak için h5py kütüphanesini içe aktarmayı dener
try:
    import h5py
    _HAS_H5PY = True
except ImportError:
    h5py = None
    _HAS_H5PY = False

class FastDeepfakeDataset(Dataset):
    """
    Derin öğrenme modeli için verileri hazırlayan, PyTorch uyumlu Dataset sınıfı.
    Farklı veri saklama formatlarını (Klasör, HDF5, NPZ) otomatik algılayıp işleyebilir.
    """
    def __init__(self, data_path: str, transform=None):
        """
        data_path: .h5 dosyasının yolu, .npz dosyalarının bulunduğu klasörün yolu 
                   veya metadata.json içeren işlenmiş yüz klasörünün (PROCESSED_DATA_DIR) yolu.
        transform: Görüntülere uygulanacak dönüştürme/zenginleştirme (augmentation) işlemleri.
        """
        self.data_path = Path(data_path)
        self.transform = transform
        
        # Verilen yolun dosya mı yoksa klasör mü olduğunu ve uzantısını kontrol et
        self.is_h5 = self.data_path.is_file() and self.data_path.suffix == '.h5'
        self.is_dir = self.data_path.is_dir()
        
        self.is_metadata_dir = False
        self.is_npz_dir = False
        
        if self.is_dir:
            # Klasör içinde metadata.json varsa, resimlerin JPG olarak klasörlerde tutulduğu moddur
            if (self.data_path / "metadata.json").exists():
                self.is_metadata_dir = True
            else:
                # metadata.json yoksa, numpy parçaları (.npz) olarak kaydedilmiş moddur
                self.is_npz_dir = True
                
        # Hiçbir geçerli format bulunamazsa hata fırlat
        if not (self.is_h5 or self.is_metadata_dir or self.is_npz_dir):
            raise ValueError("Geçersiz veri yolu! .h5 dosyası, metadata.json içeren klasör veya .npz klasörü olmalı.")

        # --- ARDIŞIK JPG KLASÖR MODU ---
        if self.is_metadata_dir:
            # JSON dosyasını oku ve videolara ait etiket ve dosya yolu bilgilerini al
            with open(self.data_path / "metadata.json", 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            self.video_keys = list(self.metadata.keys())
            self.length = len(self.video_keys)
            print(f"[BİLGİ] Sıralı Klasör Veri Seti Yüklendi. Toplam Örnek: {self.length}")

        # --- HDF5 (.h5) OKUMA MODU ---
        elif self.is_h5:
            if not _HAS_H5PY:
                raise ImportError("HDF5 (.h5) dosyalarını okumak için 'h5py' kütüphanesi kurulu olmalıdır.")
            # HDF5 dosyasını 'r' (read-only) modunda aç. 
            # Verilerin tamamı RAM'e yüklenmez, sadece endeksleri (başlıkları) okunur.
            self.h5_file = h5py.File(self.data_path, 'r')
            self.length = self.h5_file['y'].shape[0]
            print(f"[BİLGİ] HDF5 Veri Seti Yüklendi. Toplam Örnek: {self.length}")
            
        # --- NPZ PARÇALARI OKUMA MODU ---
        elif self.is_npz_dir:
            self.npz_files = list(self.data_path.glob("*.npz"))
            self.data_map = [] # Hangi verinin hangi .npz dosyasında ve kaçıncı sırada olduğunu tutar
            
            print(f"[BİLGİ] NPZ Klasörü Taranıyor... ({len(self.npz_files)} dosya bulundu)")
            for f_idx, f_path in enumerate(self.npz_files):
                # mmap_mode='r' veriyi diskin üzerinden RAM'miş gibi okur (Bellek taşmasını önler)
                with np.load(f_path, mmap_mode='r') as data:
                    num_items = data['y'].shape[0]
                    for i in range(num_items):
                        # Her bir video parçasının adresini (dosya indeksi, veri indeksi) kaydet
                        self.data_map.append((f_idx, i))
                        
            self.length = len(self.data_map)
            print(f"[BİLGİ] NPZ Veri Seti Yüklendi. Toplam Örnek: {self.length}")

    def __len__(self):
        """PyTorch'un veri setinin toplam boyutunu bilmesi için gereken zorunlu metot."""
        return self.length

    def __getitem__(self, idx):
        """
        PyTorch eğitim döngüsü sırasında her bir adımı (batch) oluşturmak için bu metodu çağırır.
        Verilen index (idx) değerine karşılık gelen videonun karelerini ve etiketini döndürür.
        """
        # 1. HAM VERİYİ HANGİ FORMATTAYSA ONA GÖRE ÇEK
        if self.is_metadata_dir:
            video_key = self.video_keys[idx]
            info = self.metadata[video_key]
            
            label = info["label_val"]
            folder_path = self.data_path / info["folder"]
            frame_files = info["frames"]
            
            video_frames = []
            # Klasördeki JPG'leri sırayla oku
            for f_name in frame_files:
                img_path = folder_path / f_name
                img = cv2.imread(str(img_path))
                if img is None:
                    # Fotoğraf bozuksa veya yoksa siyah (sıfır) bir resim oluştur
                    img_rgb = np.zeros((224, 224, 3), dtype=np.uint8)
                else:
                    # BGR'den modelin beklediği RGB formatına çevir
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                video_frames.append(img_rgb)
                
            # Padding: Eğer 5 kareden az okunabildiyse, 5'e tamamlayana kadar son kareyi kopyala (veya siyah ekle)
            while len(video_frames) < 5:
                if len(video_frames) > 0:
                    video_frames.append(video_frames[-1])
                else:
                    video_tensor = np.zeros((224, 224, 3), dtype=np.uint8)
                    video_frames.append(video_tensor)
                    
        elif self.is_h5:
            # Diskten sadece istenen index'teki videonun 5 karesini anlık olarak RAM'e al (Çok hızlıdır)
            video_frames = self.h5_file['X'][idx] 
            label = self.h5_file['y'][idx]
            
        else:
            # Haritadan verinin hangi .npz dosyasında olduğunu bul ve sadece o kısmını oku
            f_idx, item_idx = self.data_map[idx]
            with np.load(self.npz_files[f_idx]) as data:
                video_frames = data['X'][item_idx]
                label = data['y'][item_idx]

        # 2. VERİ ARTIRIMI (AUGMENTATION) VE PYTORCH FORMATINA ÇEVİRME
        processed_frames = []
        for frame in video_frames: # frame boyutu: (224, 224, 3) (Yükseklik, Genişlik, Renk Kanalı)
            if self.transform:
                # Albumentations ile fotoğrafı döndür, rengiyle oyna vs.
                augmented = self.transform(image=frame)
                frame_tensor = augmented['image']
            else:
                # Transform yoksa bile Numpy matrisini PyTorch Tensörüne çevirmeliyiz.
                # ÖNEMLİ: Görüntü işleme kütüphaneleri (H, W, C) kullanırken, PyTorch (C, H, W) ister.
                # permute(2, 0, 1) işlemi ile Renk kanalını başa alıyoruz.
                frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            
            processed_frames.append(frame_tensor)

        # 5 ayrı kareyi tek bir Tensör paketi haline getir -> Yeni Boyut: (5, 3, 224, 224)
        video_tensor = torch.stack(processed_frames)
        label_tensor = torch.tensor([label], dtype=torch.float32)

        return video_tensor, label_tensor

# Modelin genelleme yeteneğini artırmak için verileri rastgele bozan/değiştiren fonksiyonlar
def get_transforms():
    # Eğitim verisi için zorlaştırmalar (Data Augmentation)
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5), # %50 ihtimalle fotoğrafı yatay çevir
        A.RandomBrightnessContrast(p=0.2), # %20 ihtimalle parlaklık/kontrast değiştir
        # Deepfake tespitini zorlaştırmak için rastgele JPEG sıkıştırma (blur) uygula
        A.ImageCompression(quality_range=(60, 100), p=0.3),
        # Görüntü renk değerlerini (RGB) ImageNet veri setinin ortalamasına göre standartlaştır (Modellerin daha hızlı öğrenmesini sağlar)
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(), # PyTorch formatına çevir
    ])
    
    # Validasyon (Test) verisi için zorlaştırma YAPILMAZ! Sadece standartlaştırma yapılır.
    val_transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return train_transform, val_transform