import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

class FastDeepfakeDataset(Dataset):
    def __init__(self, data_path: str, transform=None):
        """
        data_path: .h5 dosyasının yolu veya .npz dosyalarının bulunduğu klasörün yolu.
        """
        self.data_path = Path(data_path)
        self.transform = transform
        
        # Dosyanın tipini otomatik algıla
        self.is_h5 = self.data_path.is_file() and self.data_path.suffix == '.h5'
        self.is_npz_dir = self.data_path.is_dir()
        
        if not (self.is_h5 or self.is_npz_dir):
            raise ValueError("Geçersiz veri yolu! .h5 dosyası veya .npz klasörü olmalı.")

        # --- HDF5 (.h5) OKUMA MODU ---
        if self.is_h5:
            # Sadece dosyanın başlıklarını okuyoruz (RAM'e yüklemiyoruz, okuma anında çekilecek)
            self.h5_file = h5py.File(self.data_path, 'r')
            self.length = self.h5_file['y'].shape[0]
            print(f"[BİLGİ] HDF5 Veri Seti Yüklendi. Toplam Örnek: {self.length}")
            
        # --- NPZ PARÇALARI OKUMA MODU ---
        elif self.is_npz_dir:
            self.npz_files = list(self.data_path.glob("*.npz"))
            self.data_map = [] # Hangi index'in hangi dosyada kaçıncı sırada olduğunu tutar
            
            print(f"[BİLGİ] NPZ Klasörü Taranıyor... ({len(self.npz_files)} dosya bulundu)")
            for f_idx, f_path in enumerate(self.npz_files):
                # Sadece metadata'yı okumak için mmap_mode kullanıyoruz
                with np.load(f_path, mmap_mode='r') as data:
                    num_items = data['y'].shape[0]
                    for i in range(num_items):
                        self.data_map.append((f_idx, i))
                        
            self.length = len(self.data_map)
            print(f"[BİLGİ] NPZ Veri Seti Yüklendi. Toplam Örnek: {self.length}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 1. Ham Veriyi Çek
        if self.is_h5:
            # Diskten sadece istenen index'teki videonun 5 karesini ışık hızında RAM'e al
            video_frames = self.h5_file['X'][idx] 
            label = self.h5_file['y'][idx]
        else:
            f_idx, item_idx = self.data_map[idx]
            with np.load(self.npz_files[f_idx]) as data:
                video_frames = data['X'][item_idx]
                label = data['y'][item_idx]

        # 2. Veri Artırımı (Augmentation) ve PyTorch Formatına Çevirme
        processed_frames = []
        for frame in video_frames: # frame boyutu: (224, 224, 3)
            if self.transform:
                augmented = self.transform(image=frame)
                frame_tensor = augmented['image']
            else:
                # Transform yoksa bile Numpy matrisini PyTorch Tensörüne çevirmeliyiz
                # Numpy (H, W, C) formatındadır, PyTorch (C, H, W) ister.
                frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            
            processed_frames.append(frame_tensor)

        # 5 kareyi paketle -> Boyut: (5, 3, 224, 224)
        video_tensor = torch.stack(processed_frames)
        label_tensor = torch.tensor([label], dtype=torch.float32)

        return video_tensor, label_tensor

# Eğitim için zorlaştırmalar (Data Augmentation)
def get_transforms():
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    val_transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return train_transform, val_transform