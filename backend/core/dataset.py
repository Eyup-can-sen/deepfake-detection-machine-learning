import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        """
        root_dir: 'C:/Users/ramaz/Desktop/Dataset/processed_faces' olmalı
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # REAL = 0, FAKE = 1 olarak etiketliyoruz
        classes = {'REAL': 0, 'FAKE': 1}
        
        for cls_name, cls_label in classes.items():
            cls_dir = self.root_dir / cls_name
            if not cls_dir.exists(): continue
                
            for img_name in os.listdir(cls_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(cls_dir / img_name)
                    self.labels.append(cls_label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = str(self.image_paths[idx])
        # OpenCV ile de okunabilir ama PIL bazen daha güvenlidir
        image = Image.open(img_path).convert("RGB")
        import numpy as np
        image = np.array(image)
        
        label = self.labels[idx]

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        # Label'ı PyTorch tensorüne çevir
        label = torch.tensor(label, dtype=torch.long)
        return image, label

# Eğitim (Train) ve Test (Val) için farklı transformasyonlar
def get_transforms():
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5), # Rastgele sağa/sola çevir
        A.RandomBrightnessContrast(p=0.2), # Işıkla oyna
        A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3), # WhatsApp kalite düşüşü simülasyonu
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # ImageNet standartları
        ToTensorV2(),
    ])
    
    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    return train_transform, val_transform