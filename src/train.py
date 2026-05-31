import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np

# Yazdığımız diğer dosyalardan fonksiyon ve sınıfları çekiyoruz
from data_loader import extract_and_crop_faces
from model import DeepfakeDetector
from config import RAW_DATA_DIR

class DeepfakeDataset(Dataset):
    """
    PyTorch'un verileri sırayla ve düzenli çekmesini sağlayan özel Veri Seti sınıfımız.
    """
    def __init__(self, metadata_path, num_frames=5, limit=None):
        self.num_frames = num_frames
        self.data = []
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        count = 0
        for video_name, info in metadata.items():
            if limit and count >= limit:
                break
            video_path = os.path.join(RAW_DATA_DIR, video_name)
            if os.path.exists(video_path):
                # PyTorch matematiksel çalıştığı için etiketleri sayıya çeviriyoruz: FAKE=1, REAL=0
                label = 1.0 if info["label"] == "FAKE" else 0.0
                self.data.append((video_path, label))
                count += 1
                
        # ResNet'in sevdiği standart ImageNet renk normalizasyonu (Renkleri -1 ile 1 arasına çeker)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path, label = self.data[idx]
        
        # MTCNN ile yüzleri çıkar (Numpy formatında)
        faces = extract_and_crop_faces(video_path, num_frames=self.num_frames)
        
        # Eğer videoda hiç yüz bulunamadıysa (karanlık vs.), sistemi çökertmek yerine siyah kareler veriyoruz
        if len(faces) == 0:
            faces = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(self.num_frames)]
            
        # Eğer videodan 5'ten az yüz çıktıysa, eksikleri tamamlamak için son kareyi kopyala (Padding)
        while len(faces) < self.num_frames:
            faces.append(faces[-1])
            
        # Çıkan yüzleri PyTorch Tensörüne (çok boyutlu matrislere) çevir
        face_tensors = []
        for face in faces:
            face_tensor = self.transform(face) 
            face_tensors.append(face_tensor)
            
        # 5 kareyi tek bir paket haline getir
        video_tensor = torch.stack(face_tensors)
        
        return video_tensor, torch.tensor([label], dtype=torch.float32)

def train_model():
    print("Veri Seti Hazırlanıyor...")
    metadata_path = os.path.join(RAW_DATA_DIR, "metadata.json")
    
    # DİKKAT: Bütün gün sürmemesi ve sistemi test etmek için şimdilik sadece 10 video okuyacağız (limit=10)
    dataset = DeepfakeDataset(metadata_path, num_frames=5, limit=10)
    
    # DataLoader: Verileri modele "batch_size=2" yani ikişerli gruplar halinde verir
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # Cihaz kontrolü
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Eğitim şu cihazda yapılacak: {device}")
    
    # Modeli ve optimizasyon araçlarını başlat
    model = DeepfakeDetector(sequence_length=5).to(device)
    criterion = nn.BCEWithLogitsLoss() # İki sınıflı (1/0) projeler için en iyi hata hesaplayıcı
    optimizer = optim.Adam(model.parameters(), lr=0.0001) # Ağırlıkları güncelleyecek algoritma
    
    epochs = 3 # Aynı 10 videoyu 3 tur (epoch) boyunca modele göstereceğiz
    
    print("\nEğitim Başlıyor...")
    for epoch in range(epochs):
        model.train() 
        epoch_loss = 0
        
        for batch_idx, (videos, labels) in enumerate(dataloader):
            videos = videos.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()         # 1. Önceki turdan kalan hafızayı temizle
            outputs = model(videos)       # 2. Modeli çalıştır ve tahminleri al
            loss = criterion(outputs, labels) # 3. Tahmin ile gerçek etiket arasındaki HATA'yı (Loss) ölç
            loss.backward()               # 4. Hatanın sebeplerini bulmak için geriye doğru git
            optimizer.step()              # 5. Modeli iyileştirmek için nöronları güncelle
            
            epoch_loss += loss.item()
            print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx+1}/{len(dataloader)}] | Anlık Hata: {loss.item():.4f}")
            
        print(f"--- Epoch {epoch+1} Tamamlandı. Ortalama Hata: {epoch_loss/len(dataloader):.4f} ---\n")
        
    print("Eğitim Testi Başarıyla Tamamlandı!")
    
    # İleride ağırlıkları tekrar kullanabilmek için modeli kaydet
    torch.save(model.state_dict(), "models/deepfake_model_test.pth")
    print("Model ağırlıkları 'models/' klasörüne kaydedildi.")

if __name__ == "__main__":
    # Modelleri kaydedeceğimiz klasör yoksa oluştur
    os.makedirs("models", exist_ok=True)
    train_model()