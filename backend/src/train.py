import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

# Yeni yazdığımız yüksek hızlı veri yükleyiciyi ve modeli çağırıyoruz
from data_loader import FastDeepfakeDataset, get_transforms
from model import DeepfakeDetector
from config import PROCESSED_DATA_DIR

def train_model():
    print("Yüksek Hızlı Veri Seti Hazırlanıyor...")
    
    # HDF5 veri setimizin yolu
    h5_path = Path(PROCESSED_DATA_DIR) / "deepfake_dataset.h5"
    
    if not h5_path.exists():
        print(f"[HATA] {h5_path} bulunamadı! Lütfen önce data_pipeline.py'yi çalıştırın.")
        return

    # Veri artırımı (Augmentation) fonksiyonlarını al
    train_transform, _ = get_transforms()
    
    # HDF5 üzerinden çalışan yeni Dataset sınıfımız
    dataset = FastDeepfakeDataset(data_path=str(h5_path), transform=train_transform)
    
    # DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Eğitim şu cihazda yapılacak: {device}")
    
    model = DeepfakeDetector(sequence_length=5).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    epochs = 5 
    
    # Arayüz (Frontend) grafikleri için metrikleri tutacağımız sözlük
    metrics_history = {
        "loss": [],
        "accuracy": []
    }
    
    print("\nModel Eğitimi Başlıyor...")
    for epoch in range(epochs):
        model.train() 
        epoch_loss = 0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (videos, labels) in enumerate(dataloader):
            videos = videos.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Doğruluk (Accuracy) Hesaplama
            predictions = torch.sigmoid(outputs) >= 0.5
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            print(f"Epoch [{epoch+1}/{epochs}] | Batch [{batch_idx+1}/{len(dataloader)}] | Anlık Hata (Loss): {loss.item():.4f}")
            
        avg_loss = epoch_loss / len(dataloader)
        avg_acc = (correct_predictions / total_samples) * 100
        
        # Grafikler için veriyi kaydet
        metrics_history["loss"].append(round(avg_loss, 4))
        metrics_history["accuracy"].append(round(avg_acc, 2))
        
        print(f"--- Epoch {epoch+1} Tamamlandı | Ortalama Hata: {avg_loss:.4f} | Doğruluk: %{avg_acc:.2f} ---\n")
        
    print("Eğitim Başarıyla Tamamlandı!")
    
    # 1. Modeli Kaydet
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), model_dir / "deepfake_model_test.pth")
    print("Model ağırlıkları 'models/' klasörüne kaydedildi.")
    
    # 2. Frontend Grafikleri için Metrikleri JSON olarak kaydet
    with open("training_metrics.json", "w") as f:
        json.dump(metrics_history, f, indent=4)
    print("Eğitim metrikleri (Grafik Verileri) 'training_metrics.json' olarak kaydedildi.")

if __name__ == "__main__":
    train_model()