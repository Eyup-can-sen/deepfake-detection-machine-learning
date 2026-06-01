import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm  # İlerleme çubuğu kütüphanesi eklendi

# Proje kütüphanelerini çağırıyoruz
from data_loader import FastDeepfakeDataset, get_transforms
from model import DeepfakeDetector
from config import PROCESSED_DATA_DIR

def set_seed(seed=42):
    """Modelin her çalıştığında aynı veri ayrımını ve başlangıç ağırlıklarını kullanmasını sağlar."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model():
    set_seed(42)
    print("Yüksek Hızlı Veri Seti Hazırlanıyor...")
    
    # Veri yolunu belirle
    metadata_json_path = Path(PROCESSED_DATA_DIR) / "metadata.json"
    h5_path = Path(PROCESSED_DATA_DIR) / "deepfake_dataset.h5"
    
    if metadata_json_path.exists():
        data_path = PROCESSED_DATA_DIR
        print(f"[BİLGİ] Veri yükleyici sıralı klasör modunda çalışacak: {data_path}")
    elif h5_path.exists():
        data_path = h5_path
        print(f"[BİLGİ] Veri yükleyici HDF5 dosya modunda çalışacak: {data_path}")
    else:
        print(f"[HATA] İşlenmiş veri bulunamadı! {PROCESSED_DATA_DIR} altında metadata.json veya {h5_path} olmalı.")
        return

    # Veri artırımı (Augmentation) fonksiyonlarını al
    train_transform, val_transform = get_transforms()
    
    # 1. Aynı veri setini İKİ KERE, farklı transformasyonlarla başlat
    full_dataset_train = FastDeepfakeDataset(data_path=str(data_path), transform=train_transform)
    full_dataset_val = FastDeepfakeDataset(data_path=str(data_path), transform=val_transform)
    
    # --- EĞİTİM VE DOĞRULAMA (VAL) OLARAK BÖLME ---
    total_size = len(full_dataset_train)
    indices = torch.randperm(total_size).tolist()
    val_size = int(0.2 * total_size) # %20 Validation
    train_size = total_size - val_size # %80 Train
    
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    
    train_dataset = Subset(full_dataset_train, train_indices)
    val_dataset = Subset(full_dataset_val, val_indices)
    
    print(f"[BİLGİ] Veri Seti Bölündü: {train_size} Eğitim Örneği | {val_size} Doğrulama Örneği")

    # RTX 5070 için optimize edilmiş DataLoader'lar
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, 
        num_workers=4, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, 
        num_workers=4, pin_memory=True
    )
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Eğitim şu cihazda yapılacak: {device}")
    
    model = DeepfakeDetector(sequence_length=5).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # --- RTX 5070 İÇİN OTOMATİK KARIŞIK HASSASİYET (AMP) ---
    scaler = torch.amp.GradScaler('cuda')
    
    # --- LEARNING RATE SCHEDULER VE EARLY STOPPING ---
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    epochs = 50 
    early_stopping_patience = 7
    epochs_no_improve = 0
    best_val_loss = float('inf')
    
    # Model kaydetme dizini
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    best_model_path = model_dir / "deepfake_model_best.pth"
    
    metrics_history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "train_precision": [], "val_precision": [],
        "train_recall": [], "val_recall": [],
        "train_f1": [], "val_f1": []
    }
    
    print("\nModel Eğitimi Başlıyor...")
    for epoch in range(epochs):
        
        # ================== TRAINING PHASE ==================
        model.train() 
        train_loss = 0
        all_train_labels = []
        all_train_preds = []
        
        # TQDM İlerleyiş Çubuğu Eklendi (Eğitim)
        train_loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] Eğitim", leave=False, dynamic_ncols=True)
        
        for batch_idx, (videos, labels) in enumerate(train_loop):
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float().view(-1)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outputs = model(videos).view(-1)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            predictions = (torch.sigmoid(outputs) >= 0.5).int()
            all_train_labels.extend(labels.cpu().numpy())
            all_train_preds.extend(predictions.cpu().numpy())
            
            # İlerleme çubuğunun yanına anlık Loss değerini yazdır
            train_loop.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)
        train_acc = accuracy_score(all_train_labels, all_train_preds)
        train_prec = precision_score(all_train_labels, all_train_preds, zero_division=0)
        train_rec = recall_score(all_train_labels, all_train_preds, zero_division=0)
        train_f1 = f1_score(all_train_labels, all_train_preds, zero_division=0)

        # ================== VALIDATION PHASE ==================
        model.eval()
        val_loss = 0
        all_val_labels = []
        all_val_preds = []
        
        # TQDM İlerleyiş Çubuğu Eklendi (Doğrulama)
        val_loop = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{epochs}] Doğrulama", leave=False, dynamic_ncols=True)
        
        with torch.no_grad():
            for videos, labels in val_loop:
                videos = videos.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).float().view(-1)
                
                with torch.amp.autocast('cuda'):
                    outputs = model(videos).view(-1)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                predictions = (torch.sigmoid(outputs) >= 0.5).int()
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(predictions.cpu().numpy())
                
                val_loop.set_postfix(loss=loss.item())
                
        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(all_val_labels, all_val_preds)
        val_prec = precision_score(all_val_labels, all_val_preds, zero_division=0)
        val_rec = recall_score(all_val_labels, all_val_preds, zero_division=0)
        val_f1 = f1_score(all_val_labels, all_val_preds, zero_division=0)
        
        # Metrikleri Kaydet
        metrics_history["train_loss"].append(avg_train_loss)
        metrics_history["val_loss"].append(avg_val_loss)
        metrics_history["train_acc"].append(train_acc)
        metrics_history["val_acc"].append(val_acc)
        metrics_history["train_precision"].append(train_prec)
        metrics_history["val_precision"].append(val_prec)
        metrics_history["train_recall"].append(train_rec)
        metrics_history["val_recall"].append(val_rec)
        metrics_history["train_f1"].append(train_f1)
        metrics_history["val_f1"].append(val_f1)

        # Epoch özeti (Çubuklar kaybolduktan sonra temiz bir şekilde yazılır)
        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"Train Loss: {avg_train_loss:.4f} Acc: %{train_acc*100:.2f} F1: {train_f1:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} Acc: %{val_acc*100:.2f} F1: {val_f1:.4f}")

        # LR Scheduler Adımı
        scheduler.step(avg_val_loss)

        # ================== EARLY STOPPING KONTROLÜ ==================
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # En iyi modeli kaydet
            torch.save(model.state_dict(), best_model_path)
            print(f"[*] Yeni en düşük Val Loss! Model '{best_model_path}' olarak kaydedildi.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print(f"\n[!] Erken Durdurma Tetiklendi! {early_stopping_patience} epoch boyunca Val Loss iyileşmedi.")
                break

    print("\nEğitim Süreci Tamamlandı!")
    
    # Front-end / Analiz için JSON kaydı
    with open(model_dir / "training_metrics.json", "w") as f:
        json.dump(metrics_history, f, indent=4)
    print("Eğitim metrikleri (JSON) kaydedildi.")

    # ================== GRAFİKLERİ ÇİZ VE KAYDET ==================
    plt.figure(figsize=(16, 10))
    
    # 1. Loss Grafiği
    plt.subplot(2, 2, 1)
    plt.plot(metrics_history["train_loss"], label='Train Loss', color='blue')
    plt.plot(metrics_history["val_loss"], label='Validation Loss', color='red')
    plt.title('Loss Eğrisi')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 2. Accuracy Grafiği
    plt.subplot(2, 2, 2)
    plt.plot(metrics_history["train_acc"], label='Train Accuracy', color='blue')
    plt.plot(metrics_history["val_acc"], label='Validation Accuracy', color='red')
    plt.title('Accuracy Eğrisi')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # 3. F1 Score Grafiği
    plt.subplot(2, 2, 3)
    plt.plot(metrics_history["train_f1"], label='Train F1 Score', color='blue')
    plt.plot(metrics_history["val_f1"], label='Validation F1 Score', color='red')
    plt.title('F1 Score Eğrisi')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    
    # 4. Precision & Recall (Sadece Validation)
    plt.subplot(2, 2, 4)
    plt.plot(metrics_history["val_precision"], label='Val Precision', color='green')
    plt.plot(metrics_history["val_recall"], label='Val Recall', color='purple')
    plt.title('Validation Precision vs Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plot_path = model_dir / "training_plots.png"
    plt.savefig(plot_path)
    print(f"Eğitim grafikleri '{plot_path}' olarak kaydedildi.")

if __name__ == "__main__":
    train_model()