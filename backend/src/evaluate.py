import torch
from torchvision import transforms
import numpy as np
import os

# Yazdığımız modülleri çağırıyoruz
from model import DeepfakeDetector
from data_loader import extract_and_crop_faces

def predict_video(video_path, model_path="models/deepfake_model_test.pth"):
    # Cihaz seçimi
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(model_path):
        print(f"HATA: Eğitilmiş model bulunamadı ({model_path}). Lütfen önce train.py dosyasını çalıştırın.")
        return
        
    if not os.path.exists(video_path):
        print(f"HATA: Video bulunamadı ({video_path}).")
        return

    print("Yapay Zeka Yükleniyor...")
    # Modeli iskelet olarak çağır ve eğitilmiş ağırlıkları içine doldur
    model = DeepfakeDetector(sequence_length=5).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # Çok Kritik: Modeli eğitim (train) modundan test (eval) moduna alıyoruz. 
    # (Bu sayede Dropout gibi sadece eğitimde çalışan katmanlar kapanır ve tam performans tahmin yapılır)
    model.eval() 

    # Görüntüleri eğitimdeki ile BİREBİR aynı formatta hazırlamalıyız
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"\n[{os.path.basename(video_path)}] Analiz ediliyor...")
    faces = extract_and_crop_faces(video_path, num_frames=5)
    
    if len(faces) == 0:
        print("Videoda yüz bulunamadı!")
        return
        
    # Yüz eksiği varsa son kareyi kopyalayarak 5'e tamamla
    while len(faces) < 5:
        faces.append(faces[-1])

    # Yüzleri tensöre çevir ve modele uygun boyuta (1, 5, 3, 224, 224) getir
    face_tensors = [transform(face) for face in faces]
    video_tensor = torch.stack(face_tensors).unsqueeze(0).to(device)

    print("Sinir ağlarından geçiriliyor...")
    # no_grad(): Test aşamasında olduğumuz için geriye dönük hesaplama (backward) yapmıyoruz, bellek tasarrufu!
    with torch.no_grad():
        output = model(video_tensor)
        # Sigmoid: Çıkan sonucu 0 (Gerçek) ile 1 (Sahte) arasında bir ihtimale (%) dönüştürür
        probability = torch.sigmoid(output).item() 

    print("\n" + "=" * 40)
    if probability > 0.5:
        print(f" YAPAY ZEKA KARARI: FAKE (SAHTE)")
        print(f" Güven Oranı: %{probability * 100:.2f}")
    else:
        print(f" YAPAY ZEKA KARARI: REAL (GERÇEK)")
        print(f" Güven Oranı: %{(1 - probability) * 100:.2f}")
    print("=" * 40 + "\n")

if __name__ == "__main__":
    # Test etmek istediğin videonun yolunu buraya yazıyorsun
    # Önceki klasöründeki FAKE veya REAL bir videonun adını kopyalayabilirsin
    ornek_video = r"D:\Deepfake_Data\train_sample_videos\aagfhgtpmv.mp4" 
    predict_video(ornek_video)