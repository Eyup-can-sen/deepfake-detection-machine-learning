import os
import sys
import torch
from pathlib import Path

# Proje ana dizinini yollara ekliyoruz
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.model import DeepfakeDetector
# Canlı videodan yüz çıkarmak için güçlü FaceExtractor'ı çağırıyoruz
from src.utils.data_pipeline import FaceExtractor
# KESİN KURAL: Eğitimde kullanılan transformların aynısını kullanmalıyız
from src.data_loader import get_transforms 

def predict_video(video_path, model_path="models/deepfake_model_best.pth"): # best.pth olarak güncellendi
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(model_path):
        print(f"HATA: Eğitilmiş model bulunamadı ({model_path}).")
        return
        
    if not os.path.exists(video_path):
        print(f"HATA: Video bulunamadı ({video_path}).")
        return

    print("Yapay Zeka Yükleniyor...")
    model = DeepfakeDetector(sequence_length=5).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() 

    # Sadece Validation (Doğrulama) transformunu alıyoruz
    _, val_transform = get_transforms()

    print(f"\n[{Path(video_path).name}] Analiz ediliyor...")
    
    # Yüz çıkarıcıyı başlat ve videoyu Numpy matrisine (5 kare) çevir
    extractor = FaceExtractor()
    face_sequence = extractor.process_video_to_array(Path(video_path))
    
    if face_sequence is None or len(face_sequence) == 0:
        print("Videoda yüz bulunamadı veya işlenemedi!")
        return

    # Albumentations mantığına göre Numpy matrisini Tensöre çeviriyoruz
    face_tensors = []
    for face in face_sequence:
        augmented = val_transform(image=face)
        face_tensors.append(augmented['image'])
    
    # Modele uygun boyuta (1, 5, 3, 224, 224) getir
    video_tensor = torch.stack(face_tensors).unsqueeze(0).to(device)

    print("Spatio-Temporal sinir ağlarından geçiriliyor...")
    with torch.no_grad():
        # Çıkarım (Inference) aşamasında da Yarı Hassasiyet (AMP) kullanıyoruz
        with torch.amp.autocast('cuda'):
            # Model çıktısını tek boyuta düşürüyoruz (view(-1))
            output = model(video_tensor).view(-1)
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
    # Test etmek istediğin ham .mp4 videosunun yolunu buraya yaz
    ornek_video = r"D:\Deepfake_Data\train_sample_videos\aagfhgtpmv.mp4" 
    predict_video(ornek_video)