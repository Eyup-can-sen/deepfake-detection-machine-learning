import os
import sys
import json
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, log_loss, f1_score, precision_score, recall_score

# Proje ana dizinini yollara ekliyoruz
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from src.model import DeepfakeDetector
from utils.data_pipeline import FaceExtractor
from src.data_loader import get_transforms

def run_local_evaluation(test_dir, model_path, output_csv):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    test_dir_path = Path(test_dir)
    metadata_path = test_dir_path / "metadata.json"
    
    if not os.path.exists(model_path):
        print(f"[HATA] Model bulunamadı: {model_path}")
        return
    if not metadata_path.exists():
        print(f"[HATA] metadata.json bulunamadı: {metadata_path}")
        return

    print(f"[BİLGİ] Cevap anahtarı (metadata.json) okunuyor...")
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    print("Yapay Zeka Yükleniyor...")
    model = DeepfakeDetector(sequence_length=5).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() 

    extractor = FaceExtractor()
    _, test_transform = get_transforms()

    y_true = []
    y_pred_probs = []
    y_pred_labels = []
    detailed_results = []

    print(f"\nToplam {len(metadata)} video analiz edilecek. İşlem başlıyor...")
    test_loop = tqdm(metadata.items(), total=len(metadata), desc="Model Sınavda", dynamic_ncols=True)
    
    for video_filename, info in test_loop:
        video_path = test_dir_path / video_filename
        
        # Gerçek etiketi JSON'dan al (FAKE -> 1, REAL -> 0)
        true_label_str = info.get("label", "FAKE")
        true_label = 1 if true_label_str == "FAKE" else 0
        
        prediction_prob = 0.5 
        
        if video_path.exists():
            face_sequence = extractor.process_video_to_array(video_path)
            
            if face_sequence is not None and len(face_sequence) > 0:
                face_tensors = []
                for face in face_sequence:
                    augmented = test_transform(image=face)
                    face_tensors.append(augmented['image'])
                
                video_tensor = torch.stack(face_tensors).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    with torch.amp.autocast('cuda'):
                        output = model(video_tensor).view(-1)
                        prediction_prob = torch.sigmoid(output).item()
                        # Log-loss stabilitesi için kırpma
                        prediction_prob = max(0.001, min(0.999, prediction_prob))
        else:
            # Video dosyası diskte yoksa es geç
            continue

        pred_label = 1 if prediction_prob > 0.5 else 0
        
        y_true.append(true_label)
        y_pred_probs.append(prediction_prob)
        y_pred_labels.append(pred_label)
        
        detailed_results.append({
            "Video": video_filename,
            "Gercek_Durum": "FAKE" if true_label == 1 else "REAL",
            "Yapay_Zeka_Tahmini": "FAKE" if pred_label == 1 else "REAL",
            "Yapay_Zeka_Guveni": f"%{prediction_prob*100:.2f}",
            "Sonuc": "DOĞRU" if true_label == pred_label else "YANLIŞ"
        })
        
        test_loop.set_postfix(file=video_filename[:8], loss=f"{prediction_prob:.2f}")

    # --- METRİKLERİ HESAPLA ---
    print("\n" + "="*50)
    print(" 📊 TEST KARNESİ (MODEL DEĞERLENDİRMESİ)")
    print("="*50)
    
    if len(y_true) > 0:
        acc = accuracy_score(y_true, y_pred_labels)
        f1 = f1_score(y_true, y_pred_labels, zero_division=0)
        prec = precision_score(y_true, y_pred_labels, zero_division=0)
        rec = recall_score(y_true, y_pred_labels, zero_division=0)
        k_loss = log_loss(y_true, y_pred_probs, labels=[0, 1])
        
        print(f" Toplam Test Edilen Video : {len(y_true)}")
        print(f" 🎯 Doğruluk (Accuracy)  : %{acc*100:.2f}")
        print(f" ⚖️ F1-Score             : {f1:.4f}")
        print(f" 🎯 Kesinlik (Precision) : {prec:.4f}")
        print(f" 🔄 Duyarlılık (Recall)  : {rec:.4f}")
        print(f" 📉 Kaggle Log-Loss Puanı: {k_loss:.4f} (Ne kadar düşük, o kadar iyi)")
    else:
        print("Test edilecek geçerli video bulunamadı!")
    print("="*50)

    # Detaylı sonuçları CSV'ye kaydet
    df = pd.DataFrame(detailed_results)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n[BAŞARILI] Her videonun detaylı analizi '{output_csv}' dosyasına kaydedildi.")


if __name__ == "__main__":
    TEST_DIR = r"C:\Users\ramaz\Desktop\Dataset\test"
    
    # Kendi eğittiğin en iyi model
    MY_MODEL = r"models\deepfake_model_best.pth"
    
    # Sonuçların raporlanacağı CSV dosyası
    OUTPUT_CSV = r"C:\Users\ramaz\Desktop\Dataset\test\lokal_test_raporu.csv"
    
    run_local_evaluation(
        test_dir=TEST_DIR, 
        model_path=MY_MODEL, 
        output_csv=OUTPUT_CSV
    )