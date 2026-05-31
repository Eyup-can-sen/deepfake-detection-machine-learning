import cv2
import os
import json
import torch
from facenet_pytorch import MTCNN
from config import RAW_DATA_DIR

# GPU varsa GPU'yu, yoksa CPU'yu otomatik seç
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Yüz algılama şu cihazda çalışıyor: {device}")

# MTCNN yüz algılama modelini başlat
mtcnn = MTCNN(keep_all=False, device=device)

def extract_and_crop_faces(video_path, num_frames=5):
    """Videodan yüzleri MTCNN ile bulup kırpar."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        return []

    step = max(1, total_frames // num_frames)
    cropped_faces = []

    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)
        success, frame = cap.read()
        
        if success:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Yüz tespiti yap
            boxes, _ = mtcnn.detect(frame_rgb)
            
            if boxes is not None:
                # İlk tespit edilen yüzün koordinatlarını al
                box = boxes[0]
                x1, y1, x2, y2 = [int(b) for b in box]
                
                # Sınırların dışına çıkmayı engelle
                h, w, _ = frame_rgb.shape
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                face_crop = frame_rgb[y1:y2, x1:x2]
                
                # Kırpılan alan boş değilse 224x224'e boyutlandır
                if face_crop.size != 0 and face_crop.shape[0] > 0 and face_crop.shape[1] > 0:
                    face_resized = cv2.resize(face_crop, (224, 224))
                    cropped_faces.append(face_resized)
        else:
            break

    cap.release()
    return cropped_faces

def process_dataset(limit=2):
    """
    JSON dosyasını okur, videoları bulur ve yüzleri çıkarır.
    """
    json_path = os.path.join(RAW_DATA_DIR, "metadata.json")
    
    if not os.path.exists(json_path):
        print(f"HATA: {json_path} bulunamadı! Yol: {json_path}")
        return

    with open(json_path, 'r') as f:
        metadata = json.load(f)

    print(f"Toplam {len(metadata)} video etiketi bulundu. İlk {limit} video işleniyor...\n")

    count = 0
    for video_name, info in metadata.items():
        if count >= limit:
            break
            
        label = info["label"] # "FAKE" veya "REAL"
        video_path = os.path.join(RAW_DATA_DIR, video_name)
        
        if os.path.exists(video_path):
            print(f"İşleniyor: {video_name} | Etiket: {label}")
            faces = extract_and_crop_faces(video_path, num_frames=5)
            print(f" -> Çıkarılan yüz sayısı: {len(faces)}\n")
        else:
            print(f"UYARI: {video_name} klasörde bulunamadı.")
            
        count += 1

if __name__ == "__main__":
    process_dataset(limit=2)