import streamlit as st
import os
import sys
import torch
import tempfile
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from facenet_pytorch import MTCNN

ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"

sys.path.append(str(BACKEND_DIR))
sys.path.append(str(BACKEND_DIR / "src"))

from src.model import DeepfakeDetector
from src.data_loader import get_transforms

st.set_page_config(page_title="Deepfake Dedektörü", page_icon="🕵️‍♂️", layout="wide")

st.title("🕵️‍♂️ Deepfake Tespit Sistemi")

tab1, tab2, tab3 = st.tabs(["🎥 Canlı Test (Demo)", "📊 Eğitim Metrikleri", "🧠 Veri Seti ve Mimari"])

@st.cache_resource
def load_deepfake_model():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = DeepfakeDetector(sequence_length=5).to(device)
    model_path = BACKEND_DIR / "models" / "deepfake_model_best.pth"
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model, device
    else:
        st.error(f"Kritik Hata: Model bulunamadı! {model_path}")
        return None, device

@st.cache_resource
def load_face_detector():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return MTCNN(margin=20, keep_all=False, post_process=False, device=device)

# ==========================================
# SEKME 1: SÜRÜKLE BIRAK VİDEO TESTİ
# ==========================================
with tab1:
    st.header("Bağımsız Video Analizi (Derin Tarama)")
    st.write("Videonun başından sonuna kadar eşit aralıklarla yüzler çıkarılır ve toplu analiz edilir.")
    
    uploaded_video = st.file_uploader("Bir video sürükleyin veya seçin (.mp4, .mov)", type=['mp4', 'mov'])
    
    if uploaded_video is not None:
        st.video(uploaded_video)
        
        if st.button("🚀 Videoyu Kapsamlı Analiz Et", use_container_width=True):
            model, device = load_deepfake_model()
            mtcnn = load_face_detector()
            
            if model is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(uploaded_video.read())
                    tmp_video_path = Path(tmp_file.name)

                try:
                    with st.spinner('Video başından sonuna kadar taranıyor...'):
                        
                        cap = cv2.VideoCapture(str(tmp_video_path))
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        
                        # Videoyu 15 eşit parçaya bölüyoruz (Videonun her yerinden örnek almak için)
                        num_segments = 15
                        segment_step = max(1, total_frames // num_segments)
                        
                        all_sequences = [] # Modelin işleyeceği tensorler
                        ui_faces = []      # Ekranda gösterilecek resimler
                        
                        progress_bar = st.progress(0)
                        
                        for i in range(num_segments):
                            start_frame = i * segment_step
                            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                            
                            segment_faces = []
                            # O saniyedeki ardışık 5 yüz karesini bulmaya çalış
                            for _ in range(15): 
                                ret, frame = cap.read()
                                if not ret: break
                                
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                boxes, _ = mtcnn.detect(frame_rgb)
                                
                                if boxes is not None:
                                    box = boxes[0]
                                    x1, y1, x2, y2 = [int(max(0, b)) for b in box]
                                    face_np = frame_rgb[y1:y2, x1:x2]
                                    if face_np.size == 0: continue
                                    
                                    face_resized = cv2.resize(face_np, (224, 224))
                                    segment_faces.append(face_resized)
                                    ui_faces.append(face_resized)
                                    
                                    if len(segment_faces) == 5:
                                        break # 5 ardışık yüz bulduysak bu segmenti bitir
                            
                            # Eğer yüz bulunduysa padding yap ve listeye ekle
                            if len(segment_faces) > 0:
                                while len(segment_faces) < 5:
                                    segment_faces.append(segment_faces[-1])
                                
                                _, val_transform = get_transforms()
                                face_tensors = []
                                for face in segment_faces[:5]:
                                    augmented = val_transform(image=face)
                                    face_tensors.append(augmented['image'])
                                
                                all_sequences.append(torch.stack(face_tensors))
                            
                            progress_bar.progress((i + 1) / num_segments)
                            
                        cap.release()
                        progress_bar.empty()

                        if len(all_sequences) == 0:
                            st.warning("⚠️ Bu videoda hiçbir saniyede yüz tespit edilemedi!")
                        else:
                            # İşlenen yüzleri Açılır Kapanır Menüde Göster
                            with st.expander(f"👁️ Modelin İncelediği Yüz Karelerini Göster (Toplam {len(ui_faces)} Kare)"):
                                st.caption("Bu kareler videonun başından, ortasından ve sonundan rastgele alınmıştır.")
                                st.image(ui_faces, width=110)
                            
                            # Tüm segmentleri tek bir Batch (Paket) haline getir -> Shape: (N, 5, 3, 224, 224)
                            batch_tensor = torch.stack(all_sequences).to(device)

                            with torch.no_grad():
                                autocast_device = 'cuda' if torch.cuda.is_available() else 'cpu'
                                with torch.amp.autocast(device_type=autocast_device, dtype=torch.float16):
                                    outputs = model(batch_tensor).view(-1)
                                    probabilities = torch.sigmoid(outputs).cpu().numpy()
                            
                            st.success("🎯 Kapsamlı Analiz Başarıyla Tamamlandı!")
                            
                            # Modelin videonun farklı saniyeleri için verdiği puanların ortalamasını al
                            final_probability = float(np.mean(probabilities))
                            final_probability = max(0.01, min(0.99, final_probability))
                            
                            if final_probability > 0.5:
                                guven_orani = final_probability * 100
                                st.error(f"🚨 YAPAY ZEKA KARARI: SAKINCALI İÇERİK (FAKE / SAHTE VİDEO)")
                                st.metric(label="Manipülasyon Tespit Güveni", value=f"%{guven_orani:.2f}")
                                st.progress(int(guven_orani))
                            else:
                                guven_orani = (1 - final_probability) * 100
                                st.success(f"✅ YAPAY ZEKA KARARI: DOĞAL İÇERİK (REAL / GERÇEK VİDEO)")
                                st.metric(label="Orijinallik Güvencesi", value=f"%{guven_orani:.2f}")
                                st.progress(int(guven_orani))
                                
                except Exception as e:
                    st.error(f"İşlem sırasında bir hata oluştu: {e}")
                finally:
                    if tmp_video_path.exists():
                        try:
                            os.unlink(tmp_video_path)
                        except:
                            pass

# ==========================================
# SEKME 2: EĞİTİM GRAFİKLERİ VE METRİKLER
# ==========================================
with tab2:
    st.header("Model Eğitim Performansı")
    col1, col2 = st.columns([2, 1])
    with col1:
        plot_path = BACKEND_DIR / "models" / "training_plots.png"
        if plot_path.exists():
            image = Image.open(plot_path)
            st.image(image, caption="Loss, Accuracy, F1-Score ve Precision-Recall Eğrileri", use_container_width=True)
        else:
            st.warning("Grafik dosyası bulunamadı.")
    with col2:
        st.subheader("Hiperparametreler")
        st.info("**Learning Rate Scheduler:** ReduceLROnPlateau")
        st.error("**Early Stopping:** 7 Epoch")
        st.success("**Optimizasyon:** torch.amp.autocast (FP16 Yarı Hassasiyet)")

with tab3:
    st.header("Veri Seti ve İşleme Hattı (Data Pipeline)")
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.subheader("Veri Kümesi (Dataset)")
        st.write("- **Kaynak:** DFDC (Deepfake Detection Challenge)")
        st.write("- **Toplam Örnek:** 45.796 Ardışık Yüz Kareleri")
        st.write("- **Eğitim (Train) Seti:** %80 (36.637 Örnek)")
        st.write("- **Doğrulama (Val) Seti:** %20 (9.159 Örnek)")
        st.write("- **Görüntü Boyutları:** 224x224 Piksel (RGB)")
        
    with col_b:
        st.subheader("Yüz Çıkarma ve Artırım (Augmentation)")
        st.write("- **Yüz Tespiti:** MTCNN (Multi-task Cascaded Convolutional Networks)")
        st.write("- **Zaman Serisi:** Her videodan kronolojik 5 ardışık kare (Sequence)")
        st.write("- **Data Augmentation:** Albumentations kütüphanesi kullanıldı.")
        st.write("  - Yatay Çevirme (Horizontal Flip %50)")
        st.write("  - Rastgele Parlaklık/Kontrast")
        st.write("  - **JPEG Sıkıştırma Artefaktları (Compression)** (Sosyal medya simülasyonu için)")
    
    with col_c:
        st.subheader("Yapay Zeka Mimarisi")
        st.write("- **Uzamsal Çıkarım (Spatial):** ResNet (Residual Networks) tabanlı özellik çıkarıcı")
        st.write("- **Zamansal Çıkarım (Temporal):** LSTM (Long Short-Term Memory) ağı (Ardışık karelerdeki mikromimik ve maske kaymalarını tespit etmek için)")
        st.write("- **Aktivasyon:** Sigmoid (İkili sınıflandırma: Fake/Real)")
        st.write("- **Kayıp Fonksiyonu (Loss):** BCEWithLogitsLoss (Binary Cross Entropy)")
        st.write("- **Optimize Edici:** AdamW (Weight Decay ile aşırı öğrenmeyi engeller)")