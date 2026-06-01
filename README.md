# 🕵️‍♂️ Spatio-Temporal Deepfake Tespit Sistemi

Bu proje, yapay zeka (Deep Learning) teknikleri kullanılarak manipüle edilmiş (Deepfake) video içeriklerini yüksek doğrulukla tespit etmek amacıyla geliştirilmiş bir **Yazılım Mühendisliği Bitirme/Araştırma Projesidir.** (Kayseri Üniversitesi).

Sistem, videoları kare kare incelemek yerine, zaman içindeki tutarsızlıkları (mikromimikler, maske kaymaları, renk artefaktları) yakalayabilmek için ardışık kareleri (Sequence) bütünsel olarak analiz eden Spatio-Temporal bir mimari kullanır.

## 🧠 Model Mimarisi ve Veri Boru Hattı (Data Pipeline)

Projenin yapay zeka omurgası şu bileşenlerden oluşmaktadır:

* **Yüz Çıkarma (Face Extraction):** Videolardaki yüzleri tespit etmek ve kırpmak için yüksek hassasiyetli **MTCNN** (Multi-task Cascaded Convolutional Networks) kullanılmıştır.
* **Zaman Serisi Analizi:** Model, videodan tek bir kare yerine, kronolojik sıraya dizilmiş **5 ardışık kareyi** bir sekans olarak (Batch Shape: `N, 5, 3, 224, 224`) işleyerek uzamsal ve zamansal (Spatio-Temporal) manipülasyonları tespit eder.
* **Derin Tarama (Deep Uniform Sampling):** Sistem sadece videonun ilk saniyelerine bakarak karar vermez. Uzun bir videonun (örneğin 7 dakika) başından, ortasından ve sonundan eşit aralıklarla (15 farklı segment) yüz sekansları koparır, bu sekansları toplu bir şekilde (Batch Inference) analiz eder ve **en yüksek şüphe (Max Probability)** oranına göre nihai kararı verir.

## 📊 Veri Seti ve Eğitim Süreci

Model, dünya çapında kabul görmüş **DFDC (Deepfake Detection Challenge)** veri seti kullanılarak eğitilmiştir.

* **Toplam Veri:** 45.796 Ardışık Yüz Karesi
* **Eğitim (Train) Seti:** %80
* **Doğrulama (Val) Seti:** %20
* **Veri Artırımı (Augmentation):** Gerçek dünya senaryolarını simüle etmek için *Albumentations* kütüphanesi ile yatay çevirme, rastgele parlaklık ve **JPEG Compression (Sıkıştırma) artefaktları** uygulanmıştır.

### Hiperparametreler ve Optimizasyon
* **Donanım Hızlandırması:** Eğitim ve çıkarım (inference) süreçleri **NVIDIA GeForce RTX 5070** GPU üzerinde optimize edilmiştir.
* **Mixed Precision (AMP):** PyTorch `autocast` kullanılarak FP16 yarı hassasiyetinde VRAM tasarrufu ve x2 işlem hızı sağlanmıştır.
* **Learning Rate Scheduler:** `ReduceLROnPlateau` kullanılarak, doğrulama hatası (Val Loss) 3 epoch boyunca düşmediğinde öğrenme oranı dinamik olarak küçültülmüştür.
* **Early Stopping:** Modelin veriyi ezberlemesini (Overfitting) engellemek için, Val Loss 7 epoch boyunca iyileşmediğinde eğitim otomatik olarak durdurulmuş ve en iyi ağırlıklar kaydedilmiştir.

---

## 🚀 Kurulum ve Çalıştırma Talimatları

Proje, bağımlılık çakışmalarını önlemek adına tek bir merkezi sanal ortam (`.venv`) üzerinden çalışacak şekilde yapılandırılmıştır.

### 1. Gereksinimleri Yükleme
Sanal ortamı aktif ettikten sonra gerekli kütüphaneleri yüklemek için:
```bash
pip install -r backend/requirements.txt

2. Sanal Ortamı Aktifleştirme (Windows)
Projeyi çalıştırmadan önce terminalde ana dizindeyken sanal ortamı başlatın:

PowerShell
.\.venv\Scripts\Activate.ps1
(Terminal satırının başında (.venv) ibaresini görmelisiniz.)

3. Arayüzü (Demo) Başlatma
Sistemi canlı bir video ile test etmek, eğitim metriklerini ve model karar süreçlerini (şeffaflık menüsünü) görmek için Streamlit arayüzünü çalıştırın:

PowerShell
streamlit run app.py
Tarayıcınızda otomatik olarak açılacak olan paneli kullanarak (varsayılan: http://localhost:8501), bilgisayarınızdaki herhangi bir videoyu sisteme sürükleyip anında analiz edebilirsiniz.