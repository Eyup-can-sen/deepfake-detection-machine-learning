import torch
import torch.nn as nn
from torchvision import models

class DeepfakeDetector(nn.Module):
    def __init__(self, sequence_length=5, hidden_dim=256, lstm_layers=1):
        super(DeepfakeDetector, self).__init__()
        self.sequence_length = sequence_length
        
        # 1. MEKANSAL ÖZELLİK ÇIKARICI (CNN)
        # ResNet18 kullanıyoruz. Önceden eğitilmiş ağırlıklar eğitimimizi hızlandıracak.
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # ResNet'in son sınıflandırma katmanını (fc) siliyoruz, çünkü amacımız kedi/köpek seçmek değil, 
        # sadece yüzün matematiksel özelliklerini (feature) elde etmek.
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # ResNet18'in özellik çıkış boyutu 512'dir.
        cnn_out_dim = 512 
        
        # 2. ZAMANSAL ANALİZ (LSTM)
        # CNN'den gelen 512 boyutlu özellikleri alıp, 5 karelik zaman serisi olarak işliyoruz.
        self.lstm = nn.LSTM(
            input_size=cnn_out_dim, 
            hidden_size=hidden_dim, 
            num_layers=lstm_layers, 
            batch_first=True
        )
        
        # 3. SON KARAR MEKANİZMASI (Classifier)
        # LSTM'den çıkan sonucu (Gerçek mi / Sahte mi) ihtimaline çeviren son katmanlar.
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3), # Ağın ezberlemesini (overfitting) engellemek için
            nn.Linear(128, 1) # Çıktı tek bir nöron (Sigmoid fonksiyonu ile 0-1 arası olasılık verecek)
        )

    def forward(self, x):
        # x'in beklenen boyutu: (batch_size, sequence_length, channels, height, width)
        # Yani: (Aynı anda işlenen video sayısı, 5 kare, 3 renk, 224 piksel, 224 piksel)
        batch_size, seq_length, c, h, w = x.size()
        
        # 5 kareyi tek tek CNN'e sokabilmek için boyutları birleştiriyoruz
        x = x.view(batch_size * seq_length, c, h, w)
        
        # CNN'den geçir (ResNet boyutları küçültüp özellikleri yoğunlaştırır)
        features = self.feature_extractor(x)
        
        # Çıktıyı düzleştir (Flatten) -> Boyut: (batch_size * 5, 512)
        features = features.view(features.size(0), -1)
        
        # LSTM'e sokabilmek için zaman (sequence) boyutunu geri getiriyoruz
        # Yeni boyut: (batch_size, 5, 512)
        features = features.view(batch_size, seq_length, -1)
        
        # LSTM'den geçir
        lstm_out, _ = self.lstm(features)
        
        # Videonun tamamına bakıp karar vermek için LSTM'in ürettiği "en son" zaman adımının çıktısını alıyoruz
        last_time_step_out = lstm_out[:, -1, :] 
        
        # Son sınıflandırıcıdan geçir
        out = self.classifier(last_time_step_out)
        
        return out

if __name__ == "__main__":
    # Kodun hatasız derlenip derlenmediğini test edelim
    print("Yapay Zeka Mimarisi İnşa Ediliyor...")
    model = DeepfakeDetector(sequence_length=5)
    
    # Sisteme gerçek videolar yerine "rastgele sayılardan oluşan" 2 adet sahte video verelim: 
    # Boyut yapısı -> (2 video, 5 kare, 3 RGB kanalı, 224x224 boyut)
    dummy_input = torch.randn(2, 5, 3, 224, 224) 
    
    print(f"Modele giren test verisi boyutu: {dummy_input.shape}")
    
    # Veriyi sinir ağımızın içinden geçirelim
    output = model(dummy_input)
    
    print(f"Model çıktısı boyutu: {output.shape}")
    print(f"Örnek Karar Çıktıları (Henüz eğitilmediği için tamamen rastgele sayılar): \n{output.detach().numpy()}")
    print("\nTEBRİKLER! CNN + LSTM mimarisi kusursuz bir şekilde ayağa kalktı.")