import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class DeepfakeDetectorModel(nn.Module):
    def __init__(self, num_classes=2):
        super(DeepfakeDetectorModel, self).__init__()
        
        # Önceden eğitilmiş (Pretrained) modeli indir (Sadece ilk çalışmada indirir)
        weights = EfficientNet_B0_Weights.DEFAULT
        self.backbone = efficientnet_b0(weights=weights)
        
        # Gelecekte Grad-CAM için özellik haritalarını (feature maps) çekeceğimiz katman
        self.features = self.backbone.features
        
        # Sınıflandırıcı kısmını değiştiriyoruz (Bizde 1000 değil, sadece 2 sınıf var: REAL/FAKE)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        # Normal eğitim sırasında sadece sonucu döndürür
        return self.backbone(x)
        
    def get_features(self, x):
        """Bu fonksiyon ileride XAI (Açıklanabilir YZ) Isı Haritaları için kullanılacak"""
        return self.features(x)

# Modelin RTX 5070'e aktarılması testi
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model {device} üzerinde çalışacak.")
    
    model = DeepfakeDetectorModel().to(device)
    
    # Test için rastgele bir tensor gönder (Batch Size: 8, Kanal: 3, Boyut: 224x224)
    dummy_input = torch.randn(8, 3, 224, 224).to(device)
    output = model(dummy_input)
    
    print(f"Çıktı boyutu: {output.shape} (8 fotoğraf için [Gerçek_Skoru, Sahte_Skoru])")