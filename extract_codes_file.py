import os
from datetime import datetime

# MERNA projesinde senin yazdığın ana klasörler
MY_STUFF = ['backend', 'frontend', 'src'] 

# Sayımı bozacak kütüphane ve sistem klasörleri
FORBIDDEN_DIRS = {
    'node_modules', 'venv', '.venv', 'env', '.git', 
    '__pycache__', '.next', 'dist', 'build', '.idea', '.vscode'
}

# İçindeki kodlar bize ait olmayan yollar
FORBIDDEN_PATHS = [
    'components/ui',
    'migrations',
    'alembic'
]

OUTPUT_FILENAME = "deepfake_proje_kod_dokumani.txt"

def merna_full_backup_and_detective():
    stats = []
    folder_totals = {}
    total_lines = 0
    
    print(f"--- DEEPFAKE PROJE KOD DÖKÜMÜ Başladı ---")
    print(f"Çıktı dosyası: {OUTPUT_FILENAME}\n")

    with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as report_file:
        # Dosya başına başlık ekle
        report_file.write(f"DEEPFAKE PROJE KOD DÖKÜMÜ\n")
        report_file.write(f"Oluşturulma Tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report_file.write("="*50 + "\n\n")

        for root, dirs, files in os.walk('.'):
            # Yasaklı klasörleri temizle
            dirs[:] = [d for d in dirs if d.lower() not in FORBIDDEN_DIRS]
            
            normalized_root = root.replace(os.sep, '/')
            
            # Yasaklı yolları atla
            if any(fp in normalized_root for fp in FORBIDDEN_PATHS):
                continue

            parts = root.split(os.sep)
            if root != '.':
                if any(p.lower() in FORBIDDEN_DIRS for p in parts):
                    continue

            for file in files:
                ext = os.path.splitext(file)[1].lower()
                # Desteklenen dosya türleri
                if ext in {'.py', '.js', '.ts', '.vue', '.css', '.html', '.tsx'}:
                    path = os.path.join(root, file)
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            line_count = len(content.splitlines())
                            
                            # İstatistikleri topla
                            stats.append((path, line_count))
                            top_dir = parts[1] if len(parts) > 1 else "Root"
                            folder_totals[top_dir] = folder_totals.get(top_dir, 0) + line_count
                            total_lines += line_count

                            # TXT Dosyasına Yazma İşlemi
                            report_file.write(f"\n{'#'*80}\n")
                            report_file.write(f"### DOSYA: {path}\n")
                            report_file.write(f"### SATIR SAYISI: {line_count}\n")
                            report_file.write(f"{'#'*80}\n\n")
                            report_file.write(content)
                            report_file.write("\n\n")
                            
                    except Exception as e:
                        print(f"Hata: {path} okunamadı. ({e})")
                        continue

        # Dosyanın en sonuna özet bilgilerini ekle
        report_file.write("\n\n" + "="*80 + "\n")
        report_file.write("GENEL İSTATİSTİKLER\n")
        report_file.write("="*80 + "\n")
        for folder, total in folder_totals.items():
            report_file.write(f"{folder:>15}: {total} satır\n")
        report_file.write("-" * 30 + "\n")
        report_file.write(f"TOPLAM KOD SATIRI: {total_lines}\n")

    # Konsol çıktısı (Sıralı liste)
    stats.sort(key=lambda x: x[1], reverse=True)
    print(f"{'SATIR':<8} | {'DURUM':<15} | {'DOSYA YOLU'}")
    print("-" * 80)
    for path, count in stats:
        status = "TAMAM" if count > 5 else "!! ŞÜPHELİ !!"
        print(f"{count:<8} | {status:<15} | {path}")

    print(f"\nİşlem tamamlandı! Tüm kodlar '{OUTPUT_FILENAME}' içerisine kaydedildi.")

if __name__ == "__main__":
    merna_full_backup_and_detective()