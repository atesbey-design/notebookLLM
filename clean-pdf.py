import PyPDF2
from typing import Optional
import os
import pickle

# PDF dosyasını doğrulayan fonksiyon
def validate_pdf(file_path: str) -> bool:
    if not os.path.exists(file_path):
        print(f"Hata: Dosya belirtilen yolda bulunamadı: {file_path}")
        return False
    if not file_path.lower().endswith('.pdf'):
        print("Hata: Dosya bir PDF değil")
        return False
    return True

# PDF'den metin çıkaran fonksiyon
def extract_text_from_pdf(file_path: str, max_chars: int = 100000) -> Optional[str]:
    if not validate_pdf(file_path):
        return None
    
    try:
        with open(file_path, 'rb') as file:
            # PDF okuyucu nesnesi oluştur
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Toplam sayfa sayısını al
            num_pages = len(pdf_reader.pages)
            print(f"PDF işleniyor, toplam {num_pages} sayfa...")
            
            extracted_text = []
            total_chars = 0
            
            # Tüm sayfalar üzerinde iterasyon yap
            for page_num in range(num_pages):
                # Sayfadan metin çıkar
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                
                # Bu sayfanın metnini eklemek karakter sınırını aşacak mı kontrol et
                if total_chars + len(text) > max_chars:
                    # Sadece sınır kadar metin ekle
                    remaining_chars = max_chars - total_chars
                    extracted_text.append(text[:remaining_chars])
                    print(f"{max_chars} karakter sınırına {page_num + 1}. sayfada ulaşıldı")
                    break
                
                extracted_text.append(text)
                total_chars += len(text)
                print(f"{page_num + 1}/{num_pages} sayfa işlendi")
            
            final_text = '\n'.join(extracted_text)
            print(f"\nÇıkarma tamamlandı! Toplam karakter: {len(final_text)}")
            return final_text
            
    except PyPDF2.PdfReadError:
        print("Hata: Geçersiz veya bozuk PDF dosyası")
        return None
    except Exception as e:
        print(f"Beklenmedik bir hata oluştu: {str(e)}")
        return None

# PDF metadata bilgilerini alan fonksiyon
def get_pdf_metadata(file_path: str) -> Optional[dict]:
    if not validate_pdf(file_path):
        return None
    
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            metadata = {
                'num_pages': len(pdf_reader.pages),
                'metadata': pdf_reader.metadata
            }
            return metadata
    except Exception as e:
        print(f"Metadata çıkarılırken hata oluştu: {str(e)}")
        return None

# PDF dosyasını işleyip metni çıkar
pdf_path = './resources/ai-short.pdf'
print("Metadata çıkarılıyor...")
metadata = get_pdf_metadata(pdf_path)
if metadata:
    print("\nPDF Metadata Bilgileri:")
    print(f"Sayfa sayısı: {metadata['num_pages']}")
    print("Belge bilgileri:")
    for key, value in metadata['metadata'].items():
        print(f"{key}: {value}")

print("\nMetin çıkarılıyor...")
extracted_text = extract_text_from_pdf(pdf_path)

# Çıkarılan metni bir dosyaya kaydet
if extracted_text:
    output_file = 'extracted_text.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(extracted_text)
    print(f"\nÇıkarılan metin {output_file} dosyasına kaydedildi")

    # data.pkl dosyasını oluştur ve çıkarılan metni kaydet
    data = {
        "extracted_text": extracted_text,
        "metadata": get_pdf_metadata(pdf_path)  # Eğer metadata da kaydetmek isterseniz
    }

    with open('./resources/data.pkl', 'wb') as file:
        pickle.dump(data, file)
    print("data.pkl dosyası oluşturuldu ve kaydedildi.")
