import PyPDF2
from typing import Optional
import os
import torch
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from tqdm.notebook import tqdm
import warnings

warnings.filterwarnings('ignore')

# Dosya yolları ve varsayılan modeli tanımla
pdf_path = './resources/ai-short.pdf'
DEFAULT_MODEL = "meta-llama/Llama-3.2-1B"

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

# Metni kelime sınırlarına göre parçalara ayıran fonksiyon
def create_word_bounded_chunks(text, target_chunk_size):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        word_length = len(word) + 1  # +1 boşluk için
        if current_length + word_length > target_chunk_size and current_chunk:
            # Mevcut parçayı birleştir ve parçalar listesine ekle
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length
    
    # Son parçayı ekle (varsa)
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

# Öncelikle metadata çıkar
print("Metadata çıkarılıyor...")
metadata = get_pdf_metadata(pdf_path)
if metadata:
    print("\nPDF Metadata Bilgileri:")
    print(f"Sayfa sayısı: {metadata['num_pages']}")
    print("Belge bilgileri:")
    for key, value in metadata['metadata'].items():
        print(f"{key}: {value}")

# Metni çıkar
print("\nMetin çıkarılıyor...")
extracted_text = extract_text_from_pdf(pdf_path)

# Çıkarılan metnin ilk 500 karakterini önizleme olarak göster
if extracted_text:
    print("\nÇıkarılan metnin önizlemesi (ilk 500 karakter):")
    print("-" * 50)
    print(extracted_text[:500])
    print("-" * 50)
    print(f"\nToplam çıkarılan karakter sayısı: {len(extracted_text)}")

# İsteğe bağlı: Çıkarılan metni bir dosyaya kaydet
if extracted_text:
    output_file = 'extracted_text.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(extracted_text)
    print(f"\nÇıkarılan metin {output_file} dosyasına kaydedildi")

# Training arguments ayarla
training_args = TrainingArguments(
    output_dir="test_trainer",
    use_mps_device=False,  # MPS hatası nedeniyle False olarak ayarlandı
    no_cuda=True,
)

# Modeli yükle ve metin parçalarını işlemeye başla
accelerator = Accelerator()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    DEFAULT_MODEL,
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
    device_map=device,
)
tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, use_safetensors=True)
model, tokenizer = accelerator.prepare(model, tokenizer)

CHUNK_SIZE = 1000
if extracted_text:
    chunks = create_word_bounded_chunks(extracted_text, CHUNK_SIZE)
    output_file = f"clean_{os.path.basename(output_file)}"
    processed_text = ""
    with open(output_file, 'w', encoding='utf-8') as out_file:
        for chunk_num, chunk in enumerate(tqdm(chunks, desc="Parçalar işleniyor")):
            conversation = [
                {"role": "system", "content": """You are a world class text pre-processor, here is the raw data from a PDF, please parse and return it in a way that is crispy and usable to send to a podcast writer.

The raw data is messed up with new lines, Latex math and you will see fluff that we can remove completely. Basically take away any details that you think might be useless in a podcast author's transcript.

Remember, the podcast could be on any topic whatsoever so the issues listed above are not exhaustive

Please be smart with what you remove and be creative ok?

Remember DO NOT START SUMMARIZING THIS, YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED

Be very smart and aggressive with removing details, you will get a running portion of the text and keep returning the processed text.

PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPATILISATION ETC LIKES

ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions ok?
Here is the text:"""
                },
                {"role": "user", "content": chunk},
            ]
            prompt = tokenizer.apply_chat_template(conversation, tokenize=False)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    temperature=0.7,
                    top_p=0.9,
                    max_new_tokens=512
                )
            
            processed_chunk = tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):].strip()
            out_file.write(processed_chunk + "\n")
            out_file.flush()
    print(f"\nİşlenmiş metin {output_file} dosyasına kaydedildi")
