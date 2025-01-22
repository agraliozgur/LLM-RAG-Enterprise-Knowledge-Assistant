# scripts/data_preprocessing.py

import os
import re
import pdfplumber
import docx
from bs4 import BeautifulSoup
import pytesseract
from PIL import Image
import json

def clean_text(text: str) -> str:
    """
    Metin üzerinde basit temizlik yapar:
    - Fazla boşlukları giderir
    - Satır sonu vb. karakterleri temizler
    - Gerekiyorsa ek normalizasyon
    """
    # satır sonlarını boşlukla değiştir
    text = text.replace('\n', ' ')
    # birden fazla boşluk varsa teke indir
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Bir PDF dosyasından metin çıkartan basit fonksiyon.
    Eğer PDF taranmışsa (metin katmanı yoksa) OCR gerekebilir.
    """
    full_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                cleaned = clean_text(text)
                full_text.append(cleaned)
    return "\n".join(full_text)

def extract_text_from_scanned_pdf(pdf_path: str) -> str:
    """
    Eğer PDF taranmışsa, pdfplumber page'lerinden her sayfayı resim olarak
    alıp OCR (tesseract) ile metin çekebiliriz.
    Bu işlem daha yavaştır ama taranmış belgeler için gereklidir.
    """
    full_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # PDF sayfasını PIL Image objesine dönüştür
            pil_image = page.to_image(resolution=300).original
            # Tesseract OCR ile metin çıkart
            ocr_text = pytesseract.image_to_string(pil_image, lang='tur')  # Türkçe dil modeli varsa
            cleaned = clean_text(ocr_text)
            full_text.append(cleaned)
    return "\n".join(full_text)

def extract_text_from_docx(docx_path: str) -> str:
    """
    Word dosyalarından (docx) metin okur.
    """
    doc = docx.Document(docx_path)
    full_text = []
    for para in doc.paragraphs:
        cleaned = clean_text(para.text)
        if cleaned:
            full_text.append(cleaned)
    return "\n".join(full_text)

def extract_text_from_html(html_path: str) -> str:
    """
    HTML dosyasından body metnini temizleyerek alır.
    """
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    soup = BeautifulSoup(html_content, 'lxml')
    # sadece body kısımlarını çekelim
    text = soup.get_text(separator=' ')
    cleaned = clean_text(text)
    return cleaned

def chunk_text(text: str, chunk_size=500, overlap=50) -> list:
    """
    Metni chunk_size kelime civarında bölümler halinde parçalara ayırır.
    Overlap kadar kelime bir önceki chunk'la ortak bırakılır (retrieval kalitesini artırabilir).
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_str = " ".join(chunk_words)
        chunks.append(chunk_str)
        start += chunk_size - overlap
    return chunks

def save_chunks_to_json(file_name, chunks, output_dir):
    """
    Parçalanmış metinleri JSON formatında kaydeder.
    """
    data_to_save = []
    for i, ch in enumerate(chunks):
        data_to_save.append({
            "chunk_id": i,
            "text": ch
        })
    out_path = os.path.join(output_dir, file_name + ".json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=4)

def preprocess_data_lake(data_lake_dir: str, output_dir: str):
    """
    data_lake_dir içinde PDF, Word, HTML vb. dosyaları tarar,
    metin olarak okuyup temizler ve output_dir'e JSON dosyaları olarak kaydeder.
    """
    os.makedirs(output_dir, exist_ok=True)

    # PDF dosyaları
    pdf_folder = os.path.join(data_lake_dir, "pdf_files")
    if os.path.exists(pdf_folder):
        pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith(".pdf")]
        for fname in pdf_files:
            fpath = os.path.join(pdf_folder, fname)
            # OCR gerekli mi yoksa normal metin mi? (basit bir kontrol ekleyebiliriz)
            # Burada varsayılan normal text extraction'ı seçiyoruz
            text_content = extract_text_from_pdf(fpath)
            # text_content boş ise OCR yapmayı deneyebiliriz
            if not text_content.strip():
                text_content = extract_text_from_scanned_pdf(fpath)

            # Metni parçalara ayır
            chunks = chunk_text(text_content)
            # JSON olarak kaydet
            base_name = os.path.splitext(fname)[0]
            save_chunks_to_json(base_name, chunks, output_dir)

    # Word dosyaları
    word_folder = os.path.join(data_lake_dir, "word_files")
    if os.path.exists(word_folder):
        word_files = [f for f in os.listdir(word_folder) if f.lower().endswith(".docx")]
        for fname in word_files:
            fpath = os.path.join(word_folder, fname)
            text_content = extract_text_from_docx(fpath)
            # Metni parçalara ayır
            chunks = chunk_text(text_content)
            # JSON olarak kaydet
            base_name = os.path.splitext(fname)[0]
            save_chunks_to_json(base_name, chunks, output_dir)

    # HTML dosyaları
    html_folder = os.path.join(data_lake_dir, "html_files")
    if os.path.exists(html_folder):
        html_files = [f for f in os.listdir(html_folder) if f.lower().endswith(".html")]
        for fname in html_files:
            fpath = os.path.join(html_folder, fname)
            text_content = extract_text_from_html(fpath)
            # Metni parçalara ayır
            chunks = chunk_text(text_content)
            # JSON olarak kaydet
            base_name = os.path.splitext(fname)[0]
            save_chunks_to_json(base_name, chunks, output_dir)

    print(f"Veri ön işleme tamamlandı. Temizlenmiş JSON dosyaları '{output_dir}' klasöründe saklandı.")

if __name__ == "__main__":
    data_lake_directory = "../data_lake"
    cleaned_output_directory = "../cleaned_data"
    preprocess_data_lake(data_lake_directory, cleaned_output_directory)

"""
3.3 OCR’ye Dikkat
Kurum içi belgelerin bir kısmı taranmış PDF olabilir. Bu durumda yukarıdaki extract_text_from_scanned_pdf fonksiyonunu kullanmak gerekir.
Bu işlem yavaş ve kaynak tüketicidir. Dolayısıyla sadece metin katmanı boş olan PDF’lerde çalıştırmak isteyebiliriz.
"""
