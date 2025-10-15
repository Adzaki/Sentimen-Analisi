import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from transformers import pipeline
import os
import sys
from PIL import Image, ImageDraw, ImageFont
import qrcode

# Set encoding untuk terminal Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ============================================
# KONFIGURASI PATH
# ============================================
PROJECT_PATH = r"C:\Users\User\Desktop\New folder (2)\Analisis sentimen"
DATA_PATH = os.path.join(PROJECT_PATH, "data")
OUTPUT_PATH = os.path.join(PROJECT_PATH, "output")
SCRIPT_PATH = os.path.join(PROJECT_PATH, "script")

# Input dan Output files
INPUT_FILE = os.path.join(DATA_PATH, "Fasilitas masjid.xlsx")
OUTPUT_FILE = os.path.join(OUTPUT_PATH, "Fasilitas_masjid_hasil.xlsx")

# Buat folder jika belum ada
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(SCRIPT_PATH, exist_ok=True)

print("\n" + "=" * 60)
print("SISTEM ANALISIS SENTIMEN FASILITAS MASJID")
print("=" * 60)
print(f"\nStruktur Folder:")
print(f"  Project : {PROJECT_PATH}")
print(f"  Data    : {DATA_PATH}")
print(f"  Output  : {OUTPUT_PATH}")
print(f"  Script  : {SCRIPT_PATH}")

# ============================================
# STEP 1: BACA FILE EXCEL
# ============================================
print("\n[*] STEP 1: Membaca file Excel...")
excel_file = pd.ExcelFile(INPUT_FILE)
df = pd.read_excel(INPUT_FILE, sheet_name=0)

print(f"[OK] File berhasil dibaca!")
print(f"    Jumlah baris: {len(df)}")
print(f"    Jumlah kolom: {len(df.columns)}")
print(f"    Nama kolom: {list(df.columns)}")

# Cari kolom ulasan dengan lebih teliti
possible_names = ['ulasan', 'komentar', 'review', 'fasilitas', 'feedback', 'pendapat', 'teks', 'isi']
text_column = None
validation_column = None

# Prioritas 1: Cek nama kolom
for col in df.columns:
    col_lower = str(col).lower()
    
    # Cari kolom ulasan
    for name in possible_names:
        if name in col_lower and 'memuaskan' not in col_lower:
            text_column = col
            print(f"[OK] Kolom ulasan ditemukan: '{col}'")
            break
    
    # Cari kolom validasi (yang ada "memuaskan")
    if 'memuaskan' in col_lower or 'puas' in col_lower:
        validation_column = col
        print(f"[OK] Kolom validasi ditemukan: '{col}'")
    
    if text_column:
        break

# Prioritas 2: Cari kolom dengan teks terpanjang
if text_column is None:
    print("[!] Nama kolom tidak cocok, mencari berdasarkan panjang teks...")
    avg_lengths = {}
    for col in df.columns:
        try:
            # Hitung rata-rata panjang karakter
            avg_length = df[col].astype(str).str.len().mean()
            if avg_length > 15:  # Minimal 15 karakter
                avg_lengths[col] = avg_length
        except:
            continue
    
    if avg_lengths:
        text_column = max(avg_lengths, key=avg_lengths.get)
        print(f"[OK] Kolom terdeteksi: '{text_column}' (avg {avg_lengths[text_column]:.0f} karakter)")

if text_column is None:
    print("[ERROR] Tidak dapat menemukan kolom ulasan!")
    print("        Silakan pilih kolom manual dari:", list(df.columns))
    exit()

# Bersihkan data
print(f"\n[*] Membersihkan data...")
df_clean = df[df[text_column].notna()].copy()
df_clean[text_column] = df_clean[text_column].astype(str)
df_clean = df_clean[df_clean[text_column].str.strip() != '']
df_clean = df_clean[df_clean[text_column].str.len() > 3]  # Minimal 3 karakter

print(f"[OK] Data valid: {len(df_clean)} ulasan (dari {len(df)} total)")

# ============================================
# STEP 2: LOAD MODEL AI
# ============================================
print(f"\n[*] STEP 2: Memuat model IndoBERT...")
print("    (Proses ini membutuhkan waktu...)")

try:
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="w11wo/indonesian-roberta-base-sentiment-classifier",
        device=-1,  # CPU
        truncation=True,
        max_length=512
    )
    print("[OK] Model berhasil dimuat!")
except Exception as e:
    print(f"[ERROR] Gagal memuat model: {str(e)}")
    print("        Pastikan sudah install: pip install transformers torch")
    exit()

# ============================================
# STEP 3: ANALISIS SENTIMEN
# ============================================
print(f"\n[*] STEP 3: Menganalisis {len(df_clean)} ulasan...")
print("    (Ini akan memakan waktu beberapa menit)\n")

sentiments = []
confidence_scores = []
failed_count = 0

for idx, text in enumerate(df_clean[text_column], 1):
    try:
        # Bersihkan teks: hapus whitespace berlebih
        text_clean = ' '.join(text.split())
        
        # Analisis dengan model AI
        result = sentiment_analyzer(text_clean)[0]
        sentiment = result['label']
        confidence = result['score']
        
        sentiments.append(sentiment)
        confidence_scores.append(confidence)
        
        # Progress indicator setiap 5 ulasan
        if idx % 5 == 0:
            progress = (idx / len(df_clean)) * 100
            print(f"    Progress: [{idx:4d}/{len(df_clean)}] {progress:5.1f}% - Last: {sentiment} ({confidence:.2f})")
            
    except Exception as e:
        print(f"    [!] Error pada baris {idx}: {str(e)[:50]}")
        sentiments.append("ERROR")
        confidence_scores.append(0.0)
        failed_count += 1

df_clean['Sentimen'] = sentiments
df_clean['Confidence'] = confidence_scores

# Normalisasi label (case-insensitive)
df_clean['Sentimen'] = df_clean['Sentimen'].str.upper()

print(f"\n[OK] Analisis selesai!")
print(f"    Sukses: {len(df_clean) - failed_count}")
print(f"    Gagal: {failed_count}")

# ============================================
# STEP 4: HITUNG STATISTIK DETAIL
# ============================================
print("\n" + "=" * 60)
print("HASIL ANALISIS SENTIMEN")
print("=" * 60)

label_mapping = {
    'POSITIVE': 'Positif',
    'NEGATIVE': 'Negatif',
    'NEUTRAL': 'Netral',
    'ERROR': 'Error'
}

sentiment_counts = df_clean['Sentimen'].value_counts()
total = len(df_clean)

print(f"\n{'Kategori':<15} {'Jumlah':<10} {'Persentase':<12} {'Avg Confidence'}")
print("-" * 55)

for sentiment in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
    count = sentiment_counts.get(sentiment, 0)
    percentage = (count / total * 100) if total > 0 else 0
    
    # Hitung rata-rata confidence per kategori
    mask = df_clean['Sentimen'] == sentiment
    avg_conf = df_clean.loc[mask, 'Confidence'].mean() * 100 if count > 0 else 0
    
    indo_label = label_mapping.get(sentiment, sentiment)
    print(f"{indo_label:<15} {count:<10} {percentage:>6.2f}%      {avg_conf:>5.1f}%")

if 'ERROR' in sentiment_counts:
    count = sentiment_counts['ERROR']
    percentage = (count / total * 100)
    print(f"{'Error':<15} {count:<10} {percentage:>6.2f}%")

print("-" * 55)
print(f"{'TOTAL':<15} {total:<10} {'100.00%':<12}")
print("-" * 55)

# Statistik confidence keseluruhan
overall_conf = df_clean[df_clean['Sentimen'] != 'ERROR']['Confidence'].mean() * 100
print(f"\nRata-rata Confidence: {overall_conf:.2f}%")

# Distribusi confidence
low_conf = (df_clean['Confidence'] < 0.6).sum()
med_conf = ((df_clean['Confidence'] >= 0.6) & (df_clean['Confidence'] < 0.8)).sum()
high_conf = (df_clean['Confidence'] >= 0.8).sum()

print(f"\nDistribusi Tingkat Keyakinan:")
print(f"  Rendah (<60%):  {low_conf:3d} ulasan ({low_conf/total*100:5.1f}%)")
print(f"  Sedang (60-80%): {med_conf:3d} ulasan ({med_conf/total*100:5.1f}%)")
print(f"  Tinggi (>80%):  {high_conf:3d} ulasan ({high_conf/total*100:5.1f}%)")

# ============================================
# VALIDASI DENGAN KOLOM GROUND TRUTH
# ============================================
accuracy = None
if validation_column is not None:
    print("\n" + "=" * 60)
    print("VALIDASI HASIL AI")
    print("=" * 60)
    
    # Normalisasi jawaban di kolom validasi
    df_clean['Validasi'] = df_clean[validation_column].astype(str).str.strip().str.lower()
    
    # Map jawaban ke sentimen
    def map_answer_to_sentiment(answer):
        if pd.isna(answer) or answer == 'nan':
            return None
        answer = str(answer).lower().strip()
        
        # Positif
        if any(word in answer for word in ['ya', 'iya', 'memuaskan', 'puas', 'bagus', 'baik']):
            return 'POSITIVE'
        # Negatif  
        elif any(word in answer for word in ['tidak', 'kurang', 'buruk', 'jelek', 'mengecewakan']):
            return 'NEGATIVE'
        # Netral
        elif any(word in answer for word in ['biasa', 'cukup', 'lumayan', 'standar']):
            return 'NEUTRAL'
        else:
            return None
    
    df_clean['Sentimen_Manual'] = df_clean['Validasi'].apply(map_answer_to_sentiment)
    
    # Hitung akurasi hanya untuk data yang punya validasi
    df_valid = df_clean[df_clean['Sentimen_Manual'].notna()].copy()
    
    if len(df_valid) > 0:
        correct = (df_valid['Sentimen'] == df_valid['Sentimen_Manual']).sum()
        accuracy = (correct / len(df_valid)) * 100
        
        print(f"\nJumlah data tervalidasi: {len(df_valid)} dari {total}")
        print(f"Prediksi benar: {correct}")
        print(f"Prediksi salah: {len(df_valid) - correct}")
        print(f"AKURASI MODEL: {accuracy:.2f}%")
        
        # Confusion matrix sederhana
        print(f"\nDetail per Kategori:")
        for sentiment in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
            manual_count = (df_valid['Sentimen_Manual'] == sentiment).sum()
            ai_count = (df_valid['Sentimen'] == sentiment).sum()
            correct_count = ((df_valid['Sentimen'] == sentiment) & 
                           (df_valid['Sentimen_Manual'] == sentiment)).sum()
            
            if manual_count > 0:
                precision = (correct_count / ai_count * 100) if ai_count > 0 else 0
                recall = (correct_count / manual_count * 100)
                print(f"  {label_mapping[sentiment]:8s}: Precision {precision:5.1f}% | Recall {recall:5.1f}%")
        
        # Tampilkan contoh kesalahan prediksi
        wrong_predictions = df_valid[df_valid['Sentimen'] != df_valid['Sentimen_Manual']]
        
        if len(wrong_predictions) > 0:
            print(f"\nContoh Prediksi Salah ({min(3, len(wrong_predictions))} dari {len(wrong_predictions)}):")
            for idx, row in wrong_predictions.head(3).iterrows():
                text = str(row[text_column])[:80]
                ai_pred = label_mapping[row['Sentimen']]
                manual = label_mapping[row['Sentimen_Manual']]
                conf = row['Confidence'] * 100
                print(f"  AI: {ai_pred:8s} | Manual: {manual:8s} | Conf: {conf:.1f}%")
                print(f"  Text: {text}...")
                print()
    else:
        print("\n[!] Tidak ada data yang bisa divalidasi")
        print("    Format jawaban mungkin tidak sesuai")
else:
    print(f"\n[!] Kolom validasi tidak ditemukan")
    print(f"    Tidak dapat menghitung akurasi")

# ============================================
# STEP 5: SIMPAN KE EXCEL
# ============================================
print(f"\n[*] STEP 4: Menyimpan hasil ke Excel...")

df_output = df_clean.copy()

# Hanya simpan kolom penting saja
# Hapus kolom duplikat, hanya pakai yang Bahasa Indonesia
df_output['Kategori_Sentimen'] = df_output['Sentimen'].map(label_mapping)

# Tambahkan kolom validasi jika ada
if validation_column is not None and 'Sentimen_Manual' in df_output.columns:
    df_output['Sentimen_Manual_Indo'] = df_output['Sentimen_Manual'].map(
        lambda x: label_mapping.get(x, '') if pd.notna(x) else ''
    )
    df_output['Status_Validasi'] = df_output.apply(
        lambda row: 'BENAR' if row['Sentimen'] == row['Sentimen_Manual'] else 'SALAH' 
        if pd.notna(row['Sentimen_Manual']) else '',
        axis=1
    )
    # Hapus kolom raw yang tidak perlu
    df_output = df_output.drop(columns=['Sentimen', 'Sentimen_Manual', 'Validasi'], errors='ignore')
else:
    # Jika tidak ada validasi, hapus kolom Sentimen raw
    df_output = df_output.drop(columns=['Sentimen'], errors='ignore')

# Format confidence jadi persentase
df_output['Tingkat_Keyakinan'] = df_output['Confidence'].apply(lambda x: f"{x*100:.2f}%")
df_output = df_output.drop(columns=['Confidence'], errors='ignore')

try:
    # Simpan ke Excel
    with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
        df_output.to_excel(writer, index=False, sheet_name='Hasil Analisis')
        
        # Akses workbook untuk atur lebar kolom
        workbook = writer.book
        worksheet = writer.sheets['Hasil Analisis']
        
        # Atur lebar kolom otomatis berdasarkan isi
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            
            # Tambah padding dan set lebar minimum
            adjusted_width = max(max_length + 3, 15)
            worksheet.column_dimensions[column_letter].width = adjusted_width
    
    print(f"[OK] Hasil disimpan ke:")
    print(f"    {OUTPUT_FILE}")
    print(f"    Total baris: {len(df_output)}")
except Exception as e:
    print(f"[ERROR] Gagal menyimpan: {str(e)}")

# ============================================
# STEP 6: VISUALISASI BAR CHART
# ============================================
print(f"\n[*] STEP 5: Membuat visualisasi...")

chart_data, chart_labels, chart_colors = [], [], []
color_map = {
    'POSITIVE': '#10b981',  # Hijau
    'NEGATIVE': '#ef4444',  # Merah
    'NEUTRAL': '#6b7280'    # Abu-abu
}

for sentiment in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
    count = sentiment_counts.get(sentiment, 0)
    if count > 0:
        chart_data.append(count)
        chart_labels.append(label_mapping[sentiment])
        chart_colors.append(color_map[sentiment])

plt.figure(figsize=(10, 6))
bars = plt.bar(chart_labels, chart_data, color=chart_colors, alpha=0.85, 
               edgecolor='black', linewidth=1.5)

# Label dengan jumlah dan persentase
for bar, count in zip(bars, chart_data):
    percentage = (count / total * 100)
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{count}\n({percentage:.1f}%)',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.title('Distribusi Sentimen Ulasan Fasilitas Masjid', 
          fontsize=15, fontweight='bold', pad=15)
plt.xlabel('Kategori Sentimen', fontsize=12, fontweight='bold')
plt.ylabel('Jumlah Ulasan', fontsize=12, fontweight='bold')
plt.ylim(0, max(chart_data) * 1.15)  # Beri ruang untuk label
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()

chart_path = os.path.join(OUTPUT_PATH, 'distribusi_sentimen.png')
plt.savefig(chart_path, dpi=300, bbox_inches='tight')
print(f"[OK] Chart disimpan: {chart_path}")
plt.close()

# ============================================
# STEP 7: WORDCLOUD
# ============================================
print(f"[*] Membuat WordCloud...")

# Gabungkan semua teks
all_text = " ".join(df_clean[text_column].astype(str))

# Buat WordCloud dengan konfigurasi lebih baik
wordcloud = WordCloud(
    width=1600,
    height=800,
    background_color='white',
    colormap='viridis',
    max_words=100,
    relative_scaling=0.5,
    min_font_size=10,
    collocations=False  # Hindari duplikasi pasangan kata
).generate(all_text)

plt.figure(figsize=(14, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('WordCloud Ulasan Fasilitas Masjid', 
          fontsize=16, fontweight='bold', pad=15)
plt.tight_layout(pad=0)

wordcloud_path = os.path.join(OUTPUT_PATH, 'wordcloud_fasilitas.png')
plt.savefig(wordcloud_path, dpi=300, bbox_inches='tight')
print(f"[OK] WordCloud disimpan: {wordcloud_path}")
plt.close()

# ============================================
# STEP 8: BUAT POSTER GABUNGAN
# ============================================
print(f"\n[*] STEP 6: Membuat poster gabungan...")

try:
    # Generate QR Code
    print(f"[*] Membuat QR Code...")
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=2,
    )
    qr.add_data('https://github.com/Adzaki/Sentimen-Analisi')
    qr.make(fit=True)
    
    qr_img = qr.make_image(fill_color="#065f46", back_color="white")
    qr_size = 200  # Ukuran QR code
    qr_img = qr_img.resize((qr_size, qr_size), Image.Resampling.LANCZOS)
    
    # Load gambar-gambar
    img_chart = Image.open(chart_path)
    img_wordcloud = Image.open(wordcloud_path)
    
    # Ukuran poster
    poster_width = 1920
    margin = 60
    title_height = 150
    stats_height = 250
    spacing = 40
    
    # Resize gambar agar sesuai lebar poster
    chart_width = poster_width - (2 * margin)
    chart_ratio = img_chart.height / img_chart.width
    chart_height = int(chart_width * chart_ratio)
    img_chart_resized = img_chart.resize((chart_width, chart_height), Image.Resampling.LANCZOS)
    
    wordcloud_width = poster_width - (2 * margin)
    wordcloud_ratio = img_wordcloud.height / img_wordcloud.width
    wordcloud_height = int(wordcloud_width * wordcloud_ratio)
    img_wordcloud_resized = img_wordcloud.resize((wordcloud_width, wordcloud_height), Image.Resampling.LANCZOS)
    
    # Hitung tinggi total poster
    poster_height = (title_height + stats_height + chart_height + 
                     wordcloud_height + (4 * spacing) + (2 * margin))
    
    # Buat canvas poster dengan gradient background (hijau pastel)
    poster = Image.new('RGB', (poster_width, poster_height), color='white')
    draw = ImageDraw.Draw(poster)
    
    # Background gradient (hijau pastel muda ke putih)
    for i in range(poster_height):
        # Gradient dari hijau pastel (#d1f2eb) ke putih (#ffffff)
        ratio = i / poster_height
        r = int(209 + (255 - 209) * ratio)
        g = int(242 + (255 - 242) * ratio)
        b = int(235 + (255 - 235) * ratio)
        draw.rectangle([(0, i), (poster_width, i+1)], fill=(r, g, b))
    
    # Tambahkan border hijau
    border_width = 15
    draw.rectangle([(border_width//2, border_width//2), 
                   (poster_width - border_width//2, poster_height - border_width//2)],
                  outline='#10b981', width=border_width)
    
    # Load font (gunakan default jika tidak ada)
    try:
        font_title = ImageFont.truetype("arial.ttf", 50)
        font_stats = ImageFont.truetype("arial.ttf", 28)
        font_label = ImageFont.truetype("arialbd.ttf", 32)
    except:
        font_title = ImageFont.load_default()
        font_stats = ImageFont.load_default()
        font_label = ImageFont.load_default()
    
    # JUDUL
    y_pos = margin + 30
    title_text = "ANALISIS SENTIMEN ULASAN FASILITAS MASJID"
    title_bbox = draw.textbbox((0, 0), title_text, font=font_title)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (poster_width - title_width) // 2
    
    # Shadow effect untuk judul dengan warna hijau gelap
    draw.text((title_x + 3, y_pos + 3), title_text, fill='#047857', font=font_title)
    draw.text((title_x, y_pos), title_text, fill='#065f46', font=font_title)
    
    # STATISTIK BOX
    y_pos += title_height
    stats_box_margin = margin + 40
    stats_box_width = poster_width - (2 * stats_box_margin)
    stats_box_height = stats_height - 40
    
    # Background box untuk statistik
    draw.rounded_rectangle(
        [(stats_box_margin, y_pos), 
         (stats_box_margin + stats_box_width, y_pos + stats_box_height)],
        radius=20, fill='white', outline='#3b82f6', width=3
    )
    
    # Statistik dalam 3 kolom
    col_width = stats_box_width // 3
    stats_y = y_pos + 30
    
    # Kolom 1: Total Ulasan
    col1_x = stats_box_margin + col_width // 2
    draw.text((col1_x, stats_y), "Total Ulasan", anchor="mm", fill='#374151', font=font_stats)
    draw.text((col1_x, stats_y + 50), str(total), anchor="mm", fill='#065f46', font=font_label)
    
    # Kolom 2: Confidence
    col2_x = stats_box_margin + col_width + col_width // 2
    draw.text((col2_x, stats_y), "Avg Confidence", anchor="mm", fill='#374151', font=font_stats)
    draw.text((col2_x, stats_y + 50), f"{overall_conf:.1f}%", anchor="mm", fill='#065f46', font=font_label)
    
    # Kolom 3: Akurasi (jika ada)
    col3_x = stats_box_margin + 2 * col_width + col_width // 2
    if accuracy is not None:
        draw.text((col3_x, stats_y), "Akurasi Model", anchor="mm", fill='#374151', font=font_stats)
        draw.text((col3_x, stats_y + 50), f"{accuracy:.1f}%", anchor="mm", fill='#059669', font=font_label)
    else:
        draw.text((col3_x, stats_y), "Sentimen Dominan", anchor="mm", fill='#374151', font=font_stats)
        dominant = sentiment_counts.idxmax()
        draw.text((col3_x, stats_y + 50), label_mapping[dominant], anchor="mm", fill='#059669', font=font_label)
    
    # Detail breakdown sentimen dengan persentase
    breakdown_y = stats_y + 120
    pos_count = sentiment_counts.get('POSITIVE', 0)
    neg_count = sentiment_counts.get('NEGATIVE', 0)
    net_count = sentiment_counts.get('NEUTRAL', 0)
    
    pos_pct = (pos_count / total * 100) if total > 0 else 0
    neg_pct = (neg_count / total * 100) if total > 0 else 0
    net_pct = (net_count / total * 100) if total > 0 else 0
    
    breakdown_text = f"Positif: {pos_count} ({pos_pct:.1f}%) | Negatif: {neg_count} ({neg_pct:.1f}%) | Netral: {net_count} ({net_pct:.1f}%)"
    breakdown_bbox = draw.textbbox((0, 0), breakdown_text, font=font_stats)
    breakdown_width = breakdown_bbox[2] - breakdown_bbox[0]
    breakdown_x = (poster_width - breakdown_width) // 2
    draw.text((breakdown_x, breakdown_y), breakdown_text, fill='#6b7280', font=font_stats)
    
    # CHART
    y_pos += stats_height + spacing
    poster.paste(img_chart_resized, (margin, y_pos))
    
    # WORDCLOUD
    y_pos += chart_height + spacing
    poster.paste(img_wordcloud_resized, (margin, y_pos))
    
    # QR CODE di pojok kanan bawah
    qr_margin = 40
    qr_x = poster_width - qr_size - qr_margin
    qr_y = poster_height - qr_size - qr_margin - 80  # 80px untuk footer text
    
    # Background putih untuk QR code
    qr_bg_padding = 15
    draw.rounded_rectangle(
        [(qr_x - qr_bg_padding, qr_y - qr_bg_padding),
         (qr_x + qr_size + qr_bg_padding, qr_y + qr_size + qr_bg_padding)],
        radius=15, fill='white', outline='#10b981', width=3
    )
    
    poster.paste(qr_img, (qr_x, qr_y))
    
    # Label untuk QR Code - Dua baris
    # Baris 1: "SOURCE CODE"
    qr_label1 = "SOURCE CODE"
    qr_label1_bbox = draw.textbbox((0, 0), qr_label1, font=font_label)
    qr_label1_width = qr_label1_bbox[2] - qr_label1_bbox[0]
    qr_label1_x = qr_x + (qr_size - qr_label1_width) // 2
    qr_label1_y = qr_y + qr_size + 20
    draw.text((qr_label1_x, qr_label1_y), qr_label1, fill='#065f46', font=font_label)
    
    # Baris 2: "Scan untuk akses"
    qr_label2 = "Scan untuk akses"
    qr_label2_bbox = draw.textbbox((0, 0), qr_label2, font=font_stats)
    qr_label2_width = qr_label2_bbox[2] - qr_label2_bbox[0]
    qr_label2_x = qr_x + (qr_size - qr_label2_width) // 2
    qr_label2_y = qr_label1_y + 45
    draw.text((qr_label2_x, qr_label2_y), qr_label2, fill='#6b7280', font=font_stats)
    
    # Footer
    footer_y = poster_height - margin - 20
    footer_text = f"Generated by IndoBERT Sentiment Analysis | Total {total} reviews analyzed"
    footer_bbox = draw.textbbox((0, 0), footer_text, font=font_stats)
    footer_width = footer_bbox[2] - footer_bbox[0]
    footer_x = (poster_width - footer_width) // 2
    draw.text((footer_x, footer_y), footer_text, fill='#6b7280', font=font_stats)
    
    # Simpan poster
    poster_path = os.path.join(OUTPUT_PATH, 'poster_analisis_sentimen.png')
    poster.save(poster_path, quality=95, dpi=(300, 300))
    
    print(f"[OK] Poster gabungan disimpan: {poster_path}")
    print(f"    Ukuran: {poster_width}x{poster_height} pixels")
    
    # Tampilkan poster
    print(f"[*] Menampilkan poster...")
    plt.figure(figsize=(16, 20))
    plt.imshow(poster)
    plt.axis('off')
    plt.title('Poster Analisis Sentimen - Fasilitas Masjid', 
              fontsize=18, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print(f"[ERROR] Gagal membuat poster: {str(e)}")
    print(f"        Pastikan library sudah terinstall:")
    print(f"        pip install pillow qrcode[pil]")
    poster_path = None

# ============================================
# STEP 9: RINGKASAN & CONTOH ULASAN
# ============================================
print("\n" + "=" * 60)
print("CONTOH ULASAN PER KATEGORI")
print("=" * 60)

for sentiment in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
    count = sentiment_counts.get(sentiment, 0)
    if count > 0:
        print(f"\n--- {label_mapping[sentiment]} ({count} ulasan) ---")
        
        # Ambil 2 contoh dengan confidence tertinggi
        samples = df_clean[df_clean['Sentimen'] == sentiment].nlargest(2, 'Confidence')
        
        for idx, row in samples.iterrows():
            text = str(row[text_column])[:100]
            conf = row['Confidence']
            print(f"  [{conf*100:.1f}%] {text}...")

# ============================================
# SELESAI
# ============================================
print("\n" + "=" * 60)
print("SEMUA PROSES SELESAI!")
print("=" * 60)
print(f"\n📁 Struktur File Output:")
print(f"  {OUTPUT_PATH}/")
print(f"  ├── Fasilitas_masjid_hasil.xlsx")
print(f"  ├── distribusi_sentimen.png")
print(f"  ├── wordcloud_fasilitas.png")
print(f"  └── poster_analisis_sentimen.png")

print(f"\n📊 File yang dihasilkan:")
print(f"  1. Excel        : Fasilitas_masjid_hasil.xlsx")
print(f"  2. Bar Chart    : distribusi_sentimen.png")
print(f"  3. WordCloud    : wordcloud_fasilitas.png")
if poster_path:
    print(f"  4. Poster       : poster_analisis_sentimen.png ⭐")

print(f"\n✅ Semua file tersimpan di: {OUTPUT_PATH}")
print("=" * 60)