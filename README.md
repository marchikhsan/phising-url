Tentu! Berikut adalah versi markdown dari pembahasan kode dan hasil yang telah Anda buat:

```markdown
# Deteksi URL Phishing Menggunakan Random Forest

## 1. Pembahasan Kode

### 1.1 Library Python yang digunakan

1) **`import pandas as pd`**  
   **Fungsi**: Mengimpor library pandas, yang digunakan untuk manipulasi dan analisis data. pandas sangat berguna untuk bekerja dengan data dalam bentuk tabel (DataFrame) dan menyediakan berbagai fungsi untuk membaca, memanipulasi, dan menganalisis data.

2) **`from sklearn.preprocessing import MinMaxScaler`**  
   **Fungsi**: Mengimpor MinMaxScaler dari library sklearn.preprocessing. MinMaxScaler digunakan untuk menormalkan atau mengubah skala fitur data ke rentang [0, 1]. Ini penting untuk algoritma machine learning, karena banyak model lebih sensitif terhadap skala fitur yang berbeda.

3) **`from sklearn.model_selection import train_test_split`**  
   **Fungsi**: Mengimpor train_test_split dari sklearn.model_selection. Fungsi ini digunakan untuk membagi dataset menjadi dua bagian: satu untuk melatih model (X_train, y_train) dan satu lagi untuk menguji model (X_test, y_test). Biasanya, data dibagi dengan proporsi 80% untuk latih dan 20% untuk uji.

4) **`from sklearn.ensemble import RandomForestClassifier`**  
   **Fungsi**: Mengimpor RandomForestClassifier dari sklearn.ensemble. RandomForestClassifier adalah model machine learning berbasis ensemble yang menggunakan banyak pohon keputusan (decision trees) untuk klasifikasi. Ini adalah model yang sangat populer untuk tugas klasifikasi karena kinerjanya yang baik dan kemampuannya menangani data yang kompleks.

5) **`from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay`**  
   **Fungsi**: Mengimpor berbagai fungsi dari sklearn.metrics untuk mengevaluasi performa model:
   - **accuracy_score**: Menghitung akurasi, yaitu persentase prediksi yang benar dibandingkan total data.
   - **classification_report**: Menyediakan laporan yang mencakup precision, recall, F1-score, dan support untuk masing-masing kelas.
   - **confusion_matrix**: Menghitung dan menghasilkan matriks kebingungan, yang menunjukkan jumlah prediksi yang benar dan salah pada setiap kelas.
   - **ConfusionMatrixDisplay**: Menampilkan matriks kebingungan dalam bentuk grafik.

6) **`import joblib`**  
   **Fungsi**: Mengimpor library joblib. joblib digunakan untuk menyimpan dan memuat objek Python (seperti model machine learning). Dalam konteks ini, joblib digunakan untuk menyimpan model yang telah dilatih (phishing_model.pkl) sehingga model tersebut dapat digunakan kembali tanpa perlu melatih ulang.

7) **`import re`**  
   **Fungsi**: Mengimpor re, library untuk menangani ekspresi reguler (regular expressions). re digunakan untuk mencari pola dalam teks, seperti dalam kasus ini untuk menghitung jumlah domain tingkat atas (.com, .org, dll.) dalam URL.

### 1.2 Kode Program Python

#### 1) Load Data
```python
data = pd.read_csv("data.csv")
```
- **Tujuan**: Membaca dataset data.csv yang berisi data URL dan label phishing (0 untuk URL legitimate, 1 untuk phishing).
- **Output**: Dataset dalam bentuk DataFrame dengan kolom-kolom fitur dan label.

#### 2) Preprocessing
```python
X = data.drop(columns=['phishing'])
y = data['phishing']
```
- **Tujuan**: Memisahkan dataset menjadi fitur (X) dan label (y). Kolom phishing adalah label target.
- **Output**: 
  - X: Data fitur URL tanpa kolom phishing.
  - y: Label target.

```python
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```
- **Tujuan**: Melakukan normalisasi nilai fitur menggunakan MinMaxScaler agar semua nilai berada pada rentang [0, 1].
- **Output**: X_scaled, yaitu fitur yang sudah dinormalisasi.

#### 3) Split Data
```python
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
```
- **Tujuan**: Membagi dataset menjadi data latih (80%) dan data uji (20%) secara acak.
- **Output**: 
  - X_train, y_train: Data latih.
  - X_test, y_test: Data uji.

#### 4) Train Model
```python
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
```
- **Tujuan**: Melatih model menggunakan algoritma Random Forest dengan 100 pohon keputusan (n_estimators=100).
- **Output**: Model yang telah dilatih (model).

#### 5) Evaluate Model
```python
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```
- **Tujuan**: Mengevaluasi performa model pada data uji.
- **Output**: 
  - Akurasi model.
  - Laporan klasifikasi (precision, recall, F1-score, dan support).

#### 6) Save Model
```python
joblib.dump(model, "phishing_model.pkl")
```
- **Tujuan**: Menyimpan model yang sudah dilatih ke file phishing_model.pkl untuk digunakan kembali tanpa perlu melatih ulang.

#### 7) Visualize Confusion Matrix
```python
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legitimate", "Phishing"])
disp.plot()
```
- **Tujuan**: Membuat Visualisasi Confusion Matrix untuk melihat jumlah prediksi benar dan salah.
- **Output**: Grafik Confusion Matrix.

#### 8) Load Model
```python
loaded_model = joblib.load("phishing_model.pkl")
```
- **Tujuan**: Memuat kembali model yang telah disimpan sebelumnya untuk digunakan dalam prediksi baru.

#### 9) Extract Features from URL
```python
def extract_features(url):
    features = {
        "qty_dot_url": url.count('.'),
        "qty_hyphen_url": url.count('-'),
        "qty_underline_url": url.count('_'),
        "qty_slash_url": url.count('/'),
        "qty_questionmark_url": url.count('?'),
        "qty_equal_url": url.count('='),
        "qty_at_url": url.count('@'),
        "qty_and_url": url.count('&'),
        "qty_exclamation_url": url.count('!'),
        "qty_space_url": url.count(' '),
        "qty_tilde_url": url.count('~'),
        "qty_comma_url": url.count(','),
        "qty_plus_url": url.count('+'),
        "qty_asterisk_url": url.count('*'),
        "qty_hashtag_url": url.count('#'),
        "qty_dollar_url": url.count('$'),
        "qty_percent_url": url.count('%'),
        "qty_tld_url": len(re.findall(r'\.\w+', url)),
        "length_url": len(url),
        "email_in_url": 1 if 'mailto:' in url else 0,
        "qty_redirects": url.count('//') - 1,
        "url_google_index": 1 if 'google.com' in url else 0,
        "domain_google_index": 1 if 'domain.google.com' in url else 0,
        "url_shortened": 1 if len(url) < 20 else 0
    }
    return features
```
- **Tujuan**: Mengekstrak fitur dari sebuah URL berdasarkan karakteristik tertentu, seperti jumlah karakter spesial, panjang URL, dan keberadaan mailto.

#### 10) Predict New Data from URL
```python
input_url = input("Enter a URL to check: ")
new_data_features = extract_features(input_url)
new_data_df = pd.DataFrame([new_data_features])
new_data_scaled = scaler.transform(new_data_df)
prediction = loaded_model.predict(new_data_scaled)
print("\nPrediction for the URL:", "Phishing" if prediction[0] == 1 else "Legitimate")
```
- **Tujuan**: 
  1. Mengekstrak fitur dari URL baru.
  2. Menskalakan fitur menggunakan scaler yang sudah dilatih.
  3. Melakukan prediksi apakah URL tersebut phishing atau tidak.
- **Output**: Menampilkan hasil prediksi (Legitimate atau Phishing).

#### 11) Update Dataset
```python
new_data_features['phishing'] = prediction[0]
new_data_df = pd.DataFrame([new_data_features])
data = pd.concat([data, new_data_df], ignore_index=True)
data.to_csv("data.csv", index=False)
```
- **Tujuan**: 
  1. Menambahkan URL baru beserta label hasil prediksi ke dataset.
  2. Menyimpan dataset yang telah diperbarui ke file data.csv.

---

## 2. Pembahasan Hasil

### 2.1 Akurasi: 0.914
- Akurasi menunjukkan persentase prediksi yang benar dari total data uji.
- Dalam hal ini, 91.4% prediksi yang dilakukan oleh model benar, artinya model cukup baik dalam mengklasifikasikan URL sebagai legitimate atau phishing.

### 2.2 Laporan Klasifikasi:
Laporan klasifikasi memberikan metrik seperti precision, recall, f1-score, dan support untuk setiap kelas (0 untuk Legitimate dan 1 untuk Phishing).

#### 1) Kelas 0 (Legitimate):
- **Precision (0.93)**: Dari semua prediksi yang dianggap sebagai kelas 0 (Legitimate), 93% di antaranya benar-benar legitimate.
- **Recall (0.94)**: Dari semua URL yang benar-benar legitimate (dalam data uji), 94% berhasil diprediksi dengan benar oleh model.
- **F1-Score (0.93)**: F1-Score adalah rata-rata harmonis antara precision dan recall, yang memberikan gambaran keseluruhan dari kemampuan model untuk mendeteksi kelas 0.

#### 2) Kelas 1 (Phishing):
- **Precision (0.89)**: Dari semua prediksi yang dianggap phishing (kelas 1), 89% di antaranya benar-benar phishing.
- **Recall (0.86)**: Dari semua URL phishing yang sebenarnya (dalam data uji), 86% berhasil diprediksi dengan benar.
- **F1-Score (0.87)**: F1-Score untuk phishing sedikit lebih rendah dibandingkan dengan legitimate.

#### 3) Akurasi Keseluruhan:
- **Akurasi keseluruhan (0.91)** menunjukkan bahwa model secara keseluruhan memiliki kemampuan yang baik dalam mengklasifikasikan data.

#### 4) Rata-rata Makro (Macro Avg):
- **Precision (0.91)**, **Recall (0.90)**, **F1-Score (0.90)**: Model cukup seimbang dalam mengklasifikasikan kedua kelas.

#### 5) Rata-rata Tertimbang (Weighted Avg):
- **Precision (0.91)**, **Recall (0.91)**, **F1-Score (0.91)**: Metrik ini mempertimbangkan jumlah instance di setiap kelas.

---

## 3. Kesimpulan

1. Model memiliki akurasi 91.4%, yang cukup baik.
2. Precision dan Recall untuk kelas legitimate (0) sangat tinggi, menunjukkan model sangat efektif dalam mengidentifikasi URL legitimate.
3. Untuk phishing (1), meskipun precision sedikit lebih rendah, recall masih cukup tinggi, yang berarti model dapat mendeteksi phishing dengan baik.
4. Secara keseluruhan, model menunjukkan keseimbangan yang baik antara menghindari kesalahan dan menangkap URL phishing yang sebenarnya.
```

Dokumen di atas sudah disusun dalam format markdown. Anda bisa menyalin dan menyimpannya sebagai file `.md`.