# Deteksi URL Phishing Menggunakan Random Forest

Proyek ini mengimplementasikan model machine learning menggunakan Random Forest untuk mendeteksi URL phishing. Model ini mengekstrak berbagai fitur dari URL dan mengklasifikasikannya sebagai URL legitimate (sah) atau upaya phishing.

- 24.55.1596 FREDY SITINJAK
- 24.55.1604 RAFI BUDIARTO
- 24.55.1619 HASNI ANAS
- 24.55.1622 MUHAMMAD IKHSAN

## Dependensi

Proyek ini membutuhkan beberapa library Python berikut:

- pandas: Untuk manipulasi dan analisis data
- scikit-learn: Untuk algoritma dan perangkat machine learning
- joblib: Untuk penyimpanan model
- re: Untuk penggunaan regular expressions

## Sumber Dataset

Dataset yang digunakan dalam proyek ini dapat diunduh dari [www.kaggle.com](https://www.kaggle.com/datasets/michaelsannova/phising-url-dataset).

## Detail Implementasi

### Import Library

```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib
import re
```

### Komponen Utama

1. **Memuat Data**
   - Membaca dataset dari `data.csv`
   - Dataset berisi URL dan labelnya (0 untuk legitimate, 1 untuk phishing)

2. **Preprocessing Data**
   - Memisahkan fitur (X) dan label (y)
   - Menormalisasi nilai fitur menggunakan MinMaxScaler ke rentang [0, 1]

   ```python
   X = data.drop(columns=['phishing'])
   y = data['phishing']
   scaler = MinMaxScaler()
   X_scaled = scaler.fit_transform(X)
   ```

3. **Pembagian Data**
   - Membagi data menjadi data latih (80%) dan data uji (20%)

   ```python
   X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
   ```

4. **Pelatihan Model**
   - Mengimplementasikan Random Forest Classifier dengan 100 pohon keputusan

   ```python
   model = RandomForestClassifier(n_estimators=100, random_state=42)
   model.fit(X_train, y_train)
   ```

5. **Evaluasi Model**
   - Menghitung akurasi dan menghasilkan laporan klasifikasi
   - Memvisualisasikan hasil menggunakan confusion matrix

6. **Penyimpanan Model**
   - Menyimpan model yang telah dilatih ke `phishing_model.pkl`
   - Memungkinkan penggunaan model tanpa perlu pelatihan ulang

### Ekstraksi Fitur

Fungsi `extract_features` mengekstrak fitur-fitur berikut dari URL:

- Jumlah karakter (., -, _, /, ?, =, @, &, !, dll.)
- Panjang URL
- Jumlah domain tingkat atas (TLD)
- Keberadaan indikator email
- Jumlah pengalihan (redirect)
- Indikator indeks Google
- Deteksi pemendekan URL

```python
def extract_features(url):
    features = {
        "qty_dot_url": url.count('.'),
        "qty_hyphen_url": url.count('-'),
        # ... (fitur lainnya)
        "url_shortened": 1 if len(url) < 20 else 0
    }
    return features
```

### Prediksi URL

Sistem dapat memprediksi apakah URL baru adalah phishing atau legitimate:

1. Menerima input URL dari pengguna
2. Mengekstrak fitur
3. Menskalakan fitur menggunakan scaler yang telah dilatih
4. Melakukan prediksi menggunakan model yang dimuat
5. Memperbarui dataset dengan entri baru

## Cara Penggunaan

1. Memuat model:

   ```python
   loaded_model = joblib.load("phishing_model.pkl")
   ```

2. Memprediksi URL baru:

   ```python
   input_url = input("Masukkan URL yang ingin dicek: ")
   new_data_features = extract_features(input_url)
   new_data_df = pd.DataFrame([new_data_features])
   new_data_scaled = scaler.transform(new_data_df)
   prediction = loaded_model.predict(new_data_scaled)
   ```

3. Melihat hasil:

   ```python
   print("Hasil prediksi URL:", "Phishing" if prediction[0] == 1 else "Legitimate")
   ```

Sistem secara otomatis memperbarui dataset dengan prediksi baru, mempertahankan set pelatihan yang berkembang untuk peningkatan model di masa depan.
