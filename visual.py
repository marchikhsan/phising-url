import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib
import re

# 1. Load data
data = pd.read_csv("data.csv")

# 2. Preprocessing
X = data.drop(columns=['phishing'])
y = data['phishing']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate model
y_pred = model.predict(X_test)
print("Akurasi:", accuracy_score(y_test, y_pred))
print("\nLaporan Klasifikasi:\n", classification_report(y_test, y_pred))

# 6. Save model
joblib.dump(model, "phishing_model.pkl")

# 7. Visualize Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Aman", "Phishing"])
disp.plot()

# 8. Load model
loaded_model = joblib.load("phishing_model.pkl")

# 9. Function to extract features from URL
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

# 10. Predict New Data from URL
# Get URL from user
input_url = input("Masukkan URL untuk diperiksa: ")

# Extract features
new_data_features = extract_features(input_url)

# Convert features to DataFrame with appropriate column names
new_data_df = pd.DataFrame([new_data_features])

# Scale features
new_data_scaled = scaler.transform(new_data_df)

# Predict
prediction = loaded_model.predict(new_data_scaled)[0]

# Label prediction
label = "Phishing" if prediction == 1 else "Aman"
print(f"\nPrediksi untuk URL: {label}")

# Adding new datalabel url to dataset
new_data_features['phishing'] = prediction

# Convert to DataFrame
new_data_df = pd.DataFrame([new_data_features])

# Adding ke dataset
data = pd.concat([data, new_data_df], ignore_index=True)

# Update dataset with new data
data.to_csv("data.csv", index=False)
print("URL berhasil ditambahkan ke dataset!")
