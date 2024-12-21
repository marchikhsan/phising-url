# Phishing Detection using Random Forest

This project implements a machine learning model using Random Forest to detect phishing URLs. The model extracts various features from URLs and classifies them as either legitimate or phishing attempts.

## Dependencies

The project requires the following Python libraries:
- pandas: Data manipulation and analysis
- scikit-learn: Machine learning algorithms and tools
- joblib: Model persistence
- re: Regular expressions

## Implementation Details

### Library Imports

```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib
import re
```

### Main Components

1. **Data Loading**
   - Reads the dataset from `data.csv`
   - Dataset contains URLs and their corresponding labels (0 for legitimate, 1 for phishing)

2. **Data Preprocessing**
   - Separates features (X) and labels (y)
   - Normalizes feature values using MinMaxScaler to range [0, 1]
   ```python
   X = data.drop(columns=['phishing'])
   y = data['phishing']
   scaler = MinMaxScaler()
   X_scaled = scaler.fit_transform(X)
   ```

3. **Data Splitting**
   - Splits data into training (80%) and testing (20%) sets
   ```python
   X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
   ```

4. **Model Training**
   - Implements Random Forest Classifier with 100 trees
   ```python
   model = RandomForestClassifier(n_estimators=100, random_state=42)
   model.fit(X_train, y_train)
   ```

5. **Model Evaluation**
   - Calculates accuracy and generates classification report
   - Visualizes results using confusion matrix

6. **Model Persistence**
   - Saves trained model to `phishing_model.pkl`
   - Allows model reuse without retraining

### Feature Extraction

The `extract_features` function extracts the following features from URLs:
- Character counts (., -, _, /, ?, =, @, &, !, etc.)
- URL length
- Number of TLDs
- Presence of email indicators
- Number of redirects
- Google index indicators
- URL shortening detection

```python
def extract_features(url):
    features = {
        "qty_dot_url": url.count('.'),
        "qty_hyphen_url": url.count('-'),
        # ... (other features)
        "url_shortened": 1 if len(url) < 20 else 0
    }
    return features
```

### URL Prediction

The system can predict whether a new URL is phishing or legitimate:
1. Takes URL input from user
2. Extracts features
3. Scales features using trained scaler
4. Makes prediction using loaded model
5. Updates dataset with new entry

## Usage

1. Load the model:
   ```python
   loaded_model = joblib.load("phishing_model.pkl")
   ```

2. Predict new URLs:
   ```python
   input_url = input("Enter a URL to check: ")
   new_data_features = extract_features(input_url)
   new_data_df = pd.DataFrame([new_data_features])
   new_data_scaled = scaler.transform(new_data_df)
   prediction = loaded_model.predict(new_data_scaled)
   ```

3. View results:
   ```python
   print("Prediction for the URL:", "Phishing" if prediction[0] == 1 else "Legitimate")
   ```

The system automatically updates the dataset with new predictions, maintaining an evolving training set for future model improvements.
