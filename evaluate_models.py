import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load processed dataset
df = pd.read_csv('data/processed_data.csv')

# Features and target
X = df[['x', 'y', 'z']]
y = df['label']

# Load XGBoost model
xgb_model = joblib.load('models/xgb_model.joblib')

# Load deep learning model
model = load_model('models/ann_model.h5')

# Evaluate XGBoost model
y_pred_xgb = xgb_model.predict(X)
print(f'XGBoost Accuracy: {accuracy_score(y, y_pred_xgb)}')
print(classification_report(y, y_pred_xgb))

# Evaluate deep learning model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_pred_ann = model.predict_classes(X_test_scaled)
print(f'Deep Learning Accuracy: {accuracy_score(y_test, y_pred_ann)}')
print(classification_report(y_test, y_pred_ann))