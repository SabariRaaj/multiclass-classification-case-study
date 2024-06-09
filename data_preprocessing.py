import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import joblib

# Load the dataset
df = pd.read_csv('data/combined_data.csv')

# Remove non-relevant features
df = df.drop(columns = ['sl', 'file_name'], axis=0)

# Remove label = 0, not part of the classification mentioned in the README file provided
df = df[df['label'] != 0].reset_index(drop=True)

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
df[['x', 'y', 'z']] = scaler.fit_transform(df[['x', 'y', 'z']])

# Address class imbalance with SMOTE
X = df[['x', 'y', 'z']]
y = df['label']

smote = SMOTE()
X_res, y_res = smote.fit_resample(X, y)

# Save the processed dataset
processed_df = pd.DataFrame(X_res, columns=['x', 'y', 'z'])

# Shift class labels to start from 0
processed_df['label'] = y_res - 1

processed_df.to_csv('data/processed_data.csv', index=False)

# Save the scaler and SMOTE object
joblib.dump(scaler, 'models/scaler.joblib')
joblib.dump(smote, 'models/smote.joblib')