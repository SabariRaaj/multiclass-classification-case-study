import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import joblib
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# Load processed dataset
df = pd.read_csv('data/processed_data.csv')

# Features and target
X = df[['x', 'y', 'z']]
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
print('training XGBC')
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)

# Save XGBoost model
print('saving XGBC')
joblib.dump(xgb_model, 'models/xgb_model.joblib')

# GridSearchCV didn't provide better accuracy than default setting due to time contstraint
# # Define the parameter grid
# param_grid = {
#     'n_estimators': [100, 200, 300],  # Increase the number of trees
#     'max_depth': [3, 5, 7],           # Depth of each tree
#     'learning_rate': [0.01, 0.1, 0.2], # Learning rate
#     'subsample': [0.8, 1.0],          # Fraction of samples to be used for fitting the individual base learners
#     'colsample_bytree': [0.8, 1.0]    # Fraction of features to be used for fitting the individual base learners
# }

# # Initialize the XGBClassifier
# xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# # Set up the GridSearchCV
# grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid,
#                            scoring='accuracy', cv=3, verbose=2, n_jobs=-1)

# # Fit the model
# grid_search.fit(X_train, y_train)

# # Best parameters from grid search
# best_params = grid_search.best_params_

# Train the model with the best parameters
# best_xgb_model = xgb.XGBClassifier(**best_params, use_label_encoder=False, eval_metric='mlogloss')
# best_xgb_model.fit(X_train, y_train)
# Save XGBoost model
# print('saving Best XGBC')
joblib.dump(xgb_model, 'models/best_xgb_model.joblib')


# Define deep learning model
print('training ANN')
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.1),
    Dense(96, activation='relu'),
    Dropout(0.1),
    Dense(80, activation='relu'),
    Dropout(0.1),
    Dense(64, activation='relu'),
    Dropout(0.1),
    Dense(32, activation='relu'),
    Dense(7, activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train deep learning model
history = model.fit(X_train, y_train, epochs=50, batch_size=512,
                    validation_data=(X_test, y_test), callbacks=[early_stopping])

# Save deep learning model
print('saving ANN')
model.save('models/ann_model.h5')
