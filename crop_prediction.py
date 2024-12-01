import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import pickle

# Load your dataset
df = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\CropPrediction\yield_df.csv")  # Make sure to have your dataset in the same folder

# Data preprocessing
df.drop_duplicates()
df.ffill(inplace=True)
df.reset_index(drop=True, inplace=True)
label_encoder = LabelEncoder()
df['Item'] = label_encoder.fit_transform(df['Item'])

# Define features and target variable
X = df.drop('Item', axis=1)
y = df['Item']

# Preprocessing
ohe = OneHotEncoder(drop='first')
scaler = StandardScaler()
preprocessor = ColumnTransformer(
    transformers=[
        ('num', scaler, ['average_rain_fall_mm_per_year', 'avg_temp']),  # Replace with your numerical columns
        ('cat', ohe, ['Area'])  # Replace with your categorical columns
    ]
)

X_processed = preprocessor.fit_transform(X)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save the preprocessor
with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)