# ✅ Complete and Correct Data Cleaning + Preprocessing Script

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# -----------------------------
# 1. Load the dataset
# -----------------------------
data = pd.read_csv("housing_price_dataset.csv")

# -----------------------------
# 2. Data Cleaning
# -----------------------------
# Remove duplicates
data = data.drop_duplicates()

# Handle missing values
data = data.fillna({
    "SquareFeet": data["SquareFeet"].median(),
    "Bedrooms": data["Bedrooms"].mode()[0],
    "Bathrooms": data["Bathrooms"].mode()[0],
    "Neighborhood": data["Neighborhood"].mode()[0],
    "YearBuilt": data["YearBuilt"].median(),
    "Price": data["Price"].median()
})

# -----------------------------
# 3. Define features and target
# -----------------------------
X = data.drop("Price", axis=1)
y = data["Price"]

# -----------------------------
# 4. One-Hot Encode categorical column
# -----------------------------
categorical_features = ["Neighborhood"]
numerical_features = ["SquareFeet", "Bedrooms", "Bathrooms", "YearBuilt"]

# ✅ Use correct parameter for your sklearn version
try:
    encoder = OneHotEncoder(drop="first", sparse_output=False)  # for sklearn >=1.4
except TypeError:
    encoder = OneHotEncoder(drop="first", sparse=False)  # for older versions

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_features),
    ("cat", encoder, categorical_features)
])

# -----------------------------
# 5. Split dataset
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 6. Create preprocessing pipeline
# -----------------------------
pipeline = Pipeline([
    ("preprocessor", preprocessor)
])

# Fit transform training data and transform test data
X_train_processed = pipeline.fit_transform(X_train)
X_test_processed = pipeline.transform(X_test)

# -----------------------------
# 7. Save processed data (optional)
# -----------------------------
np.save("X_train.npy", X_train_processed)
np.save("X_test.npy", X_test_processed)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print("✅ Data cleaning and preprocessing completed successfully!")
print("Training data shape:", X_train_processed.shape)
print("Testing data shape:", X_test_processed.shape)
