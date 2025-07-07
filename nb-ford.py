#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Load Dataset ---
url = 'https://docs.google.com/spreadsheets/d/1rU7EkaNTOt7BVVDhcYXplmYI2Rr8aNsvJ0niUTZ6lEY/export?format=csv'
df = pd.read_csv('ford.csv')

# --- Preprocessing ---
X = df.drop('price', axis=1)
y = df['price']

# One-hot encoding
X = pd.get_dummies(X, columns=['model', 'transmission', 'fuelType'], drop_first=True)
train_columns = X.columns

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Naive Bayes (HANYA eksperimen, hasil tidak akurat)
nb_model = GaussianNB()
nb_model.fit(X_train, y_train.astype(int))  # casting y jadi int agar tidak error

# Predict
y_pred = nb_model.predict(X_test)

# Evaluasi
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Hasil Evaluasi Naive Bayes ---")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"R-squared: {r2:.4f}")

# Visualisasi
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Harga Aktual")
plt.ylabel("Harga Prediksi (NB)")
plt.title("Prediksi Naive Bayes vs Harga Aktual")
plt.tight_layout()
plt.show()
#%%