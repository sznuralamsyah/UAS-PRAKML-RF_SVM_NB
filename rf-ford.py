#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --- 1. MEMUAT DATA ---
try:
    df = pd.read_csv('ford.csv')
except FileNotFoundError:
    print("Pastikan file berada di direktori yang sama.")
    exit()

print("Dataset berhasil dimuat.")
print(df.head())


# --- 2. PRA-PEMROSESAN DATA ---
X = df.drop('price', axis=1)
y = df['price']

X = pd.get_dummies(X, columns=['model', 'transmission', 'fuelType'], drop_first=True)
train_columns = X.columns


# --- 3. MEMBAGI DATA DAN MELATIH MODEL ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
print("\nMelatih model...")
rf_model.fit(X_train, y_train)
print("Model berhasil dilatih!")


# --- 4. EVALUASI MODEL ---
y_pred = rf_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Hasil Evaluasi Model ---")
print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
print(f"Mean Squared Error (MSE): ${mse:,.2f}")
print(f"R-squared (R2 Score): {r2:.4f}")


# --- 5. VISUALISASI ---

# Feature Importance
importances = rf_model.feature_importances_
feature_names = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
sns.barplot(x=importances[indices][:15], y=feature_names[indices][:15], palette="viridis")
plt.title("Top 15 Fitur yang Paling Berpengaruh terhadap Prediksi Harga")
plt.xlabel("Pentingnya Fitur")
plt.ylabel("Fitur")
plt.tight_layout()
plt.show()

# Plot Prediksi vs Aktual
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Garis diagonal
plt.xlabel("Harga Aktual")
plt.ylabel("Harga Prediksi")
plt.title("Prediksi vs Harga Aktual")
plt.tight_layout()
plt.show()


# --- 7. PERBANDINGAN HARGA AKTUAL VS PREDIKSI PADA 20% DATA ---

# Gabungkan hasil prediksi dan aktual ke dalam satu DataFrame
df_hasil = pd.DataFrame({
    'Harga Aktual': y_test.values,
    'Harga Prediksi': y_pred
})

# Tampilkan 20 baris pertama untuk perbandingan
print("\n--- Perbandingan Harga Aktual vs Prediksi (20% data test) ---")
print(df_hasil.head(20))

# Hitung selisih absolut
df_hasil['Selisih (Absolut)'] = np.abs(df_hasil['Harga Aktual'] - df_hasil['Harga Prediksi'])

# Hitung persentase error
df_hasil['Persentase Error (%)'] = (df_hasil['Selisih (Absolut)'] / df_hasil['Harga Aktual']) * 100

# Tampilkan 10 hasil dengan error terbesar
print("\n--- 10 Data dengan Error Terbesar ---")
print(df_hasil.sort_values(by='Persentase Error (%)', ascending=False).head(10))

# Visualisasi: Harga Aktual vs Prediksi dalam Bar Plot
plt.figure(figsize=(12, 6))
df_hasil_sample = df_hasil.head(20).reset_index(drop=True)
df_hasil_sample[['Harga Aktual', 'Harga Prediksi']].plot(kind='bar', figsize=(14, 6))
plt.title('Perbandingan Harga Aktual dan Prediksi (20 Sample Awal dari Test Set)')
plt.xlabel('Index Sampel')
plt.ylabel('Harga')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.show()
 #%%