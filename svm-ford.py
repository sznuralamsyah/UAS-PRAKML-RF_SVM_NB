#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
url = 'https://docs.google.com/spreadsheets/d/1rU7EkaNTOt7BVVDhcYXplmYI2Rr8aNsvJ0niUTZ6lEY/export?format=csv'
df = pd.read_csv(url)

# Fitur dan target
X = df[['model', 'year', 'transmission', 'mileage', 'fuelType', 'tax', 'mpg', 'engineSize']]
y = df['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Kolom kategorikal dan numerikal
categorical_cols = ['model', 'transmission', 'fuelType']
numerical_cols = ['year', 'mileage', 'tax', 'mpg', 'engineSize']

# Preprocessing
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', StandardScaler(), numerical_cols)
])

# Pipeline SVM
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('svm', SVR(kernel='rbf', C=100, gamma=0.1, epsilon=1))
])

# Training
pipeline.fit(X_train, y_train)

# Prediksi
y_pred = pipeline.predict(X_test)

# Evaluasi
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Visualisasi Prediksi vs Harga Aktual
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)

# Tambahkan garis diagonal (perfect prediction line)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Garis Ideal')

plt.xlabel("Harga Aktual")
plt.ylabel("Harga Prediksi")
plt.title("Prediksi vs Harga Aktual - SVM")
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()

# Ekstrak preprocessor dan model dari pipeline
preprocessor = pipeline.named_steps['preprocess']
svm_model = pipeline.named_steps['svm']

# Transformasi data testing DAN konversi ke array
X_test_transformed = preprocessor.transform(X_test).toarray()

# Dapatkan nama fitur setelah preprocessing
feature_names = preprocessor.get_feature_names_out()

# Hitung Permutation Importance
result = permutation_importance(
    svm_model, 
    X_test_transformed, 
    y_test,
    n_repeats=5,
    random_state=42,
    scoring='r2'
)

# Simpan hasil dan ambil 10 fitur teratas
feature_importance = pd.Series(
    result.importances_mean,
    index=feature_names
).sort_values(ascending=False).head(10)  # <-- Langsung ambil top 10

# Plot Top 10 dengan warna kondisional
plt.figure(figsize=(10, 6))
colors = ['#1f77b4' if 'num__' in name else '#ff7f0e' for name in feature_importance.index]
bars = plt.barh(feature_importance.index, feature_importance.values, color=colors)

# Formatting
plt.title('Top 10 Fitur Paling Berpengaruh (Permutation Importance)', fontsize=14)
plt.xlabel('Penurunan R² Score', fontsize=12)
plt.ylabel('Fitur', fontsize=12)
plt.grid(axis='x', alpha=0.3)

# Tambahkan nilai di bar
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
             f'{width:.3f}', 
             va='center', fontsize=9)

# Legenda
num_patch = mpatches.Patch(color='#1f77b4', label='Numerik')
cat_patch = mpatches.Patch(color='#ff7f0e', label='Kategorik')
plt.legend(handles=[num_patch, cat_patch], loc='lower right')

plt.tight_layout()
plt.show()
#%%