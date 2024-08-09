import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import psutil
import math
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

data = pd.read_csv('Heart_Disease_Prediction.csv') # Menyimpan hasil import ke dalam variabel data

data.head() # Menampilkan 5 data teratas dari dataset

data.shape # Melihat total dimensi data

data.info() # Memberikan informasi type data

data.isna().sum() # Mengecek kolum null

data.describe()

# Merubah value objek pada colom heart disease menjadi numerik

categorical_cols = ['Heart Disease'] # Memilih kolom yang akan diubah

label_encoder = LabelEncoder()

for col in categorical_cols: # Mengambil kolom yang ada dalam categoricals_cols
    data[col] = label_encoder.fit_transform(data[col]) # Merubah value kolom menjadi bentuk numerik

data.head()

# Menghitung jumlah row berdasarkan nilai pada colum heart disease
classification_counts = data['Heart Disease'].value_counts()
print(classification_counts)

# Merubah colum dengan value 0 - 1 menjadi No - Yes (Object)

binary_cols = ['FBS over 120', 'Exercise angina', 'Heart Disease']

def yes_no(col): # Jika value dalam kolom adalah 1 maka ubah menjadi Yes, jika tidak ubah menjadi No
    return 'Yes' if col == 1 else 'No'

for col in binary_cols: # Masukkan perubahan ke dalam kolom yang berada dalam binary_cols
    data[col] = data[col].apply(yes_no)

data.head() # Melihat 5 data setelah diperbaharui

# Visualisasi penyebaran data dengan histogram
numeric_cols = [col for col in data.columns if data[col].dtype != 'object' and col != 'Sex'] # Hanya memilih colom yang bukan object dan kolom sex
cols = 4
row = math.ceil(len(numeric_cols) / cols)

fig, axes = plt.subplots(nrows=row, ncols=cols, figsize=(28,12))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    sns.histplot(data[col],color='blue', edgecolor='black', kde=True, ax=axes[i])

plt.show()

# Visualisasi penyebaran data dengan histogram
numeric_cols = [col for col in data.columns if data[col].dtype != 'object' and col != 'Sex'] # Hanya memilih colom yang bukan object dan kolom sex
cols = 4
row = math.ceil(len(numeric_cols) / cols)

fig, axes = plt.subplots(nrows=row, ncols=cols, figsize=(28,12))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    sns.boxplot(x=data[col], ax=axes[i]) # Tipe box plot

plt.show()

# Penghapusan data outlier menggunakan teknik IQR
def iqr(data, column):
    Q1 = np.percentile(data[column], 25)
    Q3 = np.percentile(data[column], 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data_filtered = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return data_filtered

# Memilih colom yang bukan object
for col in data.columns:
    if data[col].dtype != 'object' :
      data = iqr(data, col)

# Merubah pada dataset asli
data.reset_index(drop=True, inplace=True)
data.head()

categorical_cols = ['FBS over 120', 'Exercise angina', 'Heart Disease']

label_encoder = LabelEncoder()

for col in categorical_cols:
    data[col] = label_encoder.fit_transform(data[col])

data.head()

data.shape

# Mengubah seluruh data non integer menjadi integer / numerikal
label_encoder = LabelEncoder()

for col in data:
    data[col] = label_encoder.fit_transform(data[col])

data.head()

# Membagi dataset menjadi kolom evaluasi (X) dan target (y)
array = data.values
X = array[:,0:13]
y = array[:,13]

# Membagi dataset menjadi data uji dan data latih dengan rasio 20 / 80
X_train, X_validation, y_train, y_validation = train_test_split(X,y,test_size=0.20, random_state=1, shuffle=True)

# Melakukan standarisasi nilai pada dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_validation = scaler.fit_transform(X_validation)

# Melakukan standarisasi nilai pada dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_validation = scaler.fit_transform(X_validation)
# Fungsi untuk mengevaluasi pemodelan berdasarkan akurasi, presisi, recall, f1-score, auc dan waktu eksekusi
def evaluate(model, X_validation, y_validation):
    start_time = time.time() # Memulai waktu perhitungan eksekusi model
    process = psutil.Process()

    y_prediction = model.predict(X_validation)
    y_proba = model.predict_proba(X_validation)[:, 1]

    end_time = time.time() # Waktu eksekusi model berakhir

    accuracy = accuracy_score(y_validation, y_prediction) # Menghitung skor akurasi pemodelan
    precision = precision_score(y_validation, y_prediction, pos_label=1) # Menghitung skor presisi pemodelan
    recall = recall_score(y_validation, y_prediction, pos_label=1) # Menghitung skor recall pemodelan
    f1 = f1_score(y_validation, y_prediction, pos_label=1) # Menghitung skor recall pemodelan
    auc = roc_auc_score(y_validation, y_proba) # Menghitung skor AUC
    execution_time = end_time - start_time # Waktu eksekusi

    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1-Score: ", f1)
    print("AUC: ", auc)
    print("Execution Time (s): ", execution_time)
    
    # Evaluasi model random forest dengan feature lengkap
model_final = RandomForestClassifier(n_estimators=100, random_state=1)
model_final.fit(X_train, y_train)

evaluate(model_final, X_validation, y_validation)

# Evaluasi model adaboost dengan feature lengkap
base_estimator = DecisionTreeClassifier(max_depth=1)
model_final = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=50, random_state=1)
model_final.fit(X_train, y_train)

evaluate(model_final, X_validation, y_validation)

# Pengambilan feauture dengan tingkat korelasi tertinggi terhadap hasil colum Heart Disease
correlations = data.corr()['Heart Disease'].sort_values(ascending=False).reset_index()
correlations = correlations[correlations['index'] != 'Heart Disease']

# Menampilkan 5 kolom dengan korelasi tertinggi
top_features = correlations['index'].head(5)
print(top_features)

# Membuat dataset baru untuk dibagi menjadi data latih dan uji dengan mengambil nilai korelasi tertinggi
X = data[top_features]
y = data['Heart Disease']
X_train, X_validation, y_train, y_validation = train_test_split(X,y,test_size=0.20, random_state=1, shuffle=True)

# Merubah nilai pada dataset menjadi distribusi normal
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_validation = scaler.fit_transform(X_validation)

# Evaluasi model random forest dengan feature yang memiliki korelasi tertinggi
model_final = RandomForestClassifier(n_estimators=100, random_state=1)
model_final.fit(X_train, y_train)

evaluate(model_final, X_validation, y_validation)

# Evaluasi model adaboost dengan feature yang memiliki korelasi tertinggi
base_estimator = DecisionTreeClassifier(max_depth=1)
model_final = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=50, random_state=1)
model_final.fit(X_train, y_train)

evaluate(model_final, X_validation, y_validation)