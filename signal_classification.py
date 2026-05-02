# =========================
# FIX FOR MATPLOTLIB (NO QT ERROR)
# =========================
import matplotlib
matplotlib.use('Agg')   # Safe backend (no GUI issues)

# =========================
# IMPORT LIBRARIES
# =========================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# =========================
# DATASET PATHS
# =========================
base_path = os.path.join(os.getcwd(), "archive_dataset")

diseased_path = os.path.join(base_path, "diseased")
healthy_path = os.path.join(base_path, "healthy")

# =========================
# DEBUG CHECK
# =========================
print("Diseased files:", len(os.listdir(diseased_path)))
print("Healthy files:", len(os.listdir(healthy_path)))

# =========================
# PLOT SIGNALS (ANNEXURE A)
# =========================
def plot_signals(folder1, folder2, samples=5):
    fig, axes = plt.subplots(2, samples, figsize=(15, 6))
    axes = axes.flatten()

    for i, file in enumerate(os.listdir(folder1)[:samples]):
        data = np.loadtxt(os.path.join(folder1, file))
        axes[i].plot(data)
        axes[i].set_title(f'Diseased {i+1}')

    for i, file in enumerate(os.listdir(folder2)[:samples]):
        data = np.loadtxt(os.path.join(folder2, file))
        axes[i + samples].plot(data)
        axes[i + samples].set_title(f'Healthy {i+1}')

    plt.tight_layout()
    plt.savefig("annexure_A_signals.png")   # saved
    print("Saved: annexure_A_signals.png")

plot_signals(diseased_path, healthy_path)

# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(folder, label, limit=200):
    features, labels, files = [], [], []

    for file in os.listdir(folder)[:limit]:
        data = np.loadtxt(os.path.join(folder, file))

        stats = [
            np.mean(data),
            np.var(data),
            skew(data),
            kurtosis(data),
            np.median(data)
        ]

        features.append(stats)
        labels.append(label)
        files.append(file)

    return features, labels, files

# Load data
X_d, y_d, f_d = extract_features(diseased_path, 1)
X_h, y_h, f_h = extract_features(healthy_path, 0)

X = np.array(X_d + X_h)
y = np.array(y_d + y_h)
files = f_d + f_h

# =========================
# DATAFRAME (ANNEXURE B)
# =========================
df = pd.DataFrame(X, columns=['Mean', 'Variance', 'Skewness', 'Kurtosis', 'Median'])
df['Label'] = y
df['Filename'] = files

print("\nFeature DataFrame:\n")
print(df.head())

# Save DataFrame
df.to_csv("annexure_B_features.csv", index=False)
print("Saved: annexure_B_features.csv")

# =========================
# TRAIN MODEL
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# =========================
# PREDICTION
# =========================
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

# =========================
# CONFUSION MATRIX (ANNEXURE C)
# =========================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='coolwarm',
    xticklabels=['Healthy', 'Diseased'],
    yticklabels=['Healthy', 'Diseased']
)

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

plt.savefig("annexure_C_confusion_matrix.png")   # saved
print("Saved: annexure_C_confusion_matrix.png")

# =========================
# DONE
# =========================
print("\nAll outputs generated successfully ")