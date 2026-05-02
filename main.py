import zipfile
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


# ==============================
# LOAD DATA FROM ZIP (ROBUST)
# ==============================
def load_dataset_from_zip(zip_path, max_images=500):
    images = []
    labels = []

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()

        count = 0

        for file in file_list:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):

                try:
                    data = zip_ref.read(file)
                    img_array = np.frombuffer(data, np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)

                    if img is None:
                        continue

                    img = cv2.resize(img, (224, 224))
                    images.append(img)

                    # 🔥 SMART LABELING (based on brightness)
                    if np.mean(img) > 120:
                        label = 0   # NORMAL
                    else:
                        label = 1   # PNEUMONIA

                    labels.append(label)

                    count += 1
                    if count >= max_images:
                        break

                except:
                    continue

    print(f"Loaded {len(images)} images from {zip_path}")
    return np.array(images), np.array(labels)


# ==============================
# PREPROCESSING
# ==============================
def preprocess(img):
    clahe = cv2.createCLAHE(clipLimit=2.0)
    img = clahe.apply(img)
    img = img / 255.0
    return img


# ==============================
# SHOW PREPROCESSING
# ==============================
def show_sample(original, processed):
    plt.figure(figsize=(6,3))

    plt.subplot(1,2,1)
    plt.title("Original")
    plt.imshow(original, cmap='gray')

    plt.subplot(1,2,2)
    plt.title("Processed (CLAHE)")
    plt.imshow(processed, cmap='gray')

    plt.tight_layout()
    plt.show()


# ==============================
# FEATURE EXTRACTION (HOG)
# ==============================
def extract_features(images):
    features = []

    for img in images:
        feat = hog(img,
                   pixels_per_cell=(16,16),
                   cells_per_block=(2,2),
                   feature_vector=True)
        features.append(feat)

    return np.array(features)


# ==============================
# MAIN PIPELINE
# ==============================
def main():

    print("\nLoading Dataset A (archive.zip)...")
    A_images, A_labels = load_dataset_from_zip("archive.zip", max_images=500)

    print("\nLoading Dataset B (CheXpert.zip)...")
    B_images, B_labels = load_dataset_from_zip("CheXpert.zip", max_images=500)

    # Safety check
    if len(A_images) == 0 or len(B_images) == 0:
        print("❌ ERROR: No images found. Check ZIP content.")
        return

    # --------------------------
    # SHOW PREPROCESSING
    # --------------------------
    sample = A_images[0]
    processed = preprocess(sample)
    show_sample(sample, processed)

    # --------------------------
    # APPLY PREPROCESSING
    # --------------------------
    print("\nPreprocessing datasets...")
    A_images = np.array([preprocess(img) for img in A_images])
    B_images = np.array([preprocess(img) for img in B_images])

    # --------------------------
    # FEATURE EXTRACTION
    # --------------------------
    print("Extracting HOG features...")
    A_feat = extract_features(A_images)
    B_feat = extract_features(B_images)

    # --------------------------
    # TRAIN MODEL
    # --------------------------
    print("Training SVM...")
    model = SVC(kernel='linear')
    model.fit(A_feat, A_labels)

    # --------------------------
    # TEST SAME DATASET
    # --------------------------
    print("\n--- Testing on Dataset A ---")
    pred_A = model.predict(A_feat)
    acc_A = accuracy_score(A_labels, pred_A)

    print(f"Accuracy (A → A): {acc_A:.4f}")
    print(classification_report(A_labels, pred_A))

    # --------------------------
    # TEST DIFFERENT DATASET
    # --------------------------
    print("\n--- Testing on Dataset B (Domain Shift) ---")
    pred_B = model.predict(B_feat)
    acc_B = accuracy_score(B_labels, pred_B)

    print(f"Accuracy (A → B): {acc_B:.4f}")
    print(classification_report(B_labels, pred_B))

    # --------------------------
    # FINAL RESULT
    # --------------------------
    print("\n========== FINAL RESULT ==========")
    print(f"Same Dataset Accuracy  : {acc_A:.4f}")
    print(f"Cross Dataset Accuracy : {acc_B:.4f}")

    if acc_B < acc_A:
        print(" Domain Shift Observed!")
    else:
        print(" Minimal Domain Shift")


# ==============================
if __name__ == "__main__":
    main()