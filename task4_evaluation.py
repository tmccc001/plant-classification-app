import os, cv2, numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tabulate import tabulate
from feature_extraction import extract_features  
from labels import label_map            
import matplotlib, random

# --- Fix font for Chinese characters ---
matplotlib.rcParams['font.family'] = ['Microsoft YaHei', 'SimSun', 'Arial Unicode MS']

TEST_DIR = "data/test"

def load_dataset(folder):
    X, y, paths = [], [], []
    for class_name in os.listdir(folder):
        class_dir = os.path.join(folder, class_name)
        if not os.path.isdir(class_dir): continue
        for file in os.listdir(class_dir):
            path = os.path.join(class_dir, file)
            feat = extract_features(path)
            if feat is not None:
                X.append(feat)
                y.append(class_name)
                paths.append(path)
    return np.array(X), np.array(y), paths

if __name__ == "__main__":
    # Load test set
    X_test, y_test, paths_test = load_dataset(TEST_DIR)

    # Load final trained model
    svm_linear = joblib.load("svm_linear_final.pkl")

    # Predict
    y_pred = svm_linear.predict(X_test)

    # Evaluation
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    print(f"\nTest Accuracy: {acc:.2f} ({sum(y_test==y_pred)}/{len(y_test)} correct)")
    print("\nConfusion Matrix:\n", cm)

    # Format per-class metrics as a bordered table
    df = pd.DataFrame(report).transpose()
    # Fix duplicated names: only English + Chinese
    df.index = [f"{label_map[cls]}" if cls in label_map else cls for cls in df.index]
    print("\nPer-class metrics:\n")
    print(tabulate(df, headers="keys", tablefmt="grid", floatfmt=".2f"))

    # Correct vs incorrect counts
    correct = sum(y_test == y_pred)
    incorrect = len(y_test) - correct
    total = len(y_test)
    print(f"\nCorrect predictions: {correct}")
    print(f"Incorrect predictions: {incorrect}")
    print(f"Total cases: {total}")

    # Qualitative examples: random pick for balance
    correct_indices = [i for i in range(len(y_test)) if y_test[i] == y_pred[i]]
    incorrect_indices = [i for i in range(len(y_test)) if y_test[i] != y_pred[i]]

    # Randomly sample up to 6 correct and 6 incorrect
    selected_correct = random.sample(correct_indices, min(6, len(correct_indices)))
    selected_incorrect = random.sample(incorrect_indices, min(6, len(incorrect_indices)))

    print(f"\nShowing {len(selected_correct)} correct and {len(selected_incorrect)} incorrect examples...\n")

    # Show correct examples
    for i in selected_correct:
        true_label = y_test[i]
        pred_label = y_pred[i]
        img = cv2.imread(paths_test[i])[:,:,::-1]  # BGR → RGB
        plt.imshow(img)
        plt.title(f"Correct | True: {label_map[true_label]} | Pred: {label_map[pred_label]}",
                  fontsize=10, wrap=True)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    # Show incorrect examples
    for i in selected_incorrect:
        true_label = y_test[i]
        pred_label = y_pred[i]
        img = cv2.imread(paths_test[i])[:,:,::-1]  # BGR → RGB
        plt.imshow(img)
        plt.title(f"Incorrect | True: {label_map[true_label]} | Pred: {label_map[pred_label]}",
                  fontsize=10, wrap=True)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
