import os, cv2, numpy as np
from skimage.feature import local_binary_pattern, hog
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
import joblib
from feature_extraction import extract_features 
from labels import label_map

DATA_DIR = "data/train"

def load_dataset(folder):
    X, y = [], []
    for class_name in os.listdir(folder):
        class_dir = os.path.join(folder, class_name)
        if not os.path.isdir(class_dir): continue
        for file in os.listdir(class_dir):
            path = os.path.join(class_dir, file)
            feat = extract_features(path)
            if feat is not None:
                X.append(feat); y.append(class_name)
    return np.array(X), np.array(y)

if __name__ == "__main__":
    X_train, y_train = load_dataset(DATA_DIR)
    le = LabelEncoder(); y_train_encoded = le.fit_transform(y_train)

    pca = PCA(n_components=100)

    # --- SVM (Linear) ---
    svm_linear = make_pipeline(StandardScaler(), pca, SVC(kernel="linear", C=1, probability=True, random_state=42))
    scores = cross_val_score(svm_linear, X_train, y_train, cv=5)
    print("SVM (Linear) 5-fold CV scores:", scores)
    print("SVM (Linear) CV mean accuracy:", scores.mean())
    svm_linear.fit(X_train, y_train)
    joblib.dump(svm_linear, "svm_linear_final.pkl")

    # --- SVM (RBF) ---
    svm_rbf = make_pipeline(StandardScaler(), pca, SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42))
    scores_rbf = cross_val_score(svm_rbf, X_train, y_train, cv=5)
    print("\nSVM (RBF) 5-fold CV scores:", scores_rbf)
    print("SVM (RBF) CV mean accuracy:", scores_rbf.mean())
    svm_rbf.fit(X_train, y_train)
    joblib.dump(svm_rbf, "svm_rbf_final.pkl")

    # --- Random Forest ---
    rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
    scores_rf = cross_val_score(rf, X_train, y_train, cv=5)
    print("\nRandom Forest 5-fold CV scores:", scores_rf)
    print("Random Forest CV mean accuracy:", scores_rf.mean())
    rf.fit(X_train, y_train)
    joblib.dump(rf, "rf_model_final.pkl")

    # --- XGBoost ---
    xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.01,
                        subsample=0.8, colsample_bytree=0.8, random_state=42,
                        eval_metric="mlogloss")
    scores_xgb = cross_val_score(xgb, X_train, y_train_encoded, cv=5)
    print("\nXGBoost 5-fold CV scores:", scores_xgb)
    print("XGBoost CV mean accuracy:", scores_xgb.mean())
    xgb.fit(X_train, y_train_encoded)
    joblib.dump((xgb, le), "xgb_model_final.pkl")
