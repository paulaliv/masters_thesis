import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# Feature preprocessing
def preprocess_features(X):
    # Remove near-zero variance features
    var_thresh = VarianceThreshold(threshold=0.01)
    X = var_thresh.fit_transform(X)


    # Z-score normalization
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X

def get_models():
    models = {
        "XGBoost": XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42
        ),
        "SVM": SVC(kernel="rbf", probability=True, C=1.0, gamma="scale", random_state=42),
        "LogisticRegression": LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000, random_state=42)
    }
    return models


def evaluate_model(model, X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc = cross_val_score(model, X, y, cv=skf, scoring="accuracy").mean()
    f1 = cross_val_score(model, X, y, cv=skf, scoring="f1_weighted").mean()
    auc = cross_val_score(model, X, y, cv=skf, scoring="roc_auc_ovr").mean()

    return {"accuracy": acc, "f1_score": f1, "roc_auc_ovr": auc}


# Load your features and labels
csv_file = pd.read_csv("/gpfs/home6/palfken/radiomics_features.csv")
X = csv_file.drop(columns=['case_id', 'tumor_class'])

y = csv_file['tumor_class']



X_processed = preprocess_features(X)
models = get_models()

for name, model in models.items():
    results = evaluate_model(model, X_processed, y)
    print(f"Model: {name}")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
    print("-" * 30)