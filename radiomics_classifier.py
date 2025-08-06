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
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
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


    models = {
        name: Pipeline([
            ('var_thresh', VarianceThreshold(threshold=0.01)),
            ('scaler', StandardScaler()),
            ('clf', model)
        ])
        for name, model in models.items()
    }
    return models


def evaluate_model(model, X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(
        model, X, y, cv=skf,
        scoring=["accuracy", "f1_weighted", "roc_auc_ovr"],
        return_train_score=False
    )
    return {
        "accuracy": scores['test_accuracy'].mean(),
        "f1_score": scores['test_f1_weighted'].mean(),
        "roc_auc_ovr": scores['test_roc_auc_ovr'].mean()
    }



# Load your features and labels
csv_file = pd.read_csv("/gpfs/home6/palfken/radiomics_features.csv")
X = csv_file.drop(columns=['case_id', 'tumor_class'])

y = csv_file['tumor_class']
# Encode labels if they're not numeric
if y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)

preprocessing = Pipeline([
    ('var_thresh', VarianceThreshold(threshold=0.01)),
    ('scaler', StandardScaler())
])

# Apply preprocessing
X_processed = preprocessing.fit_transform(X)

models = get_models()

for name, model in models.items():
    results = evaluate_model(model, X_processed, y)
    print(f"Model: {name}")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
    print("-" * 30)