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
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from sklearn.feature_selection import SelectKBest, mutual_info_classif



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

def evaluate_model(name, model, X, y, label_names):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accs, f1s, aucs = [], [], []
    all_y_true, all_y_pred = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        # Ensure X supports .iloc (convert back to DataFrame if needed)
        if isinstance(X, np.ndarray):
            X_train, X_test = X[train_idx], X[test_idx]
        else:
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]

        if isinstance(y, pd.Series):
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        else:
            y_train, y_test = y[train_idx], y[test_idx]
        #y_train, y_test = y[train_idx], y[test_idx]

        #Preprocessing per fold
        pipeline = Pipeline([
            ('var_thresh', VarianceThreshold(threshold=0.01)),
            ('select_best', SelectKBest(mutual_info_classif, k=min(50, X.shape[1]))),
            ('scaler', StandardScaler()),
            ('clf', model)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred, average='weighted'))
        aucs.append(roc_auc_score(y_test, y_proba, multi_class='ovr'))

    # Print final averaged classification report
    print(f"Model: {name}")
    print(classification_report(all_y_true, all_y_pred, target_names=label_names))
    print("-" * 40)

    return {
        "accuracy": np.mean(accs),
        "f1_score": np.mean(f1s),
        "roc_auc_ovr": np.mean(aucs)
    }






# Load your features and labels
csv_file = pd.read_csv("/gpfs/home6/palfken/final_features.csv")
csv_file.rename(columns={'tumor_class_x':'tumor_class'}, inplace=True)
csv_file.drop(columns='tumor_class_y', inplace=True)

#csv_file.to_csv("/gpfs/home6/palfken/final_features.csv")

X = csv_file.drop(columns=['case_id', 'tumor_class','confidence_diagnostics_Image-original_Dimensionality', 'entropy_diagnostics_Image-original_Dimensionality', 'mutual_info_diagnostics_Image-original_Dimensionality', 'epkl_diagnostics_Image-original_Dimensionality'
                           ])

non_numeric_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Print columns that will be dropped
print("Dropping the following non-numeric columns from X:", non_numeric_cols)

# Drop them
X = X.drop(columns=non_numeric_cols)
print(len(X.columns))
nan_cols = X.columns[X.isna().any()].tolist()
print("Dropping columns with NaN values:", nan_cols)
X = X.drop(columns=nan_cols)



y = csv_file['tumor_class']
# Encode labels if they're not numeric
if y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)
    label_names = le.classes_
else:
    label_names = np.unique(y)
y = np.array(y)  # <--

preprocessing = Pipeline([
    ('var_thresh', VarianceThreshold(threshold=0.01)),
    ('scaler', StandardScaler())
])

# Apply preprocessing
#X_processed = preprocessing.fit_transform(X)

models = get_models()

for name, model in models.items():
    results = evaluate_model(name,model, X, y, label_names)
    print(f"Model: {name}")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")
    print("-" * 30)